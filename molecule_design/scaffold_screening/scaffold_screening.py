"""
Murcko scaffold analysis for molecular datasets.

Extracts Murcko scaffolds from a molecular database, ranks them by frequency,
and identifies the most common scaffolds for each binary label class. Useful
for understanding structural diversity and selecting design starting points.

Outputs (timestamped directory under statistics/):
  all_scaffolds.csv            - all unique scaffolds ranked by frequency
  <label>_common_scaffolds.csv - per-label scaffold frequency table
  standardized_data.csv        - full dataset with scaffold columns appended

Usage:
  python scaffold_screening.py   # edit data_path / label_columns in main()
"""

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from datetime import datetime
import os

RDLogger.DisableLog('rdApp.*')  # suppress RDKit C++ info/warning messages


def detect_label_columns(df, smiles_column):
    """
    Automatically identify binary label columns in a DataFrame.

    A column is considered a label column if it is not the SMILES column,
    not 'Standardized_SMILES', not 'Murcko_Scaffold', and contains only
    values drawn from {0, 1} (ignoring NaN).
    """
    excluded = {smiles_column, 'Standardized_SMILES', 'Murcko_Scaffold'}
    label_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            label_cols.append(col)
    return label_cols


def standardize_smiles(smiles):
    """
    Standardize a SMILES string via RDKit canonicalization.

    Returns None if the input cannot be parsed or canonicalized.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def extract_scaffold(smiles, generic=False):
    """
    Extract the Murcko scaffold from a molecule.

    Removes all acyclic side-chains and functional group substituents,
    retaining only the ring systems and the linker atoms between them.
    This is the 'molecular subtraction' step: full molecule minus decorations.

    Parameters
    ----------
    smiles : str
        Standardized SMILES string.
    generic : bool
        If True, also erase atom/bond-type information (all atoms → C,
        all bonds → single).  Useful for shape-based clustering that ignores
        heteroatom identity.

    Returns
    -------
    tuple of (str or None, str or None)
        (bare_scaffold_smiles, attachment_scaffold_smiles)

        bare_scaffold_smiles        : canonical SMILES of the core scaffold used
                                      for deduplication and grouping.
        attachment_scaffold_smiles  : scaffold SMILES with '*' markers at every
                                      position where a side-chain or functional
                                      group was removed — ready for molecular
                                      design (inserting new substituents).
        Either value is None when extraction fails or the molecule has no rings.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            return None, None
        bare_smi = Chem.MolToSmiles(scaffold_mol, isomericSmiles=True)

        # Mark positions where side-chains / FGs were removed with [*]
        att_mol = Chem.ReplaceSidechains(mol, scaffold_mol)
        if att_mol and att_mol.GetNumAtoms() > 0:
            att_smi = Chem.MolToSmiles(att_mol, isomericSmiles=True)
        else:
            att_smi = bare_smi
        return bare_smi, att_smi
    except Exception:
        return None, None


def analyze_database_scaffolds(data_path, smiles_column):
    """
    Extract the Murcko scaffold for every molecule in the dataset.

    Reads a .csv or .xlsx file, standardizes each SMILES, extracts the
    Murcko scaffold, and appends two new columns to the DataFrame:
    - 'Standardized_SMILES': canonical SMILES after standardization
    - 'Murcko_Scaffold'    : canonical SMILES of the extracted scaffold
                             (None for acyclic molecules)

    Parameters
    ----------
    data_path : str
    smiles_column : str

    Returns
    -------
    tuple
        (scaffold_occurrence : defaultdict[str, int]  — scaffold SMILES → count,
         df                  : pd.DataFrame           — enriched DataFrame)

    Raises
    ------
    FileNotFoundError
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)

    scaffold_occurrence = defaultdict(int)
    standardized_list = []
    scaffold_list = []
    attachment_list = []

    for _, row in df.iterrows():
        smiles = row[smiles_column]
        std_smiles = standardize_smiles(smiles)
        if std_smiles is None:
            print(f"Warning: unable to standardize SMILES, skipping: {smiles}")
            standardized_list.append(None)
            scaffold_list.append(None)
            attachment_list.append(None)
            continue
        standardized_list.append(std_smiles)
        bare_smi, att_smi = extract_scaffold(std_smiles)
        scaffold_list.append(bare_smi)
        attachment_list.append(att_smi)
        if bare_smi:
            scaffold_occurrence[bare_smi] += 1

    df['Standardized_SMILES'] = standardized_list
    df['Murcko_Scaffold'] = scaffold_list
    df['Scaffold_with_Attachments'] = attachment_list
    return scaffold_occurrence, df


def analyze_common_scaffolds_by_label(df, label_columns):
    """
    Discover the most common Murcko scaffolds for each label class.

    For every binary label column, the function collects the scaffolds of all
    positive-labelled molecules and ranks them by frequency.  This reveals
    which structural cores are shared among molecules that carry
    the same biological or physicochemical label — directly usable as starting
    points for molecular design.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched DataFrame produced by analyze_database_scaffolds; must contain
        a 'Murcko_Scaffold' column.
    label_columns : list of str
        Binary (0/1) label column names to analyse.

    Returns
    -------
    tuple
        (label_scaffold_counts  : dict[label] -> defaultdict[scaffold_smi, int],
         label_total_molecules  : dict[label] -> int)
    """
    label_scaffold_counts = {}
    label_total_molecules = defaultdict(int)
    # For each unique bare scaffold, store one representative attachment SMILES.
    # We keep the attachment string from the molecule that contributes the most
    # attachment points (most decorated), giving the richest design template.
    label_attachment_rep = {}   # label -> {bare_scaffold: att_smi}

    for label_col in label_columns:
        counts = defaultdict(int)
        att_rep = {}   # bare_scaffold -> (att_smi, n_attachments)
        for _, row in df.iterrows():
            if row[label_col] == 1:
                label_total_molecules[label_col] += 1
                bare_smi = row['Murcko_Scaffold']
                att_smi = row['Scaffold_with_Attachments']
                if bare_smi:
                    counts[bare_smi] += 1
                    n_att = att_smi.count('*') if att_smi else 0
                    if bare_smi not in att_rep or n_att > att_rep[bare_smi][1]:
                        att_rep[bare_smi] = (att_smi, n_att)
        label_scaffold_counts[label_col] = counts
        label_attachment_rep[label_col] = {k: v[0] for k, v in att_rep.items()}

    return label_scaffold_counts, label_total_molecules, label_attachment_rep


def save_statistics(scaffold_occurrence, label_scaffold_counts,
                    label_total_molecules, label_attachment_rep, df,
                    output_dir='statistics'):
    """
    Persist scaffold screening results to CSV files in a timestamped directory.

    Files written
    -------------
    all_scaffolds.csv              : every unique Murcko scaffold in the database,
                                     sorted by overall frequency
    <label>_common_scaffolds.csv   : per-label scaffold table with:
                                       - Murcko_Scaffold     (bare, for grouping)
                                       - Scaffold_with_Attachments  (* at side-chain
                                         removal sites, for molecular design)
                                       - Count / Percentage
    standardized_data.csv          : full enriched DataFrame

    Returns
    -------
    str  — path to the timestamped output directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    total_mols = len(df)

    pd.DataFrame([
        {
            'Murcko_Scaffold': smi,
            'Count': cnt,
            'Percentage (all)': cnt / total_mols * 100 if total_mols > 0 else 0,
        }
        for smi, cnt in sorted(scaffold_occurrence.items(),
                                key=lambda x: x[1], reverse=True)
    ]).to_csv(os.path.join(output_dir, 'all_scaffolds.csv'), index=False)

    for label, counts in label_scaffold_counts.items():
        total = label_total_molecules[label]
        att_rep = label_attachment_rep.get(label, {})
        pd.DataFrame([
            {
                'Murcko_Scaffold': smi,
                'Scaffold_with_Attachments': att_rep.get(smi, smi),
                'Count': cnt,
                'Total Positive Molecules': total,
                'Percentage': cnt / total * 100 if total > 0 else 0,
            }
            for smi, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ]).to_csv(os.path.join(output_dir, f'{label}_common_scaffolds.csv'), index=False)

    df.to_csv(os.path.join(output_dir, 'standardized_data.csv'), index=False)
    return output_dir


def main():
    data_path = r'../../Odor/Data/processed_data/odors_data_all.csv'
    smiles_column = 'SMILES'
    # Set to None for auto-detection of binary (0/1) label columns.
    # Override with an explicit list if needed, e.g.:
    #   label_columns = ['Bitter', 'Sweet', 'Sour', 'Salty', 'Umami', 'Astringent']
    label_columns = None
    top_n = 20  # number of top common scaffolds to print per label

    base_output_dir = os.path.join(os.path.dirname(__file__), 'statistics')

    try:
        print("Step 1 — Extracting Murcko scaffolds (removing side-chains & FGs)...")
        scaffold_occurrence, df = analyze_database_scaffolds(data_path, smiles_column)

        if label_columns is None:
            label_columns = detect_label_columns(df, smiles_column)
            print(f"Auto-detected {len(label_columns)} label column(s): {label_columns}")

        print(f"\nProcessed {len(df)} molecules.")
        print(f"Unique Murcko scaffolds found: {len(scaffold_occurrence)}")

        print(f"\nTop {top_n} most frequent scaffolds (all molecules):")
        for smi, cnt in sorted(scaffold_occurrence.items(),
                                key=lambda x: x[1], reverse=True)[:top_n]:
            pct = cnt / len(df) * 100
            print(f"  [{cnt:5d} | {pct:5.2f}%]  {smi}")

        print("\nStep 2 — Discovering common scaffolds per label...")
        label_scaffold_counts, label_total_molecules, label_attachment_rep = \
            analyze_common_scaffolds_by_label(df, label_columns)

        for label, counts in label_scaffold_counts.items():
            total = label_total_molecules[label]
            att_rep = label_attachment_rep.get(label, {})
            print(f"\nLabel '{label}'  (positive molecules: {total})"
                  f"  —  top {top_n} common scaffolds:")
            for smi, cnt in sorted(counts.items(),
                                   key=lambda x: x[1], reverse=True)[:top_n]:
                pct = cnt / total * 100 if total > 0 else 0
                att = att_rep.get(smi, smi)
                print(f"  [{cnt:5d} | {pct:5.2f}%]  bare: {smi}")
                print(f"           {'':5}    design: {att}")

        saved_dir = save_statistics(
            scaffold_occurrence, label_scaffold_counts,
            label_total_molecules, label_attachment_rep, df, base_output_dir
        )
        print(f"\nResults saved to: {saved_dir}")

    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
