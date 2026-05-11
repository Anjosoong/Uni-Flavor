"""
Functional group analysis pipeline for molecular datasets.

Detects functional groups in each molecule using priority-ordered SMARTS
matching (composite groups before elementary), counts their occurrence across
the full dataset and per binary label class, and writes timestamped CSV reports.

Outputs (timestamped directory under statistics/):
  total_statistics.csv   - overall group occurrence counts with design templates
  <label>_statistics.csv - per-label frequency and percentage
  standardized_data.csv  - full dataset with canonical SMILES and Has_<group> columns

Usage:
  python analyze_molecular_groups.py   # edit data_path / label_columns in main()
"""

import pandas as pd
from rdkit import Chem, RDLogger
from collections import defaultdict
import os
from functional_groups import group_patterns, group_priorities, group_design_smiles

RDLogger.DisableLog('rdApp.*')  # suppress RDKit C++ info/warning messages

def detect_label_columns(df, smiles_column):
    """
    Automatically identify binary label columns in a DataFrame.

    A column is considered a label column if it is not the SMILES column,
    not 'Standardized_SMILES', and contains only values drawn from {0, 1}
    (ignoring NaN).  This makes the script agnostic to the number of label
    columns (e.g. 6 taste labels or 138 odor labels).
    """
    label_cols = []
    for col in df.columns:
        if col in (smiles_column, 'Standardized_SMILES'):
            continue
        if col.startswith('Has_'):  # skip indicator columns added by analyze_database_functional_groups
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            label_cols.append(col)
    return label_cols

def standardize_smiles(smiles):
    """
    Standardize a SMILES string via RDKit canonicalization.

    Steps:
    1. Parse the input into an RDKit Mol object.
    2. Canonicalize the molecular graph (hydrogen atoms are retained to
       preserve functional-group detectability, e.g. primary amines).
    3. Return the canonical isomeric SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
        if mol is None:
            return None

        standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return standardized_smiles
    except:
        return None

def find_all_functional_groups(smiles):
    """
    Identify all functional groups present in a molecule from its SMILES string.

    Groups are detected in descending priority order (high -> medium -> low).
    Once a set of atoms is claimed by a higher-priority group, those atoms are
    excluded from subsequent lower-priority matches, preventing redundant annotation.
    Each distinct group type is recorded at most once per molecule.
    """
    try:
        standardized_smiles = standardize_smiles(smiles)
        if standardized_smiles is None:
            return []

        mol = Chem.MolFromSmiles(standardized_smiles)
        if mol is None:
            return []

        detected_groups = []
        occupied_atoms = set()  # atom indices already claimed by higher-priority groups

        for priority in ['high', 'medium', 'low']:
            for group_name in group_priorities[priority]:
                if group_name in group_patterns:
                    pattern = group_patterns[group_name]
                    try:
                        query_mol = Chem.MolFromSmarts(pattern)
                        if query_mol is not None:
                            matches = mol.GetSubstructMatches(query_mol)
                            for match in matches:
                                atoms_available = all(
                                    atom_idx not in occupied_atoms
                                    for atom_idx in match
                                )
                                if atoms_available and len(match) > 0:
                                    detected_groups.append(group_name)
                                    occupied_atoms.update(match)
                                    break  # record each group type at most once
                    except Exception as e:
                        print(f"Error processing functional group '{group_name}': {e}")
                        continue  # skip invalid pattern; proceed with remaining groups

        return detected_groups
    except Exception as e:
        print(f"Error during functional group detection: {e}")
        return []

def analyze_database_functional_groups(data_path, smiles_column):
    """
    Enumerate and count functional groups across all molecules in a dataset.

    Reads a .xlsx or .csv file, standardizes each SMILES entry, detects
    functional groups, and appends binary indicator columns (Has_<group>)
    to the DataFrame.

    Raises FileNotFoundError if the specified data file does not exist.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    all_groups = set()
    group_occurrence = defaultdict(int)
    standardized_smiles_list = []
    groups_per_molecule = []  # functional group list for each molecule

    for _, row in df.iterrows():
        smiles = row[smiles_column]
        standardized_smiles = standardize_smiles(smiles)
        if standardized_smiles is None:
            print(f"Warning: unable to standardize SMILES, skipping: {smiles}")
            standardized_smiles_list.append(None)
            groups_per_molecule.append([])
            continue
        standardized_smiles_list.append(standardized_smiles)
        groups = find_all_functional_groups(standardized_smiles)
        groups_per_molecule.append(groups)
        all_groups.update(groups)
        for group in groups:
            group_occurrence[group] += 1

    df['Standardized_SMILES'] = standardized_smiles_list

    for group_name in all_groups:
        df[f'Has_{group_name}'] = [
            1 if group_name in mol_groups else 0
            for mol_groups in groups_per_molecule
        ]

    return all_groups, group_occurrence, df

def analyze_groups_by_label(data_path, smiles_column, label_columns, all_groups):
    """
    Compute per-label functional group frequency statistics.

    For each binary label column, counts the number of molecules that contain
    each detected functional group and records the total molecule count per label.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    group_stats = defaultdict(lambda: defaultdict(int))
    label_total_molecules = defaultdict(int)  # total molecule count per label

    for _, row in df.iterrows():
        for label_col in label_columns:
            if row[label_col] == 1:
                label_total_molecules[label_col] += 1

    for _, row in df.iterrows():
        smiles = row[smiles_column]
        standardized_smiles = standardize_smiles(smiles)
        if standardized_smiles:
            groups = find_all_functional_groups(standardized_smiles)
            for label_col in label_columns:
                if row[label_col] == 1:
                    for group in groups:
                        group_stats[label_col][group] += 1

    return group_stats, label_total_molecules

def save_statistics(group_occurrence, group_stats, label_total_molecules, df, output_dir='statistics'):
    """
    Persist functional group statistics to CSV files.

    A timestamped subdirectory is created under output_dir to avoid filename
    collisions across multiple analysis runs.  Three file types are written:
    - total_statistics.csv  : overall group occurrence counts
    - <label>_statistics.csv: per-label group frequency and percentage
    - standardized_data.csv : full DataFrame with canonical SMILES and
                              binary indicator columns
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    total_stats = pd.DataFrame([
        {
            'Functional Group': group,
            'Design_Pattern': group_design_smiles.get(group, ''),
            'Count': count,
        }
        for group, count in sorted(group_occurrence.items(), key=lambda x: x[1], reverse=True)
    ])
    total_stats.to_csv(os.path.join(output_dir, 'total_statistics.csv'), index=False)

    for label, groups in group_stats.items():
        total_molecules = label_total_molecules[label]
        label_stats = pd.DataFrame([
            {
                'Functional Group': group,
                'Design_Pattern': group_design_smiles.get(group, ''),
                'Molecules with Group': count,
                'Total Molecules': total_molecules,
                'Percentage': (count / total_molecules * 100) if total_molecules > 0 else 0,
            }
            for group, count in sorted(groups.items(), key=lambda x: x[1], reverse=True)
        ])
        label_stats.to_csv(os.path.join(output_dir, f'{label}_statistics.csv'), index=False)

    df.to_csv(os.path.join(output_dir, 'standardized_data.csv'), index=False)
    return output_dir

def main():
    data_path = r'../../Odor/Data/processed_data/odors_data_all.csv'
    smiles_column = 'SMILES'
    label_columns = None

    base_output_dir = os.path.join(os.path.dirname(__file__), 'statistics')

    try:
        print("Analyzing functional groups across all molecules in the database...")
        all_groups, group_occurrence, df = analyze_database_functional_groups(data_path, smiles_column)

        if label_columns is None:
            label_columns = detect_label_columns(df, smiles_column)
            print(f"Auto-detected {len(label_columns)} label column(s): {label_columns}")

        print(f"\nSuccessfully analyzed {len(df)} molecules.")
        print("\nFunctional groups detected in the database (sorted by occurrence):")
        for group, count in sorted(group_occurrence.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(df) * 100
            print(f"  {group}: {count} ({percentage:.2f}%)")

        print("\nAnalyzing per-label functional group distribution...")
        group_stats, label_total_molecules = analyze_groups_by_label(
            data_path, smiles_column, label_columns, all_groups
        )

        for label, groups in group_stats.items():
            total_molecules = label_total_molecules[label]
            print(f"\nLabel '{label}' statistics (total molecules: {total_molecules}):")
            for group, count in sorted(groups.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_molecules * 100) if total_molecules > 0 else 0
                print(f"  {group}: {count} molecules ({percentage:.2f}%)")

        saved_dir = save_statistics(group_occurrence, group_stats, label_total_molecules, df, base_output_dir)
        print(f"\nStatistics saved to: {saved_dir}")

    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()