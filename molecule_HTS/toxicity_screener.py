#!/usr/bin/env python3
"""
Toxicity screening for food additive candidates.

Checks performed:
  1. Ames mutagenicity structural alerts (Kazius et al. 2005 + extended)
  2. Carcinogenicity structural alerts
  3. Reactive / electrophilic groups
  4. Food-specific hazard patterns (nitrosamines, acrylamide, PAH, etc.)
  5. RDKit built-in filters: PAINS, Brenk (undesirable functions)
  6. Physicochemical ADMET properties (MW, LogP, TPSA, HBD, HBA, RotBonds)
  7. Overall safety verdict: PASS / CAUTION / CONCERN

Usage:
    python toxicity_screener.py --input SWEET_unique.csv
    python toxicity_screener.py --input SWEET_unique.csv --smiles_col standardized_smiles --output tox_report.csv
"""

import argparse
import logging
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog('rdApp.*')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuration
CONFIG = {
    'input':      r'./input.csv',
    'smiles_col': 'SMILES',
    'output':     r'input_toxicity_report.csv',
    'output_dir': r'./toxicity',
    # MW upper bound for "plausible small-molecule" food additives
    'mw_max':     1500,
    # LogP range of concern (very lipophilic → bioaccumulation risk)
    'logp_max':   6.0,
    # Minimum TPSA for very low oral bioavailability flag (not a safety concern per se)
    'tpsa_flag':  150,
}

# Structural alert SMARTS
# Each entry: (name, SMARTS, severity)
# severity: 'CONCERN' = strong alert, 'CAUTION' = moderate alert
STRUCTURAL_ALERTS = [
    # Ames mutagenicity (Kazius 2005 + Ashby/Tennant)
    ('Aromatic_nitro',         'c[N+](=O)[O-]',                     'CONCERN'),
    ('Aliphatic_nitro',        '[CX4][N+](=O)[O-]',                  'CONCERN'),
    ('Aromatic_amine',         '[NH2]c',                             'CONCERN'),
    ('Secondary_aromatic_amine','[NH](-c)-c',                        'CONCERN'),
    ('Azo_aromatic',           'c/N=N/c',                            'CONCERN'),
    ('Azo_aliphatic',          '[NX2]=[NX2]',                        'CAUTION'),
    ('Nitroso_aromatic',       'cN=O',                               'CONCERN'),
    ('Nitroso_aliphatic',      '[CX4]N=O',                           'CONCERN'),
    ('Nitrosamine',            '[N;!$(NC=O);!$(Nc)]N=O',             'CONCERN'),
    ('N-Nitrosamine',          '[N;!$(NC=O)]([!H])[N;!$(NC=O)]=O',   'CONCERN'),
    ('Hydrazine',              '[NH][NH2]',                          'CONCERN'),
    ('Hydrazide',              'C(=O)[NH][NH2]',                     'CAUTION'),
    ('Acrylonitrile',          'C=CC#N',                             'CONCERN'),
    ('Acrylamide',             'C=CC(=O)N',                          'CONCERN'),
    ('Propiolactone',          'C1CC(=O)O1',                         'CONCERN'),
    ('Butyrolactone_reactive', 'C1OC1=O',                            'CONCERN'),  # epoxy ketone
    ('Epoxide',                '[C;R1]1O[C;R1]1',                    'CONCERN'),
    ('Alkyl_halide_primary',   '[CH2][Cl,Br,I]',                     'CAUTION'),
    ('Alkyl_halide_activated', '[C;$(C=O),$(C=S),$(C#N)][Cl,Br,I]', 'CONCERN'),
    ('Vinyl_halide',           '[Cl,Br,I]/C=C/',                     'CONCERN'),
    ('Allylic_halide',         '[Cl,Br,I]CC=C',                      'CAUTION'),

    # Carcinogenicity alerts
    ('Polycyclic_aromatic_3ring', 'c1ccc2ccccc2c1',                  'CAUTION'),   # naphthalene core
    ('Polycyclic_aromatic_4ring', 'c1ccc2cc3ccccc3cc2c1',            'CONCERN'),   # pyrene/anthracene core
    ('Benzidine',              'c1ccc(Nc2ccccc2)cc1',                'CONCERN'),
    ('Aromatic_diazo',         'c[N+]#N',                            'CONCERN'),
    ('Thiourea',               '[NH]C(=S)[NH]',                      'CAUTION'),
    ('Urethane_aromatic',      'cNC(=O)O[CX4]',                      'CAUTION'),
    ('Aromatic_aldehyde_mutagenic', 'c[CX3H1]=O',                     'CAUTION'),  # aldehyde only; [CX3H1] excludes COOH/ester/ketone

    # Reactive / electrophilic
    ('Michael_acceptor',       '[CX3]=[CX3][CX3]=[O,N,S]',          'CAUTION'),
    ('Isocyanate',             'N=C=O',                              'CONCERN'),
    ('Isothiocyanate',         'N=C=S',                              'CAUTION'),
    ('Acid_halide',            'C(=O)[Cl,Br,F]',                     'CONCERN'),
    ('Sulfonyl_halide',        'S(=O)(=O)[Cl,Br,F]',                 'CONCERN'),
    ('Peroxide',               'OO',                                  'CAUTION'),
    ('Quinone',                'O=C1C=CC(=O)C=C1',                   'CAUTION'),
    ('Imine_reactive',         '[CX3]=[NX2][H]',                     'CAUTION'),
    ('Aldehyde',               '[CX3H1](=O)[#6]',                    'CAUTION'),
    ('Alpha_beta_aldehyde',    'C=CC=O',                             'CONCERN'),   # enal (mucosal irritant)

    # Organophosphate/organosulfur hazards
    ('Organophosphate_PS',     'P(=S)',                              'CONCERN'),   # pesticide-like
    ('Organophosphate_PO',     'P(=O)([O])[O]',                     'CAUTION'),
    ('Dithiocarbamate',        'NC(=S)S',                            'CAUTION'),

    # Halogenated hazards
    ('Polychlorinated_aromatic','c[Cl].c[Cl].c[Cl]',                 'CONCERN'),   # PCB-like
    ('Trifluoromethyl_aromatic','cC(F)(F)F',                         'CAUTION'),

    # Additional alkylating / reactive groups
    ('Benzyl_halide',          'c[CH2][Cl,Br,I]',                    'CONCERN'),   # strong alkylating; more reactive than simple primary alkyl halide
    ('Alpha_haloketone',       '[Cl,Br,I][CX4]C(=O)',                'CONCERN'),   # alpha-haloketone, potent electrophile
    ('Formaldehyde_releaser',  '[NX3][CH2][OX2H]',                   'CAUTION'),   # N-methylol group
    ('Hydroxamic_acid',        'C(=O)NO',                            'CAUTION'),   # can form reactive intermediates

    # Food-specific hazards
    ('Coumarin_3_substituted', 'O=C1OC2=CC=CC=C2C=C1',              'CAUTION'),   # hepatotoxic
    ('Pulegone_like',          '[CX3](=O)C(/C=C\C)=C',              'CAUTION'),   # hepatotoxic (mint)
    ('Safrole_like',           'C=CCc1ccc2c(c1)OCO2',               'CONCERN'),   # carcinogenic in rodents
    ('Estragole_like',         'C=CCc1ccc(OC)cc1',                   'CAUTION'),   # genotoxic concern
    ('Furan_ring',             'c1ccco1',                            'CAUTION'),   # some furans are genotoxic
    ('Cyanogenic',             '[C;!$(C=O)]#N',                      'CONCERN'),   # cyanide release potential
]

# Compile SMARTS once
_COMPILED_ALERTS = []
for name, smarts, severity in STRUCTURAL_ALERTS:
    pat = Chem.MolFromSmarts(smarts)
    if pat is not None:
        _COMPILED_ALERTS.append((name, pat, severity))
    else:
        logging.warning(f"Invalid SMARTS for alert '{name}': {smarts}")


# RDKit built-in filter catalogs
def _build_filter_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)

_FILTER_CATALOG = _build_filter_catalog()


# Property computation
def compute_properties(mol) -> dict:
    try:
        mw   = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
        heavy = mol.GetNumHeavyAtoms()
        return dict(MW=round(mw,2), LogP=round(logp,2), TPSA=round(tpsa,2),
                    HBD=hbd, HBA=hba, RotBonds=rot, Rings=rings,
                    AromaticRings=arom, HeavyAtoms=heavy)
    except Exception:
        return dict(MW=np.nan, LogP=np.nan, TPSA=np.nan,
                    HBD=np.nan, HBA=np.nan, RotBonds=np.nan,
                    Rings=np.nan, AromaticRings=np.nan, HeavyAtoms=np.nan)


# Alert checking
def check_structural_alerts(mol) -> dict:
    """Returns dict: {alert_name: severity} for all matching alerts."""
    hits = {}
    for name, pat, severity in _COMPILED_ALERTS:
        if mol.HasSubstructMatch(pat):
            hits[name] = severity
    return hits


def check_rdkit_filters(mol) -> list:
    """Returns list of PAINS/Brenk filter names matched."""
    entries = _FILTER_CATALOG.GetMatches(mol)
    return [e.GetDescription() for e in entries]


# Safety classification
def classify_safety(props: dict, alert_hits: dict, rdkit_hits: list) -> tuple:
    """
    Returns (verdict, reasons_list).
    verdict: 'PASS', 'CAUTION', 'CONCERN'
    """
    reasons = []
    max_severity = 'PASS'

    def upgrade(current, new):
        order = {'PASS': 0, 'CAUTION': 1, 'CONCERN': 2}
        return new if order[new] > order[current] else current

    # Structural alerts
    for name, sev in alert_hits.items():
        reasons.append(f"{sev}:{name}")
        max_severity = upgrade(max_severity, sev)

    # PAINS/Brenk
    for hit in rdkit_hits:
        reasons.append(f"CAUTION:PAINS_Brenk({hit})")
        max_severity = upgrade(max_severity, 'CAUTION')

    # Physicochemical flags
    mw   = props.get('MW', np.nan)
    logp = props.get('LogP', np.nan)
    tpsa = props.get('TPSA', np.nan)
    arom = props.get('AromaticRings', 0)

    if not np.isnan(mw) and mw > CONFIG['mw_max']:
        reasons.append(f"CAUTION:MW_high({mw:.0f}Da)")
        max_severity = upgrade(max_severity, 'CAUTION')

    if not np.isnan(logp) and logp > CONFIG['logp_max']:
        reasons.append(f"CAUTION:LogP_high({logp:.1f},bioaccumulation_risk)")
        max_severity = upgrade(max_severity, 'CAUTION')

    if not np.isnan(arom) and arom >= 4:
        reasons.append(f"CAUTION:PAH_like(aromatic_rings={int(arom)})")
        max_severity = upgrade(max_severity, 'CAUTION')

    return max_severity, reasons


# Main screening pipeline
def screen_molecules(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    records = []
    total = len(df)
    interval = max(1, total // 20)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % interval == 0:
            logging.info(f"  Progress: {i}/{total} ({100*i//total}%)")

        smi = row.get(smiles_col, '')
        rec = {smiles_col: smi}

        if not isinstance(smi, str) or not smi.strip():
            rec.update({'MW': np.nan, 'LogP': np.nan, 'TPSA': np.nan,
                        'HBD': np.nan, 'HBA': np.nan, 'RotBonds': np.nan,
                        'Rings': np.nan, 'AromaticRings': np.nan, 'HeavyAtoms': np.nan,
                        'alert_hits': '', 'pains_brenk_hits': '',
                        'concern_alerts': '', 'caution_alerts': '',
                        'n_alerts': 0, 'n_pains_brenk': 0, 'n_concern': 0, 'n_caution': 0,
                        'verdict': 'INVALID', 'reasons': 'invalid_smiles'})
            records.append(rec)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rec.update({'verdict': 'INVALID', 'reasons': 'rdkit_parse_failed',
                        'alert_hits': '', 'pains_brenk_hits': '',
                        'concern_alerts': '', 'caution_alerts': '',
                        'n_alerts': 0, 'n_pains_brenk': 0, 'n_concern': 0, 'n_caution': 0})
            records.append(rec)
            continue

        props      = compute_properties(mol)
        alerts     = check_structural_alerts(mol)
        rdkit_hits = check_rdkit_filters(mol)
        verdict, reasons = classify_safety(props, alerts, rdkit_hits)

        concern_alerts  = [k for k, v in alerts.items() if v == 'CONCERN']
        caution_alerts  = [k for k, v in alerts.items() if v == 'CAUTION']

        rec.update(props)
        rec['alert_hits']      = '; '.join(f"{k}({v})" for k, v in alerts.items())
        rec['pains_brenk_hits'] = '; '.join(rdkit_hits)
        rec['concern_alerts']  = '; '.join(concern_alerts)
        rec['caution_alerts']  = '; '.join(caution_alerts)
        rec['n_alerts']        = len(alerts)
        rec['n_concern']       = len(concern_alerts)
        rec['n_caution']       = len(caution_alerts)
        rec['n_pains_brenk']   = len(rdkit_hits)
        rec['verdict']         = verdict
        rec['reasons']         = ' | '.join(reasons)

        records.append(rec)

    return pd.DataFrame(records)


# Summary stats
def print_summary(result_df: pd.DataFrame, output_path: str):
    total   = len(result_df)
    invalid = (result_df['verdict'] == 'INVALID').sum()
    valid   = total - invalid
    passed  = (result_df['verdict'] == 'PASS').sum()
    caution = (result_df['verdict'] == 'CAUTION').sum()
    concern = (result_df['verdict'] == 'CONCERN').sum()

    logging.info("Toxicity screening summary")
    logging.info(f"Total molecules  : {total:>8,}")
    logging.info(f"Valid structures : {valid:>8,}")
    logging.info(f"  PASS           : {passed:>8,}  ({100*passed/max(valid,1):.1f}%)")
    logging.info(f"  CAUTION        : {caution:>8,}  ({100*caution/max(valid,1):.1f}%)")
    logging.info(f"  CONCERN        : {concern:>8,}  ({100*concern/max(valid,1):.1f}%)")
    logging.info(f"  INVALID        : {invalid:>8,}")
    # Top alert triggers
    all_concerns = result_df[result_df['concern_alerts'] != '']['concern_alerts']
    counter = Counter()
    for cell in all_concerns:
        for a in str(cell).split('; '):
            if a:
                counter[a] += 1
    if counter:
        logging.info("Top CONCERN alerts:")
        for name, cnt in counter.most_common(10):
            logging.info(f"  {name:<40} {cnt:>6,}")

    all_cautions = result_df[result_df['caution_alerts'] != '']['caution_alerts']
    caution_counter = Counter()
    for cell in all_cautions:
        for a in str(cell).split('; '):
            if a:
                caution_counter[a] += 1
    if caution_counter:
        logging.info("Top CAUTION alerts:")
        for name, cnt in caution_counter.most_common(10):
            logging.info(f"  {name:<40} {cnt:>6,}")

    logging.info(f"Full report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Toxicity screening for food additive candidate molecules"
    )
    parser.add_argument('--input',      type=str, default=CONFIG['input'],
                        help='Input CSV/Excel file')
    parser.add_argument('--smiles_col', type=str, default=CONFIG['smiles_col'],
                        help='SMILES column name (default: standardized_smiles)')
    parser.add_argument('--output',     type=str, default=CONFIG['output'],
                        help='Output CSV path')
    parser.add_argument('--output_dir', type=str, default=CONFIG['output_dir'],
                        help='Directory for verdict-split CSV files (default: ./toxicity)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"File not found: {args.input}")
        return

    logging.info(f"Loading: {args.input}")
    if input_path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    # Auto-detect SMILES column
    smiles_col = args.smiles_col
    if smiles_col not in df.columns:
        candidate = next((c for c in df.columns if 'smiles' in c.lower()), None)
        if candidate:
            logging.info(f"Column '{smiles_col}' not found; using '{candidate}'")
            smiles_col = candidate
        else:
            logging.error(f"No SMILES column found. Columns: {list(df.columns)}")
            return

    logging.info(f"Screening {len(df):,} molecules on column '{smiles_col}'...")
    result_df = screen_molecules(df, smiles_col)

    # Merge with original columns (other than SMILES)
    other_cols = [c for c in df.columns if c != smiles_col]
    if other_cols:
        result_df = pd.concat([df[other_cols].reset_index(drop=True),
                               result_df.reset_index(drop=True)], axis=1)

    # Sort: CONCERN first, then CAUTION, then PASS
    order_map = {'CONCERN': 0, 'CAUTION': 1, 'PASS': 2, 'INVALID': 3}
    result_df['_sort'] = result_df['verdict'].map(order_map)
    result_df = result_df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)

    output_path = Path(args.output)
    output_dir  = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(str(output_path), index=False, encoding='utf-8-sig')

    print_summary(result_df, str(output_path))

    # Also save verdict-split files
    for verdict in ('PASS', 'CAUTION', 'CONCERN'):
        subset = result_df[result_df['verdict'] == verdict]
        if not subset.empty:
            p = output_dir / f"{output_path.stem}_{verdict.lower()}.csv"
            subset.to_csv(str(p), index=False, encoding='utf-8-sig')
            logging.info(f"  Saved {verdict}: {p}  ({len(subset):,} molecules)")


if __name__ == '__main__':
    main()
