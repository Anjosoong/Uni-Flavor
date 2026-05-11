#!/usr/bin/env python3
"""
COCONUT dataset multi-stage filter.

Pipeline stages:
  Stage 1  Pre-screen   — drop empty / non-string / obviously invalid entries
  Stage 2  Parse        — reject molecules that RDKit cannot parse
  Stage 3  Organic      — reject inorganic compounds (no carbon atoms)
  Stage 4  Desalt       — keep the largest fragment for multi-component SMILES
  Stage 5  Canonicalize — output RDKit canonical SMILES
  Stage 6  Size filter  — reject molecules exceeding any of:
                            heavy_atoms      > max_heavy_atoms  (default 120)
                            mw               > max_mw           (default 1500 Da)
                            rotatable_bonds  > max_rot_bonds    (default 28)
                            max_ring_size    > max_ring_size    (default None)
  Stage 7  Dedup        — deduplicate by canonical SMILES (default: enabled)

Outputs:
  {output_dir}/coconut_filtered.csv   — molecules that passed all filters
  {output_dir}/coconut_removed.csv    — rejected molecules with filter_reason column
  {output_dir}/filter_stats.json      — per-stage statistics
"""

import json
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

input_csv        = r"./coconut_csv_lite-03-2026.csv"
output_dir       = r"./filtered_coconut"
smiles_col       = 'canonical_smiles'
max_heavy_atoms  = 120    # keeps mid-sized glycosides/peptides, blocks very large molecules
max_mw           = 1500.0 # keeps non-oversized glycosides/peptides
max_rot_bonds    = 28     # blocks extreme long-chain / highly flexible molecules
max_ring_size    = None   # upper limit on ring size (None = no limit)
dedup            = True   # deduplicate by canonical SMILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Filter & normalise COCONUT dataset for UniMol / Mordred processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',           type=str,   default=str(input_csv))
    parser.add_argument('--output-dir',      type=str,   default=str(output_dir))
    parser.add_argument('--smiles-col',      type=str,   default=smiles_col)
    parser.add_argument('--max-heavy-atoms', type=int,   default=max_heavy_atoms,
                        help='Max heavy atom count')
    parser.add_argument('--max-mw',          type=float, default=max_mw,
                        help='Max molecular weight (Da)')
    parser.add_argument('--max-rot-bonds',   type=int,   default=max_rot_bonds,
                        help='Max rotatable bond count (prevents extreme topological indices)')
    parser.add_argument('--max-ring-size',   type=int,   default=max_ring_size,
                        help='Max ring size allowed (None = no limit)')
    parser.add_argument('--no-dedup',        action='store_true',
                        help='Skip deduplication by canonical SMILES')
    parser.add_argument('--no-removed',      action='store_true',
                        help='Do not save removed molecules CSV')
    return parser.parse_args()


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest fragment (by heavy atom count) from a disconnected mol."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def _max_ring_size(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    return max((len(r) for r in rings), default=0)


def process_molecule(smi_raw: str, cfg: dict):
    """
    Process one SMILES string through all filter stages.

    Returns
    -------
    canonical_smi : str | None   — normalised SMILES (None if rejected)
    props         : dict         — computed properties (heavy_atoms, mw, rot_bonds, max_ring)
    reason        : str | None   — filter reason (None = passed)
    """
    props = dict(heavy_atoms=None, mw=None, rot_bonds=None, max_ring=None)

    # Stage 1 — pre-screen
    if not isinstance(smi_raw, str) or not smi_raw.strip():
        return None, props, 'empty_or_non_string'

    smi = smi_raw.strip()
    if len(smi) < 2:
        return None, props, 'too_short'

    # Stage 2 — RDKit parse
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, props, 'invalid_smiles'

    # Stage 3 — organic check (must contain carbon)
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if 6 not in atoms:
        return None, props, 'non_organic'

    # Stage 4 — salt removal (keep largest fragment)
    if '.' in smi:
        mol = _largest_fragment(mol)

    # Stage 5 — canonical SMILES
    canonical = Chem.MolToSmiles(mol, canonical=True)
    if not canonical:
        return None, props, 'canonicalization_failed'

    # Reparse canonical to ensure consistency
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None, props, 'canonical_reparse_failed'

    # Compute properties
    heavy    = mol.GetNumHeavyAtoms()
    mw       = Descriptors.ExactMolWt(mol)
    rot      = rdMolDescriptors.CalcNumRotatableBonds(mol)
    max_ring = _max_ring_size(mol)

    props = dict(heavy_atoms=heavy, mw=round(mw, 4), rot_bonds=rot, max_ring=max_ring)

    # Stage 6 — size filters
    if heavy > cfg['max_heavy']:
        return canonical, props, f'heavy_atoms>{cfg["max_heavy"]}'
    if mw > cfg['max_mw']:
        return canonical, props, f'mw>{cfg["max_mw"]:.0f}'
    if rot > cfg['max_rot']:
        return canonical, props, f'rot_bonds>{cfg["max_rot"]}'
    if cfg['max_ring'] is not None and max_ring > cfg['max_ring']:
        return canonical, props, f'ring_size>{cfg["max_ring"]}'

    return canonical, props, None


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        max_heavy = args.max_heavy_atoms,
        max_mw    = args.max_mw,
        max_rot   = args.max_rot_bonds,
        max_ring  = args.max_ring_size,
    )

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Total rows: {len(df):,}  |  columns: {list(df.columns)}")

    if args.smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{args.smiles_col}' not found. "
                       f"Available: {list(df.columns)}")

    smiles_raw = df[args.smiles_col].tolist()

    canonical_list  = []
    heavy_list      = []
    mw_list         = []
    rot_list        = []
    max_ring_list   = []
    reason_list     = []

    for smi in tqdm(smiles_raw, desc='Processing', unit='mol'):
        canon, props, reason = process_molecule(smi, cfg)
        canonical_list.append(canon)
        heavy_list.append(props['heavy_atoms'])
        mw_list.append(props['mw'])
        rot_list.append(props['rot_bonds'])
        max_ring_list.append(props['max_ring'])
        reason_list.append(reason)

    df['canonical_smiles'] = canonical_list
    df['heavy_atoms']      = heavy_list
    df['mw']               = mw_list
    df['rot_bonds']        = rot_list
    df['max_ring']         = max_ring_list
    df['_reason']          = reason_list

    mask_pass = df['_reason'].isna()
    df_pass   = df[mask_pass].drop(columns=['_reason']).reset_index(drop=True)
    df_remove = df[~mask_pass].rename(columns={'_reason': 'filter_reason'}).reset_index(drop=True)

    # Stage 7 — deduplication by canonical SMILES
    n_before_dedup = len(df_pass)
    if not args.no_dedup:
        df_pass = df_pass.drop_duplicates(subset=['canonical_smiles']).reset_index(drop=True)
    n_dedup = n_before_dedup - len(df_pass)

    reason_counts = df_remove['filter_reason'].value_counts().to_dict()
    stats = {
        'input_total':       len(df),
        'passed':            len(df_pass),
        'removed_by_filter': len(df_remove),
        'removed_by_dedup':  n_dedup,
        'total_removed':     len(df) - len(df_pass),
        'removal_rate_pct':  round(100.0 * (len(df) - len(df_pass)) / len(df), 2),
        'filter_reasons':    reason_counts,
        'filters_applied': {
            'max_heavy_atoms':  cfg['max_heavy'],
            'max_mw':           cfg['max_mw'],
            'max_rot_bonds':    cfg['max_rot'],
            'max_ring_size':    cfg['max_ring'],
            'dedup_by_canonical': not args.no_dedup,
        },
    }
    if len(df_pass) > 0:
        stats['property_stats_passed'] = {
            'heavy_atoms': {
                'min': int(df_pass['heavy_atoms'].min()),
                'median': float(df_pass['heavy_atoms'].median()),
                'mean': round(float(df_pass['heavy_atoms'].mean()), 1),
                'max': int(df_pass['heavy_atoms'].max()),
            },
            'mw': {
                'min': round(float(df_pass['mw'].min()), 1),
                'median': round(float(df_pass['mw'].median()), 1),
                'max': round(float(df_pass['mw'].max()), 1),
            },
            'rot_bonds': {
                'max': int(df_pass['rot_bonds'].max()),
                'mean': round(float(df_pass['rot_bonds'].mean()), 1),
            },
        }

    out_filtered = output_dir / 'coconut_filtered.csv'
    out_removed  = output_dir / 'coconut_removed.csv'
    out_stats    = output_dir / 'filter_stats.json'

    df_pass.to_csv(out_filtered, index=False)

    if not args.no_removed and len(df_remove) > 0:
        df_remove.to_csv(out_removed, index=False)

    with open(out_stats, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nFilter summary:")
    print(f"  Input total         : {len(df):>10,}")
    print(f"  Removed by filters  : {len(df_remove):>10,}")
    print(f"  Removed by dedup    : {n_dedup:>10,}")
    print(f"  Passed (final)      : {len(df_pass):>10,}  "
          f"({100 - stats['removal_rate_pct']:.1f}%)")
    print(f"\n  Filter reasons:")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<35s}: {cnt:>8,}")
    if 'property_stats_passed' in stats:
        ha = stats['property_stats_passed']['heavy_atoms']
        mw = stats['property_stats_passed']['mw']
        rb = stats['property_stats_passed']['rot_bonds']
        print(f"\n  Passed properties:")
        print(f"    heavy_atoms  min={ha['min']}  median={ha['median']}  "
              f"mean={ha['mean']}  max={ha['max']}")
        print(f"    mw           min={mw['min']}  median={mw['median']}  max={mw['max']}")
        print(f"    rot_bonds    mean={rb['mean']}  max={rb['max']}")
    print(f"\n  Output → {out_filtered}")
    print(f"  Stats  → {out_stats}")


if __name__ == '__main__':
    main()
