"""
Collect latest MPNN / ChemBERTa test metrics into benchmark_summary.csv.

Usage:
    python aggregate_results.py --task odor
    python aggregate_results.py --task taste
    python aggregate_results.py --task odor --extra-csv "DLMOF-Net=../Odor/model/.../overall_results.csv"
"""

import argparse
import os
from pathlib import Path

import pandas as pd


MODEL_PATTERNS = {
    'MPNN': 'mpnn_*/test_results/overall_results.csv',
    'ChemBERTa': 'chemberta_*/test_results/overall_results.csv',
}


def find_latest(pattern: str, base: Path):
    matches = sorted(base.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def parse_extra_csv(spec: str) -> tuple[str, Path]:
    if '=' not in spec:
        raise ValueError(f'Expected NAME=PATH, got: {spec}')
    name, path = spec.split('=', 1)
    return name.strip(), Path(path).resolve()


def main():
    parser = argparse.ArgumentParser(description='Aggregate benchmark test results')
    parser.add_argument('--task', type=str, choices=['odor', 'taste'], default='odor')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument(
        '--extra-csv',
        action='append',
        default=[],
        help='Optional row: ModelName=path/to/overall_results.csv',
    )
    args = parser.parse_args()

    base = Path(args.output_dir or f'./output/{args.task}').resolve()
    if not base.exists():
        print(f'Output directory not found: {base}')
        return

    rows = []
    for model_name, pattern in MODEL_PATTERNS.items():
        path = find_latest(pattern, base)
        if path is None:
            print(f'  [skip] {model_name}: no results found')
            continue

        df = pd.read_csv(path)
        df['Model'] = model_name
        rows.append(df)
        print(f'  [ok]   {model_name}: {path}')

    for spec in args.extra_csv:
        model_name, path = parse_extra_csv(spec)
        if not path.exists():
            print(f'  [skip] {model_name}: not found ({path})')
            continue
        df = pd.read_csv(path)
        df['Model'] = model_name
        rows.append(df)
        print(f'  [ok]   {model_name}: {path}')

    if not rows:
        print('No benchmark results to aggregate.')
        return

    summary = pd.concat(rows, ignore_index=True)
    metric_cols = [c for c in summary.columns if c not in ('Model', 'N_Folds')]
    cols = ['Model'] + [c for c in metric_cols if c in summary.columns]
    summary = summary[cols]

    out_path = base / 'benchmark_summary.csv'
    summary.to_csv(out_path, index=False, float_format='%.4f')
    print(f'\nSummary saved to: {out_path}')
    print(summary.to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
