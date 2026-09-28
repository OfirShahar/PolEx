"""
Evaluate PolEx on a split of the pairwise compatibility dataset.

For every pair, the top --max_candidates ranked candidates are evaluated against the ground
truth and stored in a CSV (one row per candidate). A best-of-top-k summary is then printed
(and saved next to the CSV).

Examples:
    # Top-1 optimized hyperparameters (default), full test set
    python evaluate.py --data_dir <dataset>/test --output results/top1.csv

    # Top-10 optimized hyperparameters
    python evaluate.py --data_dir <dataset>/test --preset top10 --output results/top10.csv

    # Re-summarize an existing results CSV
    python evaluate.py --summarize results/top1.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from align_pair import add_hyperparameter_args, get_hyperparameters
from polex import find_candidates
from polex.evaluation import evaluate_alignment, load_pair, load_pairs_csv, summarize_best_of_k


def print_summary(results: pd.DataFrame, k_values, path=None):
    summary = summarize_best_of_k(results, k_values)
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 200):
        print(summary.to_string(index=False))
    print('rotation_error is the absolute angle difference in [0, 360), as reported in the paper; '
          'rotation_error_wrapped is min(d, 360 - d).')
    if path:
        summary.to_csv(path, index=False)
        print(f'Summary saved to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', help='dataset split directory (containing the pairs CSV and pieces/)')
    parser.add_argument('--csv', default=None, help='pairs CSV name inside --data_dir (default: the only CSV there)')
    parser.add_argument('--extrapolations_dir', default=None, help='default: <data_dir>/extrapolated_pieces')
    parser.add_argument('--output', default='results/polex_results.csv', help='per-candidate results CSV')
    parser.add_argument('--max_candidates', type=int, default=10, help='number of top-ranked candidates to evaluate per pair')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 3, 10], help='k values for the best-of-top-k summary')
    parser.add_argument('--num_pairs', type=int, default=None, help='evaluate only the first N pairs')
    parser.add_argument('--seed', type=int, default=0, help='seed for the random patch sampling')
    parser.add_argument('--summarize', metavar='RESULTS_CSV', default=None,
                        help='only print the best-of-top-k summary of an existing results CSV')
    add_hyperparameter_args(parser)
    args = parser.parse_args()

    if args.summarize:
        print_summary(pd.read_csv(args.summarize), args.k)
        return
    if args.data_dir is None:
        parser.error('--data_dir is required')

    torch.manual_seed(args.seed)
    params = get_hyperparameters(args)
    extrapolations_dir = args.extrapolations_dir or os.path.join(args.data_dir, 'extrapolated_pieces')
    pairs = load_pairs_csv(args.data_dir, args.csv)
    if args.num_pairs is not None:
        pairs = pairs.iloc[:args.num_pairs]

    print('Hyperparameters:', params)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    rows = []
    for _, pair_row in tqdm(pairs.iterrows(), total=len(pairs)):
        pair = load_pair(pair_row, args.data_dir, extrapolations_dir)
        candidates = find_candidates(pair['target'], pair['src'],
                                     pair['target_extrapolation'], pair['src_extrapolation'], **params)
        base = {'pair_idx': pair_row['pair_idx'], 'num_candidates': len(candidates)}

        if not candidates:
            rows.append({**base, 'candidate_rank': np.nan})
            continue

        for rank, c in enumerate(candidates[:args.max_candidates]):
            metrics = evaluate_alignment(pair['target'], pair['src'], pair['target_gt'], pair['src_gt'],
                                         c['rotation'], c['translation_x'], c['translation_y'],
                                         pad_by=params['pad_by'])
            rows.append({**base, 'candidate_rank': rank, 'rotation': c['rotation'],
                         'translation_x': c['translation_x'], 'translation_y': c['translation_y'],
                         'score': c['score'], **metrics})

    results = pd.DataFrame(rows)
    results.to_csv(args.output, index=False)
    print(f'Per-candidate results saved to {args.output}')
    print_summary(results, args.k, os.path.splitext(args.output)[0] + '_summary.csv')


if __name__ == '__main__':
    main()
