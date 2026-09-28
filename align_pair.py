"""
Align a single pair of fragments with PolEx and report the ranked candidate alignments.

Example:
    python align_pair.py --target frag_a.png --src frag_b.png \
        --target_ext frag_a_ext.png --src_ext frag_b_ext.png --top_k 5 --visualize out.png
"""

import argparse
import json

import cv2
import numpy as np
import torch

from polex import DEFAULT_PAD, DEFAULT_PRESET, PRESETS, find_candidates, load_rgba
from polex.imaging import tensor_to_rgba, to_tensor, warp


def add_hyperparameter_args(parser):
    """Preset selection plus optional per-hyperparameter overrides."""
    parser.add_argument('--preset', choices=sorted(PRESETS), default=DEFAULT_PRESET,
                        help='hyperparameter preset (optimized for top-1, top-3 or top-10 candidates)')
    for name, value in PRESETS[DEFAULT_PRESET].items():
        parser.add_argument(f'--{name}', type=type(value), default=None, help=f'override the preset value of {name}')
    parser.add_argument('--pad_by', type=int, default=DEFAULT_PAD, help='zero-padding around each fragment (px)')


def get_hyperparameters(args) -> dict:
    params = dict(PRESETS[args.preset])
    for name in params:
        if getattr(args, name) is not None:
            params[name] = getattr(args, name)
    params['pad_by'] = args.pad_by
    return params


def render_alignment(target, src, candidate, pad_by):
    """RGBA image of the target with the source overlaid at the candidate alignment, cropped to content."""
    device = torch.device('cpu')
    target_t = to_tensor(target, pad_by, device)
    src_t = to_tensor(src, pad_by, device)
    warped = warp(src_t,
                  torch.tensor([candidate['rotation']], dtype=torch.float64),
                  torch.tensor([candidate['translation_x']]),
                  torch.tensor([candidate['translation_y']]))
    a, b = tensor_to_rgba(target_t), tensor_to_rgba(warped)
    H, W = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    b_alpha = b[:, :, 3:4].astype(np.float32) / 255.0
    region = canvas[:b.shape[0], :b.shape[1]]
    region[:] = (b * b_alpha + region * (1 - b_alpha)).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(canvas[:, :, 3])
    return canvas[y:y + h, x:x + w]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', required=True, help='target (fixed) fragment PNG with transparency')
    parser.add_argument('--src', required=True, help='source (moving) fragment PNG with transparency')
    parser.add_argument('--target_ext', required=True, help='extrapolated target fragment (see extrapolate.py)')
    parser.add_argument('--src_ext', required=True, help='extrapolated source fragment (see extrapolate.py)')
    parser.add_argument('--top_k', type=int, default=5, help='number of candidates to report')
    parser.add_argument('--output', default=None, help='optional JSON file for the top-k candidates')
    parser.add_argument('--visualize', default=None, help='optional PNG file showing the top-k alignments side by side')
    parser.add_argument('--seed', type=int, default=0, help='seed for the random patch sampling')
    add_hyperparameter_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    target, src = load_rgba(args.target), load_rgba(args.src)
    params = get_hyperparameters(args)
    candidates = find_candidates(target, src, load_rgba(args.target_ext), load_rgba(args.src_ext), **params)

    if not candidates:
        print('No admissible candidate alignments were found (try a lower --min_length_ratio or --min_edge_length).')
        return

    top = [{'rank': i,
            'rotation': c['rotation'],
            'translation_x': c['translation_x'],
            'translation_y': c['translation_y'],
            'score': c['score']} for i, c in enumerate(candidates[:args.top_k])]

    print(f'{len(candidates)} candidate alignments; top {len(top)}:')
    print(f"{'rank':>4} {'rotation[deg]':>14} {'tx[px]':>9} {'ty[px]':>9} {'dissimilarity':>14}")
    for c in top:
        print(f"{c['rank']:>4} {c['rotation']:>14.2f} {c['translation_x']:>9.2f} {c['translation_y']:>9.2f} {c['score']:>14.5f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'hyperparameters': params, 'candidates': top}, f, indent=2)

    if args.visualize:
        renders = [render_alignment(target, src, c, args.pad_by) for c in candidates[:args.top_k]]
        H = max(r.shape[0] for r in renders)
        row = np.concatenate([np.pad(r, ((0, H - r.shape[0]), (0, 10), (0, 0))) for r in renders], axis=1)
        cv2.imwrite(args.visualize, cv2.cvtColor(row, cv2.COLOR_RGBA2BGRA))


if __name__ == '__main__':
    main()
