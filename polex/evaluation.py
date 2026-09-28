"""
Dataset access and pairwise evaluation metrics (Sec. 5 of the paper):
Pairwise Relative Position Score and Pairwise Anchored RMSE (rotation / translation).
"""

import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .imaging import load_rgba


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

def load_pairs_csv(data_dir: str, csv_name: str = None) -> pd.DataFrame:
    """Load the pairs CSV of a dataset split (e.g. test/testset.csv)."""
    if csv_name is None:
        csvs = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
        if len(csvs) != 1:
            raise ValueError(f"Expected exactly one CSV in {data_dir}, found {csvs}; pass the CSV name explicitly.")
        csv_name = csvs[0]
    return pd.read_csv(os.path.join(data_dir, csv_name))


def load_pair(row: pd.Series, data_dir: str, extrapolations_dir: str) -> dict:
    """Fragments, extrapolations and ground-truth transformations of one dataset pair."""
    def extrapolation_path(piece_path):
        name = os.path.splitext(os.path.basename(piece_path))[0]
        return os.path.join(extrapolations_dir, name + '_ext.png')

    def gt(prefix):
        return {'rotation': row[f'{prefix}_gt_rotation'],
                'translation_x': row[f'{prefix}_gt_translation_x'],
                'translation_y': row[f'{prefix}_gt_translation_y']}

    return {
        'target': load_rgba(os.path.join(data_dir, row['target_piece_path'])),
        'src': load_rgba(os.path.join(data_dir, row['src_piece_path'])),
        'target_extrapolation': load_rgba(extrapolation_path(row['target_piece_path'])),
        'src_extrapolation': load_rgba(extrapolation_path(row['src_piece_path'])),
        'target_gt': gt('target'),
        'src_gt': gt('src'),
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _centroid(image: np.ndarray):
    """Integer centroid of the alpha channel."""
    m = cv2.moments(image[:, :, 3])
    if m['m00'] == 0:
        return 0, 0
    return int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])


def _transform(image: np.ndarray, transformation: dict, centroid=None) -> np.ndarray:
    """
    Dataset convention: paste the image at (max(0, tx), max(0, ty)) on a transparent canvas,
    then rotate the canvas by `rotation` degrees (counter-clockwise, OpenCV convention) about
    the alpha-centroid (or about `centroid`, if given).
    """
    tx = max(0, int(transformation['translation_x']))
    ty = max(0, int(transformation['translation_y']))
    piece = Image.fromarray(image)
    canvas = Image.new('RGBA', (piece.width + tx, piece.height + ty), (0, 0, 0, 0))
    canvas.paste(piece, (tx, ty))
    canvas = np.array(canvas)

    if centroid is None:
        centroid = _centroid(canvas)
    M = cv2.getRotationMatrix2D(centroid, transformation['rotation'], 1)
    return cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))


def _mutual_alpha_area(a: np.ndarray, b: np.ndarray) -> int:
    H, W = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    alpha_a = np.zeros((H, W), dtype=bool)
    alpha_b = np.zeros((H, W), dtype=bool)
    alpha_a[:a.shape[0], :a.shape[1]] = a[:, :, 3] > 0
    alpha_b[:b.shape[0], :b.shape[1]] = b[:, :, 3] > 0
    return int(np.logical_and(alpha_a, alpha_b).sum())


def evaluate_alignment(target: np.ndarray,
                       src: np.ndarray,
                       target_gt: dict,
                       src_gt: dict,
                       rotation: float,
                       translation_x: float,
                       translation_y: float,
                       pad_by: int = 300) -> dict:
    """
    Compare a predicted alignment of `src` relative to `target` (as returned by
    polex.find_candidates) with the ground truth.

    The target (anchor) is placed at its ground-truth pose, and the source is placed relative to it
    using the prediction. Returns:
        relative_position_score: |predicted source ∩ ground-truth source| / |ground-truth source|
        rotation_error: absolute rotation difference in degrees, in [0, 360)
        rotation_error_wrapped: min(rotation_error, 360 - rotation_error), in [0, 180]
        translation_error: distance (px) between the predicted and ground-truth source centroids
    """
    # Predicted transformations in the dataset convention (OpenCV rotation sign is opposite)
    pred_src = {'rotation': -rotation, 'translation_x': translation_x, 'translation_y': translation_y}
    pred_tgt = {'rotation': 0, 'translation_x': -translation_x, 'translation_y': -translation_y}

    pad = int(max(abs(pred_src['translation_x'] + pred_tgt['translation_x']),
                  abs(pred_src['translation_y'] + pred_tgt['translation_y']),
                  abs(pred_tgt['translation_x'] - pred_src['translation_x']),
                  abs(pred_tgt['translation_y'] - pred_src['translation_y']),
                  0) + pad_by)
    target_p = cv2.copyMakeBorder(target, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    src_p = cv2.copyMakeBorder(src, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    # Predicted configuration, anchored at the target's ground-truth pose
    pred_target_img = _transform(_transform(target_p, pred_tgt), target_gt)
    pred_src_img = _transform(_transform(src_p, pred_src), target_gt, centroid=_centroid(pred_target_img))

    # Ground-truth configuration, in the same frame
    gt_target_img = _transform(_transform(target_p, target_gt), pred_tgt)
    gt_src_img = _transform(_transform(src_p, src_gt), pred_tgt, centroid=_centroid(gt_target_img))

    relative_position_score = _mutual_alpha_area(pred_src_img, gt_src_img) / int((gt_src_img[:, :, 3] > 0).sum())

    (px_, py_), (gx, gy) = _centroid(pred_src_img), _centroid(gt_src_img)
    translation_error = float(np.sqrt((px_ - gx) ** 2 + (py_ - gy) ** 2))

    pred_rotation = pred_src['rotation'] + target_gt['rotation']
    rotation_error = (abs((360 + pred_rotation) % 360 - (360 + src_gt['rotation']) % 360) + 360) % 360

    return {
        'relative_position_score': float(relative_position_score),
        'rotation_error': float(rotation_error),
        'rotation_error_wrapped': float(min(rotation_error, 360 - rotation_error)),
        'translation_error': translation_error,
    }


def summarize_best_of_k(results: pd.DataFrame, k_values=(1, 3, 10)) -> pd.DataFrame:
    """
    Best-of-top-k summary. `results` has one row per (pair, evaluated candidate) with a
    `candidate_rank` column (0 = best ranked); pairs without candidates have NaN metrics.
    For each pair, the candidate with the highest relative position score among the k best-ranked
    ones is selected, and all its metrics are reported. Averages are over pairs with candidates.
    """
    metrics = ['relative_position_score', 'rotation_error', 'rotation_error_wrapped', 'translation_error']
    valid = results.dropna(subset=['relative_position_score'])
    n_pairs = results['pair_idx'].nunique()

    rows = []
    for k in k_values:
        top_k = valid[valid['candidate_rank'] < k]
        best = top_k.loc[top_k.groupby('pair_idx')['relative_position_score'].idxmax()]
        row = {'k': k, 'pairs_with_candidates': len(best), 'pairs_total': n_pairs}
        row.update(best[metrics].mean().to_dict())
        rows.append(row)
    return pd.DataFrame(rows)
