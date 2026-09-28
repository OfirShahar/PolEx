"""
PolEx pairwise alignment: enumerate candidate alignments from matching polygon edges and
rank them by the pictorial dissimilarity of the fragments' extrapolated bands.
"""

import numpy as np
import torch

from .config import DEFAULT_PAD, DEFAULT_PRESET, PRESETS
from .geometry import compute_edge_alignment, extract_polygon, get_augmented_edges
from .imaging import to_tensor
from .scoring import score_alignment


def find_candidates(target: np.ndarray,
                    src: np.ndarray,
                    target_extrapolation: np.ndarray,
                    src_extrapolation: np.ndarray,
                    min_edge_length: float = PRESETS[DEFAULT_PRESET]['min_edge_length'],
                    min_length_ratio: float = PRESETS[DEFAULT_PRESET]['min_length_ratio'],
                    epsilon_ratio: float = PRESETS[DEFAULT_PRESET]['epsilon_ratio'],
                    angle_threshold_deg: float = PRESETS[DEFAULT_PRESET]['angle_threshold_deg'],
                    smoothing_kernel_size: int = PRESETS[DEFAULT_PRESET]['smoothing_kernel_size'],
                    gap: float = PRESETS[DEFAULT_PRESET]['gap'],
                    pad_by: int = DEFAULT_PAD,
                    device: torch.device = None) -> list:
    """
    Compute ranked candidate alignments of `src` (moving) onto `target` (fixed).

    Args:
        target, src: HxWx4 uint8 BGRA fragment images (alpha > 0 marks the fragment).
        target_extrapolation, src_extrapolation: HxWx4 BGRA extrapolated fragments, in the
            same frame as the corresponding fragment (see extrapolate.py).
        min_edge_length: edges shorter than 1.5 * min_edge_length are discarded.
        min_length_ratio: gamma, the minimal min(Lt,Ls)/max(Lt,Ls) for an admissible edge pair.
        epsilon_ratio: alpha, Ramer-Douglas-Peucker tolerance as a fraction of the perimeter.
        angle_threshold_deg: threshold for merging collinear vertices and for edge augmentation.
        smoothing_kernel_size: Gaussian kernel size used to smooth the fragment mask.
        gap: g, offset (px) between the matched edges, compensating for erosion.
        pad_by: zero-padding added around each image before processing.

    Returns:
        A list of candidates sorted by dissimilarity (best first). Each candidate is a dict with
        'rotation' (degrees, about the source alpha-centroid; positive values rotate
        clockwise as displayed, since the image y-axis points down), 'translation_x', 'translation_y' (pixels, applied after the rotation),
        'score' (lower is better), and the matched 'target_edge' / 'src_edge'.
        Edge coordinates are in the padded frame (offset by pad_by); the transformation itself
        is independent of the padding.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_t = to_tensor(target, pad_by, device)
    src_t = to_tensor(src, pad_by, device)
    target_ext_t = to_tensor(target_extrapolation, pad_by, device)
    src_ext_t = to_tensor(src_extrapolation, pad_by, device)

    poly_kwargs = dict(epsilon_ratio=epsilon_ratio,
                       angle_threshold_deg=angle_threshold_deg,
                       smoothing_kernel_size=smoothing_kernel_size)
    target_poly = extract_polygon(target_t, **poly_kwargs)
    src_poly = extract_polygon(src_t, **poly_kwargs)
    if target_poly is None or src_poly is None:
        return []

    target_edges = get_augmented_edges(target_poly, min_edge_length, angle_threshold_deg)
    src_edges = get_augmented_edges(src_poly, min_edge_length, angle_threshold_deg)

    candidates = []
    for t_edge in target_edges:
        for s_edge in src_edges:
            lt, ls = t_edge['length'], s_edge['length']
            if min(lt, ls) / max(lt, ls) < min_length_ratio:
                continue
            rotation, tx, ty = compute_edge_alignment(target_t, src_t, t_edge, s_edge, gap)
            score = score_alignment(target_t, src_t, target_ext_t, src_ext_t, rotation, tx, ty)
            candidates.append({
                'rotation': -((-rotation + 180.0) % 360.0 - 180.0),  # normalized to (-180, 180]
                'translation_x': float(tx),
                'translation_y': float(ty),
                'score': score,
                'target_edge': t_edge,
                'src_edge': s_edge,
            })

    candidates.sort(key=lambda c: c['score'])
    return candidates
