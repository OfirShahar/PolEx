"""
Pictorial part of PolEx (Sec. 4.3 of the paper): dissimilarity between the extrapolated
bands of two fragments under a candidate alignment.
"""

import kornia
import torch
import torchvision.ops as ops

from .imaging import extrapolated_band, warp

MAX_LAB_DISTANCE = 441.7


def _bgra_to_lab(bgra: torch.Tensor) -> torch.Tensor:
    rgb = torch.flip(bgra[:, 0:3], dims=[1])
    return kornia.color.rgb_to_lab(rgb)


def _shared_region(band_a: torch.Tensor, band_b: torch.Tensor):
    """Place both bands on a common canvas and keep only their overlap."""
    _, _, Ha, Wa = band_a.shape
    _, _, Hb, Wb = band_b.shape
    H, W = max(Ha, Hb), max(Wa, Wb)
    out_a = torch.zeros((1, 4, H, W), device=band_a.device, dtype=band_a.dtype)
    out_b = torch.zeros((1, 4, H, W), device=band_a.device, dtype=band_a.dtype)
    out_a[:, :, :Ha, :Wa] = band_a
    out_b[:, :, :Hb, :Wb] = band_b
    overlap = ((out_a[:, 3:4] > 0) & (out_b[:, 3:4] > 0)).expand(-1, 4, -1, -1)
    return out_a * overlap, out_b * overlap


def lab_dissimilarity_random_patches(image_a: torch.Tensor,
                                     image_b: torch.Tensor,
                                     min_patch_size: int = 8,
                                     max_patch_size: int = 32,
                                     stride: int = 6,
                                     p: float = 2.0) -> torch.Tensor:
    """
    Mean LAB distance inside patches of random sizes in [min_patch_size, max_patch_size],
    placed on a grid with the given stride, aggregated with a p-norm over patches and
    normalized by the maximal LAB distance. Returns 1.0 if the images do not overlap.
    """
    eps = 1e-8
    overlap = (image_a[:, 3:4] > 0) & (image_b[:, 3:4] > 0)
    if not overlap.any():
        return torch.tensor([1.0], device=image_a.device, dtype=image_a.dtype)

    diff = _bgra_to_lab(image_a) - _bgra_to_lab(image_b)
    valid = overlap.float()
    delta_e = torch.sqrt((diff * diff).sum(dim=1, keepdim=True)) * valid

    _, _, H, W = delta_e.shape
    xs = torch.arange(0, W - max_patch_size + 1, step=stride, device=image_a.device)
    ys = torch.arange(0, H - max_patch_size + 1, step=stride, device=image_a.device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
    n = grid_x.shape[0]
    sizes = torch.randint(low=min_patch_size, high=max_patch_size + 1, size=(n,), device=image_a.device)

    rois = torch.zeros((n, 5), device=image_a.device, dtype=torch.float32)
    rois[:, 1] = grid_x.float()
    rois[:, 2] = grid_y.float()
    rois[:, 3] = (grid_x + sizes).float()
    rois[:, 4] = (grid_y + sizes).float()

    # Per-patch mean via ROI-align with a 1x1 output, then recover sums with the patch area
    area = sizes.float() ** 2
    sum_delta = ops.roi_align(delta_e, rois, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=-1, aligned=True).view(-1) * area
    sum_valid = ops.roi_align(valid, rois, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=-1, aligned=True).view(-1) * area
    patch_scores = sum_delta / (sum_valid + eps)

    score = (patch_scores ** p).mean().clamp_min(eps) ** (1.0 / p)
    return score / MAX_LAB_DISTANCE


def color_exception_amplifier(region_a: torch.Tensor,
                              region_b: torch.Tensor,
                              unusual_color_threshold: float = 15.0,
                              min_cluster_size: int = 40,
                              mismatch_color_threshold: float = 20.0,
                              mismatch_amp: float = 10.0) -> float:
    """
    Penalizes alignments where a prominent, unusual color in one band is absent from the other.

    In region_a's shared area, the pixel farthest (in LAB) from the mean color defines an
    'unusual' color; if at least min_cluster_size pixels are close to it, and region_b's mean
    color at those pixels differs from it by more than mismatch_color_threshold, the
    dissimilarity is multiplied by mismatch_amp. Otherwise returns 1.0.
    """
    shared = ((region_a[:, 3:4] > 0) & (region_b[:, 3:4] > 0)).view(-1)
    if not shared.any():
        return 1.0
    coords = torch.where(shared)[0]
    if coords.numel() < min_cluster_size:
        return 1.0

    lab_a = _bgra_to_lab(region_a).view(3, -1).permute(1, 0)
    lab_b = _bgra_to_lab(region_b).view(3, -1).permute(1, 0)
    shared_a = lab_a[shared]

    dist = torch.linalg.norm(shared_a - shared_a.mean(dim=0, keepdim=True), dim=1)
    max_idx = torch.argmax(dist)
    if dist[max_idx] < 1.0:  # uniform region
        return 1.0
    unusual = shared_a[max_idx]

    cluster = torch.where(torch.linalg.norm(shared_a - unusual, dim=1) < unusual_color_threshold)[0]
    if cluster.numel() < min_cluster_size:
        return 1.0

    mean_b = lab_b[coords[cluster]].mean(dim=0)
    if torch.linalg.norm(mean_b - unusual).item() > mismatch_color_threshold:
        return mismatch_amp
    return 1.0


def score_alignment(target_t: torch.Tensor,
                    src_t: torch.Tensor,
                    target_ext_t: torch.Tensor,
                    src_ext_t: torch.Tensor,
                    angle_deg: float,
                    tx: float,
                    ty: float) -> float:
    """
    Dissimilarity of a candidate alignment (lower is better).

    The source fragment and its extrapolation are warped by (angle_deg, tx, ty); the LAB
    dissimilarity is computed over the intersection of the two extrapolated bands (with the
    color-exception amplifier), and the number of overlapping pixels between the original
    fragments is added as a penalty for physically-impossible (overlapping) placements.
    """
    device, dtype = target_t.device, target_t.dtype
    angle = torch.tensor([angle_deg], device=device, dtype=torch.float64)
    tx_t = torch.tensor([tx], device=device, dtype=dtype)
    ty_t = torch.tensor([ty], device=device, dtype=dtype)

    warped_src = warp(src_t, angle, tx_t, ty_t)
    warped_src_ext = warp(src_ext_t, angle, tx_t, ty_t)

    region_t, region_s = _shared_region(extrapolated_band(target_t, target_ext_t),
                                        extrapolated_band(warped_src, warped_src_ext))
    lab = lab_dissimilarity_random_patches(region_t, region_s)
    lab = lab * color_exception_amplifier(region_t, region_s)

    # Overlap (in pixels) between the original fragment contents
    _, _, Ht, Wt = target_t.shape
    _, _, Hs, Ws = warped_src.shape
    H, W = max(Ht, Hs), max(Wt, Ws)
    alpha_t = torch.zeros((H, W), device=device, dtype=torch.bool)
    alpha_s = torch.zeros((H, W), device=device, dtype=torch.bool)
    alpha_t[:Ht, :Wt] = target_t[0, 3] > 0
    alpha_s[:Hs, :Ws] = warped_src[0, 3] > 0
    overlap = (alpha_t & alpha_s).sum().float()

    return float((lab + overlap).item())
