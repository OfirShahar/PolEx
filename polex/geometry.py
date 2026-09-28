"""
Geometric part of PolEx (Sec. 4.2 of the paper):
polygonal approximation of a fragment, edge augmentation, and computation of the
rigid transformation that aligns a pair of candidate edges.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .imaging import find_centroid


def merge_collinear_vertices(points: np.ndarray, angle_threshold_deg: float) -> np.ndarray:
    """Remove the middle vertex of any three consecutive vertices whose turning angle is below the threshold."""
    if len(points) < 3:
        return points

    pts = points.tolist()
    i = 0
    while i < len(pts) and len(pts) >= 3:
        p0, p1, p2 = pts[i], pts[(i + 1) % len(pts)], pts[(i + 2) % len(pts)]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-8 or n2 < 1e-8:
            angle = 0.0
        else:
            cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            angle = math.degrees(math.acos(cos_a))

        if angle < angle_threshold_deg:
            pts.pop((i + 1) % len(pts))
        else:
            i += 1
        if i > len(pts) * 2:  # safety guard
            break

    return np.array(pts, dtype=points.dtype)


def extract_polygon(image_t: torch.Tensor,
                    epsilon_ratio: float,
                    angle_threshold_deg: float,
                    smoothing_kernel_size: int,
                    morph_kernel_size: int = 15,
                    min_contour_area: float = 100.0):
    """
    Polygonal approximation of a fragment given as a [1,4,H,W] BGRA tensor.

    The alpha mask is morphologically closed, smoothed with a Gaussian and re-binarized.
    The largest external contour is approximated with Ramer-Douglas-Peucker
    (tolerance = epsilon_ratio * perimeter), and nearly-collinear vertices are merged.
    Returns an (N, 2) array of vertices, or None if no valid contour is found.
    """
    mask = (image_t[0, 3] > 0).float()[None, None]

    # Morphological closing (dilation followed by erosion) on the GPU
    kernel = torch.ones((1, 1, morph_kernel_size, morph_kernel_size), dtype=torch.float, device=mask.device)
    dilated = (F.conv2d(mask, kernel, padding=morph_kernel_size // 2) > 0).float()
    closed = (F.conv2d(dilated, kernel, padding=morph_kernel_size // 2) == kernel.numel()).float()
    closed_mask = closed[0, 0].cpu().numpy().astype(np.uint8) * 255

    smoothed = cv2.GaussianBlur(closed_mask, (smoothing_kernel_size, smoothing_kernel_size), 0)
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_contour_area:
        return None

    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return merge_collinear_vertices(polygon, angle_threshold_deg)


def _make_edge(p1: np.ndarray, p2: np.ndarray) -> dict:
    vec = p2 - p1
    return {
        'p1': p1,
        'p2': p2,
        'midpoint': (p1 + p2) / 2.0,
        'angle': math.atan2(vec[1], vec[0]),
        'length': float(np.linalg.norm(vec)),
    }


def _angle_between_edges(e1: dict, e2: dict) -> float:
    diff = abs(e1['angle'] - e2['angle'])
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return math.degrees(diff)


def get_augmented_edges(polygon: np.ndarray,
                        min_edge_length: float,
                        angle_threshold_deg: float,
                        small_edge_ratio: float = 1.5) -> list:
    """
    Multi-scale edge representation of a polygon.

    Base edges connect consecutive vertices. For every triple of consecutive edges
    (e1, e2, e3), if e2 is short and e1, e3 are nearly collinear, an extra edge spanning
    e1.p1 -> e3.p2 is added (without removing the originals). Finally, edges shorter than
    small_edge_ratio * min_edge_length are discarded.
    """
    n = len(polygon)
    if n < 3:
        return []

    base_edges = [_make_edge(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    small_edge_threshold = min_edge_length * small_edge_ratio

    edges = list(base_edges)
    m = len(base_edges)
    for i in range(m):
        e1, e2, e3 = base_edges[i], base_edges[(i + 1) % m], base_edges[(i + 2) % m]
        if e2['length'] >= small_edge_threshold:
            continue
        if _angle_between_edges(e1, e3) < angle_threshold_deg:
            edges.append(_make_edge(e1['p1'], e3['p2']))

    return [e for e in edges if e['length'] >= small_edge_threshold]


def compute_edge_alignment(target_t: torch.Tensor,
                           src_t: torch.Tensor,
                           target_edge: dict,
                           src_edge: dict,
                           gap: float):
    """
    Rigid transformation (rotation [deg], tx, ty) that places the source edge facing the
    target edge, with its midpoint at the target edge midpoint offset by `gap` pixels along
    the target edge's outward normal. The rotation is about the source alpha-centroid.
    """
    mid_t = (target_edge['p1'] + target_edge['p2']) / 2.0
    mid_s = (src_edge['p1'] + src_edge['p2']) / 2.0

    vec_t = target_edge['p2'] - target_edge['p1']
    vec_s = src_edge['p2'] - src_edge['p1']
    theta = (math.atan2(vec_t[1], vec_t[0]) + math.pi) - math.atan2(vec_s[1], vec_s[0])

    # Outward normal of the target edge (pointing away from the target centroid)
    normal = np.array([-vec_t[1], vec_t[0]], dtype=np.float64)
    tcx, tcy = find_centroid(target_t)
    if np.dot(mid_t - np.array([tcx.item(), tcy.item()]), normal) < 0:
        normal = -normal
    norm_len = np.linalg.norm(normal)
    normal = normal / norm_len if norm_len >= 1e-8 else np.zeros(2, dtype=np.float64)

    mid_t_offset = mid_t + gap * normal

    # Rotate the source edge midpoint about the source centroid
    scx, scy = find_centroid(src_t)
    pivot = np.array([scx.item(), scy.item()])
    rel = mid_s - pivot
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    mid_s_rot = pivot + np.array([cos_t * rel[0] - sin_t * rel[1],
                                  sin_t * rel[0] + cos_t * rel[1]])

    tx, ty = mid_t_offset - mid_s_rot
    return math.degrees(theta), tx, ty
