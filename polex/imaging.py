"""
Image / tensor utilities: conversion to padded BGRA tensors, alpha-centroids and rigid warps.
All images are handled as [1,4,H,W] BGRA float tensors in [0,1].
"""

import math

import cv2
import kornia
import numpy as np
import torch


def load_rgba(path: str) -> np.ndarray:
    """Load an image with its alpha channel (OpenCV BGRA order)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"Expected a 4-channel (BGRA) image with transparency: {path}")
    return img


def to_tensor(image: np.ndarray, pad_by: int, device: torch.device) -> torch.Tensor:
    """HxWx4 uint8 BGRA array -> zero-padded [1,4,H+2p,W+2p] float tensor in [0,1]."""
    padded = np.pad(image, ((pad_by, pad_by), (pad_by, pad_by), (0, 0)), mode='constant', constant_values=0)
    t = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return t / 255.0


def find_centroid(image_t: torch.Tensor):
    """Alpha-weighted centroid (cx, cy) of a [1,4,H,W] tensor, as 0-D tensors."""
    alpha = image_t[:, 3, :, :].reshape(-1)
    H, W = image_t.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(H, device=image_t.device, dtype=image_t.dtype),
                            torch.arange(W, device=image_t.device, dtype=image_t.dtype),
                            indexing='ij')
    m00 = alpha.sum()
    if m00 < 1e-5:
        return (torch.tensor([W / 2], device=image_t.device, dtype=image_t.dtype),
                torch.tensor([H / 2], device=image_t.device, dtype=image_t.dtype))
    return (xs.reshape(-1) * alpha).sum() / m00, (ys.reshape(-1) * alpha).sum() / m00


def warp(image_t: torch.Tensor, angle_deg: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    """Rotate by angle_deg about the alpha-centroid, then translate by (tx, ty)."""
    _, _, H, W = image_t.shape
    angle = angle_deg * (math.pi / 180.0)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    cx, cy = find_centroid(image_t)

    M = torch.zeros((1, 2, 3), device=image_t.device, dtype=image_t.dtype)
    M[:, 0, 0] = cos_a
    M[:, 0, 1] = -sin_a
    M[:, 1, 0] = sin_a
    M[:, 1, 1] = cos_a
    M[:, 0, 2] = cx - cx * cos_a + cy * sin_a + tx
    M[:, 1, 2] = cy - cx * sin_a - cy * cos_a + ty

    return kornia.geometry.transform.warp_affine(image_t, M, dsize=(H, W), mode='bilinear', align_corners=True)


def extrapolated_band(piece_t: torch.Tensor, extrapolation_t: torch.Tensor) -> torch.Tensor:
    """Extrapolation with the original fragment content removed (the 'extrapolated band')."""
    band = extrapolation_t.clone()
    band[(piece_t[:, 3:4] > 0).expand_as(band)] = 0.0
    return band


def tensor_to_rgba(image_t: torch.Tensor) -> np.ndarray:
    """[1,4,H,W] BGRA tensor -> HxWx4 uint8 RGBA array (for visualization)."""
    arr = np.clip(image_t[0].cpu().numpy() * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
