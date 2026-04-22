"""Decode a phone photo of the printed pattern → 1536-d vector.

No SVG required. Data cell positions are computed from constants in
shape_encoder (_DATA_POS), so any decoder with the same constants can
locate data cells from the image alone.

Pipeline
--------
1. Load image. Auto-detect colour vs grayscale:
   - RGB (real photo, cyan on navy): extract cyan = (g+b)/2 - r.
   - Grayscale (simulation, dark symbols on white): use darkness directly.
2. Detect 4 corner fiducial markers via connected-component labelling.
3. DLT homography: SVG coords → image coords.
4. For each data cell in _DATA_POS: project 4 jitter candidates into image
   space, measure ink signal in a small patch → argmax = jit_idx; winning
   signal magnitude is proxy for circle size.
5. Quantile-normalise winning signals across all D cells → size_idx (0..3).
6. Combine size_idx × 4 + jit_idx → 16-level idx → float → vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as nd_label

from .shape_encoder import (
    CANVAS_H,
    CANVAS_W,
    D,
    _DATA_POS,
    _FID_CENTERS,
    _FID_INNER,
    _FID_OUTER,
    _JITTER_BASE,
    _JITTER_LEVELS,
    _JITTER_STEP,
    _N_SIZE,
    _NS,
    idx_to_float,
    size_jit_to_idx,
)

_PATCH_FRAC = 0.4   # patch radius as fraction of jitter step


def _compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.column_stack([pts, np.ones(len(pts))])
    t = (H @ h.T).T
    return t[:, :2] / t[:, 2:3]


def _detect_fiducials(binary: np.ndarray) -> np.ndarray:
    """Find 4 fiducial centres (True = ink) in order TL, TR, BL, BR."""
    h, w = binary.shape
    labelled, n = nd_label(binary)
    if n == 0:
        raise RuntimeError("No ink regions found in image")

    scale_x = w / CANVAS_W
    scale_y = h / CANVAS_H
    expected = np.array(
        [(cx * scale_x, cy * scale_y) for cx, cy in _FID_CENTERS],
        dtype=np.float64,
    )

    sizes = np.bincount(labelled.ravel())[1:]
    ys, xs = np.indices(binary.shape)
    cxs = np.bincount(labelled.ravel(), weights=xs.ravel())[1:] / np.maximum(sizes, 1)
    cys = np.bincount(labelled.ravel(), weights=ys.ravel())[1:] / np.maximum(sizes, 1)

    exp_ring_svg = _FID_OUTER * _FID_OUTER - _FID_INNER * _FID_INNER
    min_size = 0.3 * exp_ring_svg * scale_x * scale_y
    cand_mask = sizes >= min_size
    if cand_mask.sum() < 4:
        raise RuntimeError(
            f"Only {int(cand_mask.sum())} ink components ≥ min_size; need ≥ 4 fiducials"
        )

    idx = np.arange(1, n + 1)
    cand_centres = np.column_stack([cxs[cand_mask], cys[cand_mask]])
    cand_sizes = sizes[cand_mask]

    chosen = []
    for ex, ey in expected:
        d2 = (cand_centres[:, 0] - ex) ** 2 + (cand_centres[:, 1] - ey) ** 2
        order = np.lexsort((-cand_sizes, d2))
        chosen.append(cand_centres[order[0]])

    ordered = np.asarray(chosen, dtype=np.float64)
    if len({tuple(c) for c in ordered}) < 4:
        raise RuntimeError("Fiducial detection matched the same component twice")
    return ordered


def _ink_score(ink: np.ndarray, cx: float, cy: float, patch_r: int) -> float:
    """Mean ink signal in a ±patch_r window. Higher = more ink."""
    h, w = ink.shape
    x0 = max(0, int(cx) - patch_r)
    x1 = min(w, int(cx) + patch_r + 1)
    y0 = max(0, int(cy) - patch_r)
    y1 = min(h, int(cy) + patch_r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.clip(ink[y0:y1, x0:x1], 0, None).mean())


def decode_image(image_path: str | Path) -> np.ndarray:
    """Decode a phone photo of the printed pattern → (D,) float32 vector.

    Supports both real photos (RGB, cyan on navy) and simulation PNGs
    (grayscale, dark symbols on white).
    """
    rgb = np.array(Image.open(image_path).convert("RGB"))
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)

    # Auto-detect colour vs grayscale by checking channel variance.
    is_colour = float((np.abs(g - r) + np.abs(b - r)).mean()) > 5.0

    if is_colour:
        # Real photo: cyan ink on navy background.
        ink = (g + b) * 0.5 - r          # positive where cyan
        binary = ink > 30                  # True = ink
    else:
        # Simulation PNG: dark symbols on white background.
        grey = r                           # r == g == b for L-mode
        ink = 255.0 - grey                 # positive where dark ink
        binary = grey < 128               # True = ink

    img_fid = _detect_fiducials(binary)
    svg_fid = np.array(_FID_CENTERS, dtype=np.float64)
    H = _compute_homography(svg_fid, img_fid)

    svg_unit = _apply_homography(H, np.array([[0.0, 0.0], [1.0, 0.0]]))
    px_per_svg = float(np.linalg.norm(svg_unit[1] - svg_unit[0]))
    patch_r = max(2, int(round(_PATCH_FRAC * _JITTER_STEP * px_per_svg)))

    offsets = np.array(
        [(ix * _JITTER_STEP + _JITTER_BASE, iy * _JITTER_STEP + _JITTER_BASE)
         for iy in range(_JITTER_LEVELS)
         for ix in range(_JITTER_LEVELS)],
        dtype=np.float64,
    )

    jit_ids = np.zeros(D, dtype=np.int32)
    areas = np.zeros(D, dtype=np.float64)

    for i in range(D):
        bx, by = float(_DATA_POS[i, 0]), float(_DATA_POS[i, 1])
        svg_cands = np.column_stack([
            np.full(len(offsets), bx) + offsets[:, 0],
            np.full(len(offsets), by) + offsets[:, 1],
        ])
        img_cands = _apply_homography(H, svg_cands)
        scores = np.array([_ink_score(ink, cx, cy, patch_r) for cx, cy in img_cands])
        j = int(np.argmax(scores))
        jit_ids[i] = j
        areas[i] = float(scores[j])

    # Quantile-normalise winning scores → size_idx (0..3).
    order = np.argsort(areas, kind="stable")
    size_ids = np.zeros(D, dtype=np.int32)
    per_bin = D // _N_SIZE
    for rank, cell_i in enumerate(order):
        size_ids[cell_i] = min(_N_SIZE - 1, rank // per_bin)

    vector = np.zeros(D, dtype=np.float32)
    for i in range(D):
        vector[i] = idx_to_float(size_jit_to_idx(int(size_ids[i]), int(jit_ids[i])))

    return vector
