"""Decode a phone photo / flat scan of the printed pattern → 1536-d vector.

Pipeline (v2: size + 2×2 jitter)
--------------------------------
1. Load image → grayscale → threshold (dark symbols on light shirt).
2. Detect 4 corner fiducial markers (large dark rings) via connected-component
   size filtering.
3. Compute the SVG→image homography from the 4 fiducial centres.
4. For each of the 1536 data cells (read from the companion SVG): project the
   4 candidate jitter positions into image space and measure continuous
   darkness in a small patch around each. Argmax → jit_idx; the winning
   darkness magnitude is a proxy for the cell's ink area (size channel).
5. Quantile-normalise winning darknesses across all 1536 cells → size_idx per
   cell (0..3).
6. Combine size_idx × 4 + jit_idx → 16-level idx → float → vector.

Usage
-----
    from poc_catfur.shape_scan_decoder import decode_image
    from poc_catfur import shape_encoder as enc

    svg = open("pattern.svg").read()
    vector = decode_image("photo.jpg", svg)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as nd_label

from .shape_encoder import (
    CANVAS_H,
    CANVAS_W,
    D,
    _FID_CENTERS,
    _FID_HALF,
    _FID_INNER,
    _FID_OUTER,
    _JITTER_BASE,
    _JITTER_LEVELS,
    _JITTER_STEP,
    _N_JIT,
    _N_SIZE,
    _NS,
    _snap_to_grid,
    idx_to_float,
    size_jit_to_idx,
)

# Patch radius is derived per-decode from the homography scale so it always
# covers ~40% of the distance between candidate jitter positions. See
# decode_image() below.
_PATCH_FRAC = 0.4


def _compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT homography: src (N,2) SVG points → dst (N,2) image points.

    Returns 3×3 H such that dst_h ∝ H @ src_h (homogeneous coords).
    """
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3×3 homography to (N,2) points → (N,2) result."""
    h = np.column_stack([pts, np.ones(len(pts))])
    t = (H @ h.T).T
    return t[:, :2] / t[:, 2:3]


def _detect_fiducials(binary: np.ndarray) -> np.ndarray:
    """Find 4 fiducial centres in a binary image (True = dark ink).

    Strategy: the fiducials sit at known SVG corners. We estimate the SVG→image
    scale from the image dimensions (no perspective needed for this bootstrap
    step; canvas aspect is preserved in the simulation/photo), then for each
    expected corner we pick the largest dark component whose centroid falls in
    that image quadrant. This is DPI-independent and doesn't care whether the
    symbol region forms one merged blob or thousands of isolated components.

    Returns (4, 2) array of (x, y) centres in the SAME order as
    shape_encoder._FID_CENTERS (TL, TR, BL, BR).
    """
    h, w = binary.shape
    labelled, n = nd_label(binary)
    if n == 0:
        raise RuntimeError("No dark regions found in image")

    # Rough image positions of the 4 expected fiducials. Scale is approximate;
    # we only use it to pick a search quadrant per fiducial.
    scale_x = w / CANVAS_W
    scale_y = h / CANVAS_H
    expected = np.array(
        [(cx * scale_x, cy * scale_y) for cx, cy in _FID_CENTERS],
        dtype=np.float64,
    )

    # Precompute centroid + size for every component.
    sizes = np.bincount(labelled.ravel())[1:]
    # Use ndimage.center_of_mass-style computation without the extra import.
    idx = np.arange(1, n + 1)
    ys, xs = np.indices(binary.shape)
    cxs = np.bincount(labelled.ravel(), weights=xs.ravel())[1:] / np.maximum(sizes, 1)
    cys = np.bincount(labelled.ravel(), weights=ys.ravel())[1:] / np.maximum(sizes, 1)

    # Minimum plausible fiducial size, to avoid picking specks of noise. The
    # inner bright square is BG, so the dark ring area at scale s is
    # (outer² − inner²) × s². Use 30 % of that as the floor.
    exp_ring_svg = _FID_OUTER * _FID_OUTER - _FID_INNER * _FID_INNER
    min_size = 0.3 * exp_ring_svg * scale_x * scale_y
    cand_mask = sizes >= min_size
    if cand_mask.sum() < 4:
        raise RuntimeError(
            f"Only {int(cand_mask.sum())} dark components ≥ min_size; need ≥ 4 fiducials"
        )

    cand_labels = idx[cand_mask]
    cand_centres = np.column_stack([cxs[cand_mask], cys[cand_mask]])
    cand_sizes = sizes[cand_mask]

    chosen = []
    for ex, ey in expected:
        # Distance from each candidate centroid to this expected fiducial.
        d2 = (cand_centres[:, 0] - ex) ** 2 + (cand_centres[:, 1] - ey) ** 2
        # Score: distance penalty, ties broken by size (bigger ring wins).
        order = np.lexsort((-cand_sizes, d2))
        chosen.append(cand_centres[order[0]])
    ordered = np.asarray(chosen, dtype=np.float64)

    # Sanity: the 4 centres should be distinct.
    if len({tuple(c) for c in ordered}) < 4:
        raise RuntimeError("Fiducial detection matched the same component twice")
    return ordered


def _parse_data_cells(svg: str) -> list[tuple[int, float, float]]:
    """Return [(i, bx, by)] for each data cell id=e{i} in the SVG.

    bx, by are the base (grid) positions in SVG coordinates. v2 data cells are
    emitted as <circle id="e{i}" cx cy r fill>.
    """
    root = ET.fromstring(svg)
    circle_tag = f"{{{_NS}}}circle"
    cells = []
    for el in root.iter(circle_tag):
        uid = el.get("id", "")
        if not (uid.startswith("e") and uid[1:].isdigit()):
            continue
        i = int(uid[1:])
        if i < 0 or i >= D:
            continue
        try:
            x = float(el.get("cx", ""))
            y = float(el.get("cy", ""))
        except ValueError:
            continue
        bx, by = _snap_to_grid(x, y)
        cells.append((i, bx, by))
    return cells


def _ink_score(img_grey: np.ndarray, cx: float, cy: float, patch_r: int) -> float:
    """Average darkness (0–1) in a ±patch_r patch around (cx, cy).

    Using continuous darkness instead of a hard threshold makes the score
    robust to blur: a softly-haloed symbol still contributes signal even when
    no pixel is literally < 128.
    """
    h, w = img_grey.shape
    x0 = max(0, int(cx) - patch_r)
    x1 = min(w, int(cx) + patch_r + 1)
    y0 = max(0, int(cy) - patch_r)
    y1 = min(h, int(cy) + patch_r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = img_grey[y0:y1, x0:x1].astype(np.float32)
    return float((255.0 - patch).mean() / 255.0)


def decode_image(
    image_path: str | Path,
    svg: str,
    dark_thresh: int = 128,
) -> np.ndarray:
    """Decode a phone photo of the printed pattern → 1536-d float32 vector.

    Parameters
    ----------
    image_path : path to the photograph or flat scan.
    svg        : the original SVG string produced by shape_encoder.encode().
    dark_thresh: pixels below this grey value count as "ink" (0–255).

    Returns
    -------
    (1536,) float32 vector, or raises if fiducials cannot be found.
    """
    # ── Load + threshold ────────────────────────────────────────────────────
    img = Image.open(image_path).convert("L")
    img_arr = np.array(img, dtype=np.uint8)
    binary = img_arr < dark_thresh   # True = dark ink

    # ── Detect fiducials ────────────────────────────────────────────────────
    img_fid_centres = _detect_fiducials(binary)   # (4,2) image coords in _FID_CENTERS order
    svg_fid = np.array(_FID_CENTERS, dtype=np.float64)

    # Homography: SVG coords → image coords.
    H = _compute_homography(svg_fid, img_fid_centres)

    # Derive patch radius from the homography scale so it covers ~_PATCH_FRAC
    # of the distance between neighbouring jitter candidates. One SVG pixel
    # projects to `px_per_svg` image pixels; candidates are _JITTER_STEP SVG
    # pixels apart.
    svg_unit = _apply_homography(H, np.array([[0.0, 0.0], [1.0, 0.0]]))
    px_per_svg = float(np.linalg.norm(svg_unit[1] - svg_unit[0]))
    patch_r = max(2, int(round(_PATCH_FRAC * _JITTER_STEP * px_per_svg)))

    # ── Parse data cells from SVG ────────────────────────────────────────────
    cells = _parse_data_cells(svg)
    if len(cells) != D:
        raise ValueError(f"SVG contains {len(cells)} data cells; expected {D}")

    # Pre-build all _N_JIT (=4) candidate SVG offsets.
    offsets = np.array(
        [(ix * _JITTER_STEP + _JITTER_BASE, iy * _JITTER_STEP + _JITTER_BASE)
         for iy in range(_JITTER_LEVELS)
         for ix in range(_JITTER_LEVELS)],
        dtype=np.float64,
    )   # (_N_JIT, 2)

    # ── First pass: for each cell, find the jitter winner + its ink score ───
    jit_ids = np.zeros(D, dtype=np.int32)
    areas = np.full(D, np.nan, dtype=np.float64)

    for cell_i, bx, by in cells:
        svg_cands = np.column_stack([
            np.full(len(offsets), bx) + offsets[:, 0],
            np.full(len(offsets), by) + offsets[:, 1],
        ])   # (_N_JIT, 2)
        img_cands = _apply_homography(H, svg_cands)

        scores = np.array([
            _ink_score(img_arr, cx, cy, patch_r)
            for cx, cy in img_cands
        ])
        j = int(np.argmax(scores))
        jit_ids[cell_i] = j
        areas[cell_i] = float(scores[j])

    # ── Second pass: quantile-normalise winning-position darkness → size_idx.
    # The encoder distributes ada-002's ~uniform atan-quantised bins evenly
    # across the 4 size levels, so after ranking areas across all D cells the
    # bottom 25 % are size 0, next 25 % size 1, etc. This is exposure-invariant
    # (any monotonic gamma on the photo preserves the ranking).
    order = np.argsort(areas, kind="stable")
    size_ids = np.zeros(D, dtype=np.int32)
    per_bin = D // _N_SIZE
    for rank, cell_i in enumerate(order):
        size_ids[cell_i] = min(_N_SIZE - 1, rank // per_bin)

    # ── Combine → float → vector ─────────────────────────────────────────────
    vector = np.zeros(D, dtype=np.float32)
    for cell_i in range(D):
        idx = size_jit_to_idx(int(size_ids[cell_i]), int(jit_ids[cell_i]))
        vector[cell_i] = idx_to_float(idx)

    return vector
