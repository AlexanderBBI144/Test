"""Decode a photographed blob pattern back to a vector and nearest phrase.

Pipeline:
  1. Binarise the photo (adaptive threshold + Otsu-like mean).
  2. Label connected components; drop specks; get (centroid, area) per blob.
  3. Pick the three largest components as fiducial anchors.
  4. Fit an affine transform mapping detected anchors to canonical anchors.
  5. Push every blob centroid through the affine; compensate areas by |det|.
  6. For each canonical data position, take the nearest blob's area.
  7. Group into DIM bins of REDUNDANCY blobs; median → scalar per component.
  8. Invert the area-to-value mapping; L2-normalise; nearest in corpus.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, label, sum_labels

from .encoder import (
    DIM,
    N_DATA_BLOBS,
    REDUNDANCY,
    V_MAX,
    V_MIN,
    _area_bounds,
    canonical_layout,
)


def _binarise(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    # Threshold at a fraction of the mean. Works well for ink-on-paper / tee.
    thresh = 0.65 * a.mean()
    return (a < thresh).astype(np.uint8)


def _detect_blobs(bin_img: np.ndarray, min_area_px: int = 50) -> tuple[np.ndarray, np.ndarray]:
    lbl, n = label(bin_img)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    labels = np.arange(1, n + 1)
    areas = sum_labels(bin_img, lbl, labels).astype(np.float32)
    # centre_of_mass returns (row, col); convert to (x, y).
    coms = np.array(center_of_mass(bin_img, lbl, labels))
    centroids = coms[:, [1, 0]]
    keep = areas >= min_area_px
    return centroids[keep], areas[keep]


def _sort_by_angle(pts: np.ndarray, center: np.ndarray) -> np.ndarray:
    d = pts - center
    angles = np.arctan2(d[:, 1], d[:, 0])
    return pts[np.argsort(angles)]


def _affine_from_3pts(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve for 2x3 affine mapping src -> dst (exact fit for 3 pairs)."""
    A, b = [], []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0]); b.append(xp)
        A.append([0, 0, 0, x, y, 1]); b.append(yp)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[params[0], params[1], params[2]],
                     [params[3], params[4], params[5]]])


def _apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (M @ h.T).T


def decode(img: Image.Image, canvas: int = 1200) -> tuple[np.ndarray, dict]:
    """Return (reconstructed vector, diagnostics)."""
    diag: dict = {}
    data_canon, anchor_canon = canonical_layout(canvas)

    bin_img = _binarise(img)
    centroids, areas = _detect_blobs(bin_img)
    diag["n_blobs"] = int(len(centroids))
    if len(centroids) < 3 + DIM:
        raise RuntimeError(f"too few blobs detected: {len(centroids)}")

    # Three largest blobs are the anchors.
    top3 = np.argsort(-areas)[:3]
    det_anchors = centroids[top3]

    # Sort both detected and canonical anchors consistently by angle around
    # the global centroid.
    det_centre = centroids.mean(axis=0)
    det_sorted = _sort_by_angle(det_anchors, det_centre)
    canon_centre = np.array([canvas / 2.0, canvas / 2.0])
    canon_sorted = _sort_by_angle(anchor_canon, canon_centre)

    M = _affine_from_3pts(det_sorted, canon_sorted)
    diag["affine"] = M
    scale_area = abs(float(np.linalg.det(M[:2, :2])))

    # Non-anchor blobs in canonical coordinates.
    mask = np.ones(len(centroids), dtype=bool)
    mask[top3] = False
    data_centroids_canon = _apply_affine(M, centroids[mask])
    data_areas_canon = areas[mask] * scale_area
    diag["data_blobs"] = int(mask.sum())

    # For each canonical data position, nearest detected data blob (greedy).
    # Many-to-one is fine; worst case a whole group is wiped → value clamps.
    used = np.zeros(len(data_centroids_canon), dtype=bool)
    matched_areas = np.zeros(N_DATA_BLOBS)
    for i, p in enumerate(data_canon):
        dists = np.linalg.norm(data_centroids_canon - p, axis=1)
        # Soft-greedy: prefer unused but fall back to any.
        order = np.argsort(dists)
        chosen = None
        for j in order:
            if not used[j] and dists[j] < canvas * 0.06:
                chosen = j
                break
        if chosen is None:
            chosen = int(order[0])
        used[chosen] = True
        matched_areas[i] = data_areas_canon[chosen]

    # Group into DIM components, take median.
    matched_areas = matched_areas.reshape(DIM, REDUNDANCY)
    component_areas = np.median(matched_areas, axis=1)

    min_a, max_a, _ = _area_bounds(canvas)
    t = np.clip((component_areas - min_a) / (max_a - min_a), 0.0, 1.0)
    v = V_MIN + t * (V_MAX - V_MIN)
    v = v / (np.linalg.norm(v) + 1e-12)

    return v.astype(np.float32), diag
