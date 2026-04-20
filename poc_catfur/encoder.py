"""Encode a d-dim unit vector as a field of organic black blobs on white.

Layout:
  * 3 large anchor blobs in an equilateral triangle (for rectification).
  * DIM * REDUNDANCY small "fur" blobs placed on a golden-angle (sunflower)
    spiral inside a disk. Each blob's AREA encodes one scalar.

Each vector component v_i is carried by REDUNDANCY separate blobs; the
decoder takes the median so any one blob can be lost without harming the
reconstruction.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .embed import DIM

REDUNDANCY = 4
N_DATA_BLOBS = DIM * REDUNDANCY
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))

# Value clipping: corpus-derived PCA components are roughly in [-0.5, 0.5]
# after sphere-normalisation; widen to be safe.
V_MIN, V_MAX = -0.6, 0.6


def _sunflower(n: int, r_max: float) -> np.ndarray:
    idx = np.arange(n)
    r = r_max * np.sqrt((idx + 0.5) / n)
    theta = idx * GOLDEN_ANGLE
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def canonical_layout(canvas: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """Return (data_positions, anchor_positions) in canvas pixel coords."""
    cx, cy = canvas / 2.0, canvas / 2.0
    r_max = canvas * 0.36
    data = _sunflower(N_DATA_BLOBS, r_max) + [cx, cy]

    r_anchor = canvas * 0.46
    angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 - 2 * np.pi / 3]
    anchors = np.array(
        [[r_anchor * np.cos(a) + cx, -r_anchor * np.sin(a) + cy] for a in angles]
    )
    return data, anchors


def _area_bounds(canvas: int) -> tuple[float, float, float]:
    """Return (min_data_area, max_data_area, anchor_area) in px^2."""
    min_area = np.pi * (canvas * 0.014) ** 2
    max_area = np.pi * (canvas * 0.028) ** 2
    anchor_area = np.pi * (canvas * 0.055) ** 2  # ~3.9x max data area
    return min_area, max_area, anchor_area


def _render_blob(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    target_area: float,
    seed: int,
    organicness: float = 0.30,
) -> None:
    """Draw an organic black blob centred at (cx, cy) with the given area."""
    rng = np.random.default_rng(seed)
    n_pts = 28
    thetas = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    # low-freq radial noise: few lobes + per-point jitter
    phase = rng.uniform(0, 2 * np.pi, size=2)
    lobes = rng.integers(2, 5)
    shape = 1.0 + organicness * np.cos(lobes * thetas + phase[0])
    shape *= 1.0 + 0.5 * organicness * np.cos(
        (lobes + 1) * thetas + phase[1]
    )
    shape *= rng.uniform(1 - 0.10, 1 + 0.10, size=n_pts)

    # Scale so polygon area (shoelace) exactly matches target_area.
    # Area of polygon with vertices (r_i cos θ_i, r_i sin θ_i) on equi-angle grid:
    #   A = 0.5 * sin(Δθ) * Σ r_i * r_{i+1}
    dtheta = 2 * np.pi / n_pts
    ring_sum = float(np.sum(shape * np.roll(shape, -1)))
    unit_area = 0.5 * np.sin(dtheta) * ring_sum
    scale = np.sqrt(target_area / unit_area)
    r = shape * scale

    pts = [(cx + r[i] * np.cos(thetas[i]), cy + r[i] * np.sin(thetas[i])) for i in range(n_pts)]
    draw.polygon(pts, fill=0)


def encode(vector: np.ndarray, canvas: int = 1200) -> Image.Image:
    assert vector.shape == (DIM,)
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)

    data_pts, anchor_pts = canonical_layout(canvas)
    min_a, max_a, anchor_a = _area_bounds(canvas)

    # Anchors first (large, nearly round — low organicness so they stand out)
    for i, (x, y) in enumerate(anchor_pts):
        _render_blob(draw, x, y, anchor_a, seed=9000 + i, organicness=0.10)

    # Data blobs
    v_clipped = np.clip(vector, V_MIN, V_MAX)
    t = (v_clipped - V_MIN) / (V_MAX - V_MIN)  # (DIM,) in [0, 1]
    target_areas = min_a + t * (max_a - min_a)  # (DIM,)

    for comp in range(DIM):
        for k in range(REDUNDANCY):
            idx = comp * REDUNDANCY + k
            x, y = data_pts[idx]
            _render_blob(draw, x, y, float(target_areas[comp]), seed=100 + idx)

    return img
