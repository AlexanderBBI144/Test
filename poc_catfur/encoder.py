"""Encode a DIM-dim unit vector as a field of organic black blobs on white.

Layout:
  * 3 large anchor blobs in an equilateral triangle (for rectification).
  * DIM * REDUNDANCY small "fur" blobs placed on a golden-angle (sunflower)
    spiral inside a disk. Each blob's AREA encodes one scalar.

Each vector component v_i is carried by REDUNDANCY separate blobs; the
decoder takes the median so any one blob can be lost without harming the
reconstruction. The mapping from value to area is linear, calibrated per
component from the vocab's observed (v_lo, v_hi) percentiles.
"""

import numpy as np
from PIL import Image, ImageDraw

from .vocab import DIM

REDUNDANCY = 4
N_DATA_BLOBS = DIM * REDUNDANCY
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def _sunflower(n: int, r_max: float) -> np.ndarray:
    idx = np.arange(n)
    r = r_max * np.sqrt((idx + 0.5) / n)
    theta = idx * GOLDEN_ANGLE
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def canonical_layout(canvas: int = 1400) -> tuple[np.ndarray, np.ndarray]:
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


def area_bounds(canvas: int = 1400) -> tuple[float, float, float]:
    """Return (min_data_area, max_data_area, anchor_area) in px^2."""
    min_area = np.pi * (canvas * 0.012) ** 2
    max_area = np.pi * (canvas * 0.024) ** 2
    anchor_area = np.pi * (canvas * 0.048) ** 2  # ~4x max data area
    return min_area, max_area, anchor_area


def _render_blob(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    target_area: float,
    seed: int,
    organicness: float = 0.30,
) -> None:
    rng = np.random.default_rng(seed)
    n_pts = 28
    thetas = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    phase = rng.uniform(0, 2 * np.pi, size=2)
    lobes = rng.integers(2, 5)
    shape = 1.0 + organicness * np.cos(lobes * thetas + phase[0])
    shape *= 1.0 + 0.5 * organicness * np.cos((lobes + 1) * thetas + phase[1])
    shape *= rng.uniform(1 - 0.10, 1 + 0.10, size=n_pts)
    dtheta = 2 * np.pi / n_pts
    unit_area = 0.5 * np.sin(dtheta) * float(np.sum(shape * np.roll(shape, -1)))
    scale = np.sqrt(target_area / unit_area)
    r = shape * scale
    pts = [
        (cx + r[i] * np.cos(thetas[i]), cy + r[i] * np.sin(thetas[i]))
        for i in range(n_pts)
    ]
    draw.polygon(pts, fill=0)


def encode(
    vector: np.ndarray,
    v_lo: np.ndarray,
    v_hi: np.ndarray,
    canvas: int = 1400,
) -> Image.Image:
    assert vector.shape == (DIM,) and v_lo.shape == (DIM,) and v_hi.shape == (DIM,)
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    data_pts, anchor_pts = canonical_layout(canvas)
    min_a, max_a, anchor_a = area_bounds(canvas)

    for i, (x, y) in enumerate(anchor_pts):
        _render_blob(draw, x, y, anchor_a, seed=9000 + i, organicness=0.10)

    t = np.clip((vector - v_lo) / np.maximum(v_hi - v_lo, 1e-6), 0.0, 1.0)
    target_areas = min_a + t * (max_a - min_a)

    for comp in range(DIM):
        for k in range(REDUNDANCY):
            idx = comp * REDUNDANCY + k
            x, y = data_pts[idx]
            _render_blob(draw, x, y, float(target_areas[comp]), seed=100 + idx)

    return img
