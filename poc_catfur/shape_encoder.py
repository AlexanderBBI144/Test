"""
Layered matrix codec: 1536-d ada-002 vector → SVG matching TARGET_PATTERN.svg aesthetic.

Aesthetic layers (driven by a seed derived from the vector):
  1. Anisotropic mask — organic horizontal ribbons, ~40% canvas fill.
  2. Symbol zones — within the mask, each contiguous zone is ONE symbol type (v/d/t/b).
     Target within-mask frequencies: v=50%, d=28%, t=20%, b=2%.
  3. Dot field — sparse isolated `d` symbols scattered in empty cells (starry background).
  4. Seam rows — a few horizontal rows overlay alternating v/b for the "=V=V=" look.

Data channel: each of the first 1536 filled cells carries one float of the input
vector via coarse-grid jitter. 9-level quantisation (3×3 offset grid, step 2 px,
range ±2 px) → robust to the print → phone-photo pipeline (step ≈ 0.7 mm at
25 cm print width, ≈ 4 phone pixels per step at 60 cm distance).

Four corner fiducial markers (40×40 px dark ring) enable homography-based
perspective correction in the scan decoder.
"""

import hashlib
import math
import re

import numpy as np
import xml.etree.ElementTree as ET

# ── Canvas / grid ───────────────────────────────────────────────────────────
CANVAS_W = 710
CANVAS_H = 1069
CELL: float = 13.0
D = 1536
BG = "#0E142C"
FG = "#1FD9FE"
_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _NS)

SYMBOLS = ["d", "t", "v", "b"]
SYM_IDX = {s: i for i, s in enumerate(SYMBOLS)}
EMPTY = -1

# ── Jitter / size / float codec ─────────────────────────────────────────────
# Real-print-robust v2: 16 levels per cell = 4 sizes × 4 jitter positions.
#   • Size survives any blur (total ink area is conserved).
#   • Jitter step 4 SVG px → always ≥ 4.8 image px even at 96 DPI cell phone.
#   • Per-image quantile normalization handles overall exposure shifts.
_JITTER_LEVELS = 2                     # per axis → 2×2 = 4 positions/cell
_JITTER_STEP = 6.0                     # SVG px between jitter positions (1.5× scale)
_JITTER_BASE = -(_JITTER_LEVELS - 1) / 2 * _JITTER_STEP   # = -3.0
_N_JIT = _JITTER_LEVELS ** 2           # = 4
_N_SIZE = 4                            # 4 discrete data-dot radii
_N_LEVELS = _N_SIZE * _N_JIT           # 16
_ATAN_SCALE = 0.035                    # tuned to ada-002 σ ≈ 0.026

# Circle radii in SVG px for the 4 size levels (1.5× larger than v2).
# At 1.1 img px/SVG px the smallest dot is ~2 img px diameter — clearly visible.
_SIZE_RADII = (0.9, 1.5, 2.1, 2.7)

# ── Fiducial markers ─────────────────────────────────────────────────────────
# Four corner rings used by the scan decoder for perspective correction.
_FID_OUTER = 40   # outer square side (px)
_FID_INNER = 20   # inner bright square side (px)
_FID_CENTERS = [
    (20, 20),
    (CANVAS_W - 20, 20),
    (20, CANVAS_H - 20),
    (CANVAS_W - 20, CANVAS_H - 20),
]
# Grid cells whose base position falls within any fiducial zone are excluded
# from the data set so they don't interfere with the markers.
_FID_HALF = _FID_OUTER // 2


def float_to_idx(f: float) -> int:
    t = math.atan(f / _ATAN_SCALE) / math.pi + 0.5
    return min(int(t * _N_LEVELS), _N_LEVELS - 1)


def idx_to_float(idx: int) -> float:
    t = (idx + 0.5) / _N_LEVELS
    arg = max(-1.55, min(1.55, (t - 0.5) * math.pi))
    return _ATAN_SCALE * math.tan(arg)


def idx_to_size(idx: int) -> int:
    """Low 2 bits of a 16-level idx → size index (0..3).

    Using LOW bits for size keeps the size distribution uniform even when the
    source vector (e.g. ada-002) concentrates values near zero and therefore
    concentrates idx around the middle of [0, 16). The scan decoder recovers
    size via quantile normalisation of ink darkness, which requires an
    approximately uniform size histogram to assign correct bin boundaries.
    Jit, recovered independently per cell via argmax, is noise-tolerant under
    a skewed distribution, so it takes the high bits.
    """
    return int(idx) % _N_SIZE


def idx_to_radius(idx: int) -> float:
    return _SIZE_RADII[idx_to_size(idx)]


def idx_to_jit_idx(idx: int) -> int:
    """High 2 bits of a 16-level idx → jitter index (0..3)."""
    return int(idx) // _N_SIZE


def idx_to_jitter(idx: int) -> tuple[float, float]:
    jit = idx_to_jit_idx(idx)
    ix = jit % _JITTER_LEVELS
    iy = jit // _JITTER_LEVELS
    return _JITTER_BASE + ix * _JITTER_STEP, _JITTER_BASE + iy * _JITTER_STEP


def jitter_to_jit_idx(dx: float, dy: float) -> int:
    ix = max(0, min(_JITTER_LEVELS - 1, round((dx - _JITTER_BASE) / _JITTER_STEP)))
    iy = max(0, min(_JITTER_LEVELS - 1, round((dy - _JITTER_BASE) / _JITTER_STEP)))
    return int(ix + iy * _JITTER_LEVELS)


def size_jit_to_idx(size_idx: int, jit_idx: int) -> int:
    return int(jit_idx) * _N_SIZE + int(size_idx)


def radius_to_size_idx(r: float) -> int:
    """Nearest discrete size level for a given circle radius (SVG px)."""
    diffs = [abs(r - rr) for rr in _SIZE_RADII]
    return int(min(range(_N_SIZE), key=lambda i: diffs[i]))


# Back-compat shim: some call sites previously computed idx purely from jitter.
def jitter_to_idx(dx: float, dy: float) -> int:   # deprecated
    return jitter_to_jit_idx(dx, dy)


# ── Grid positions ──────────────────────────────────────────────────────────

def _grid() -> tuple[np.ndarray, int, int]:
    """Returns (positions[N,2], cols, rows). positions is row-major."""
    xs = []
    x = 3.5
    while x <= CANVAS_W - 3:
        xs.append(round(x * 2) / 2)
        x += CELL
    ys = []
    y = 4.25
    while y <= CANVAS_H - 3:
        ys.append(round(y * 2) / 2)
        y += CELL
    cols, rows = len(xs), len(ys)
    pos = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return pos, cols, rows


_POS, _COLS, _ROWS = _grid()


# ── Aesthetic generation ────────────────────────────────────────────────────

def _vector_seed(vector: np.ndarray) -> int:
    h = hashlib.sha256(vector.tobytes()).digest()
    return int.from_bytes(h[:4], "big")


def _smooth_field(rng: np.random.Generator, coarse_cols: int, coarse_rows: int) -> np.ndarray:
    """Bilinear-upsample a coarse random grid to the full (rows, cols)."""
    coarse = rng.standard_normal((coarse_rows, coarse_cols)).astype(np.float32)
    y = np.linspace(0, coarse_rows - 1, _ROWS, dtype=np.float32)
    x = np.linspace(0, coarse_cols - 1, _COLS, dtype=np.float32)
    y0 = np.floor(y).astype(int); y1 = np.clip(y0 + 1, 0, coarse_rows - 1)
    x0 = np.floor(x).astype(int); x1 = np.clip(x0 + 1, 0, coarse_cols - 1)
    yf = (y - y0)[:, None]
    xf = (x - x0)[None, :]
    v00 = coarse[np.ix_(y0, x0)]
    v01 = coarse[np.ix_(y0, x1)]
    v10 = coarse[np.ix_(y1, x0)]
    v11 = coarse[np.ix_(y1, x1)]
    return v00 * (1 - xf) * (1 - yf) + v01 * xf * (1 - yf) \
         + v10 * (1 - xf) * yf       + v11 * xf * yf


def _build_grid(vector: np.ndarray) -> np.ndarray:
    """Returns (rows, cols) symbol_grid with values in {-1, 0..3}.

    -1 = empty dark background; 0..3 = index into SYMBOLS.
    Always yields ≥ D filled cells.
    """
    rng = np.random.default_rng(_vector_seed(vector))

    # Outer silhouette — distorted ellipse with organic edges.
    # Radial distance from centre guarantees no symbols touch canvas borders.
    # Centre shifted slightly downward (row_centre=0.1) to sit below collar.
    row_idx = np.linspace(-1, 1, _ROWS) - 0.10
    col_idx = np.linspace(-1, 1, _COLS)
    rg, cg = np.meshgrid(row_idx, col_idx, indexing='ij')
    # Narrower horizontally (0.95) so it stays within shirt body, not the arms.
    radial = np.sqrt((cg * 0.95) ** 2 + (rg * 0.82) ** 2)
    # Noise perturbation makes the ellipse boundary irregular/organic.
    perturb = _smooth_field(rng, coarse_cols=7, coarse_rows=5) * 0.22
    silhouette = (radial + perturb) < 0.90

    # Layer 1 — mask: anisotropic (stretched horizontally) → flowing ribbons.
    # Coarse 48×11 → horizontally-elongated ribbons ~65px tall × ~15px wide.
    mask_noise = _smooth_field(rng, coarse_cols=48, coarse_rows=11)
    # Target ~75% fill within silhouette so we always have ≥ D = 1536 filled cells.
    thresh = np.quantile(mask_noise, 0.25)
    mask = (mask_noise > thresh) & silhouette

    # Layer 2 — symbol zones: different noise field, blobby (less anisotropic).
    zone_noise = _smooth_field(rng, coarse_cols=18, coarse_rows=16)
    grid = np.full((_ROWS, _COLS), EMPTY, dtype=np.int8)
    # Within-mask thresholds by target cumulative: b=2%, v=12%, d=25%, t=100%.
    # t (company logo / tech glyph) is dominant at ~75%.
    masked = zone_noise[mask]
    q_b = np.quantile(masked, 0.02)
    q_v = np.quantile(masked, 0.12)
    q_d = np.quantile(masked, 0.25)
    grid[mask & (zone_noise <= q_b)] = SYM_IDX["b"]
    grid[mask & (zone_noise > q_b) & (zone_noise <= q_v)] = SYM_IDX["v"]
    grid[mask & (zone_noise > q_v) & (zone_noise <= q_d)] = SYM_IDX["d"]
    grid[mask & (zone_noise > q_d)] = SYM_IDX["t"]

    # Layer 3 — dot field: sparse `d` only inside silhouette but outside mask.
    empty_in_sil = silhouette & ~mask
    dot_probs = rng.random((_ROWS, _COLS))
    dots = empty_in_sil & (dot_probs < 0.04)
    grid[dots] = SYM_IDX["d"]

    # Layer 4 — seam rows: short alternating v/b fragments only inside mask.
    # Kept very sparse (1–2 rows, short spans) so they don't dominate visually.
    n_seams = int(rng.integers(1, 3))
    seam_rows = rng.choice(_ROWS, size=n_seams, replace=False)
    for r in seam_rows:
        c0 = int(rng.integers(_COLS // 4, _COLS // 2))
        c1 = int(c0 + rng.integers(_COLS // 8, _COLS // 4))
        for c in range(c0, min(c1, _COLS)):
            if grid[r, c] != EMPTY:   # only over existing filled cells
                grid[r, c] = SYM_IDX["b"] if (c & 1) == 0 else SYM_IDX["v"]

    return grid


# ── SVG defs ─────────────────────────────────────────────────────────────────

_T_PATH_0 = (
    "M -1.228 0.36162 C -1.3312 0.42861 -1.4688 0.39883 -1.5364 0.29464 "
    "V 0.2934 C -1.5376 0.29214 -1.5376 0.2909 -1.5389 0.28971 "
    "L -2.9763 -2.0994 C -3.0955 -2.2905 -3.0316 -2.5782 -2.7687 -2.6737 "
    "C -2.7589 -2.6775 -2.7491 -2.6787 -2.7392 -2.6787 H -1.2329 "
    "C -1.1088 -2.6787 -1.0057 -2.5658 -1.0057 -2.438 V -2.1577 "
    "C -1.0044 -2.0324 -1.1039 -1.9294 -1.228 -1.9282 H -2.0119 "
    "L -0.9282 -0.10478 C -0.8618 0.001905 -0.8938 0.14332 -0.9982 0.21155 "
    "L -1.228 0.36162 Z"
)
_T_PATH_1 = (
    "M -0.1788 -1.927 C -0.3004 -1.927 -0.3999 -2.0275 -0.3999 -2.1502 "
    "V -2.1515 L -0.4012 -2.3996 C -0.4024 -2.5931 -0.2476 -2.6663 "
    "-0.0596 -2.6663 L 2.6863 -2.67 C 2.9075 -2.6713 3.0955 -2.4653 "
    "3.034 -2.1863 C 3.0316 -2.1776 3.0279 -2.1689 3.023 -2.1602 "
    "L 2.2686 -0.89 C 2.2047 -0.78581 2.0696 -0.75354 1.9664 -0.81677 "
    "C 1.9652 -0.81677 1.9652 -0.81677 1.9639 -0.81802 L 1.7108 -0.96441 "
    "C 1.6076 -1.0314 1.5769 -1.1703 1.642 -1.2757 L 2.0278 -1.9258 "
    "L -0.1788 -1.927 Z"
)
_T_PATH_2 = (
    "M 1.1089 -0.37273 C 1.1716 -0.47816 1.3079 -0.51288 1.4124 -0.44964 "
    "L 1.4136 -0.44839 L 1.6249 -0.32185 C 1.7908 -0.22265 1.7736 -0.052725 "
    "1.6765 0.11228 L 0.2955 2.4381 C 0.1825 2.6304 -0.0485 2.6787 "
    "-0.2549 2.4827 C -0.261 2.4766 -0.2672 2.4691 -0.2708 2.4605 "
    "L -1.0264 1.1741 C -1.0817 1.0649 -1.0387 0.93221 -0.9318 0.87641 "
    "C -0.9306 0.87641 -0.9294 0.87515 -0.9282 0.87515 L -0.6739 0.73125 "
    "C -0.5633 0.67669 -0.4306 0.72008 -0.3741 0.83052 L 0.0081 1.4755 "
    "L 1.1089 -0.37273 Z"
)


def _tag(name: str) -> str:
    return f"{{{_NS}}}{name}"


def _defs(root: ET.Element) -> None:
    defs = ET.SubElement(root, _tag("defs"))

    sd = ET.SubElement(defs, _tag("symbol"), {"id": "d", "overflow": "visible"})
    ET.SubElement(sd, _tag("circle"), {"r": "0.8", "fill": FG})

    st = ET.SubElement(defs, _tag("symbol"), {"id": "t", "overflow": "visible"})
    for pdata in (_T_PATH_0, _T_PATH_1, _T_PATH_2):
        ET.SubElement(st, _tag("path"), {"d": pdata, "fill": FG})

    sv = ET.SubElement(defs, _tag("symbol"), {"id": "v", "overflow": "visible"})
    ET.SubElement(sv, _tag("path"), {
        "d": "M -2.6 -5.2 L 0 0 M 2.6 -5.2 L 0 0",
        "stroke": FG, "stroke-width": "0.88", "stroke-linecap": "round",
    })

    sb = ET.SubElement(defs, _tag("symbol"), {"id": "b", "overflow": "visible"})
    ET.SubElement(sb, _tag("path"), {
        "d": "M -2.4 -1 H 2.4 M -2.4 1 H 2.4",
        "stroke": FG, "stroke-width": "0.72", "stroke-linecap": "round",
    })


# ── Public API ──────────────────────────────────────────────────────────────

def encode(vector: np.ndarray) -> str:
    """(1536,) float32 → SVG string."""
    assert vector.shape == (D,), f"expected ({D},), got {vector.shape}"

    grid = _build_grid(vector)
    all_filled = np.argwhere(grid.reshape(-1) != EMPTY).ravel()

    # Exclude cells whose base position overlaps a fiducial zone.
    def _in_fid(pi: int) -> bool:
        px, py = _POS[pi]
        return any(
            abs(px - cx) <= _FID_HALF and abs(py - cy) <= _FID_HALF
            for cx, cy in _FID_CENTERS
        )

    filled = all_filled[[not _in_fid(int(pi)) for pi in all_filled]]
    if filled.size < D:
        raise RuntimeError(f"only {filled.size} filled cells; need ≥ {D}")

    # Data cells: evenly spaced across filled positions so jitter distributes
    # visually rather than piling up in the top-left.
    step = filled.size / D
    data_pi = np.array([filled[int(i * step)] for i in range(D)], dtype=np.int64)
    data_set = set(data_pi.tolist())

    root = ET.Element(_tag("svg"), {
        "width": str(CANVAS_W), "height": str(CANVAS_H),
        "viewBox": f"0 0 {CANVAS_W} {CANVAS_H}",
    })
    ET.SubElement(root, _tag("rect"), {
        "width": str(CANVAS_W), "height": str(CANVAS_H), "fill": BG,
    })
    _defs(root)

    # Fiducial corner markers: dark ring (outer square + bright centre).
    half_o = _FID_OUTER // 2
    half_i = _FID_INNER // 2
    for cx, cy in _FID_CENTERS:
        ET.SubElement(root, _tag("rect"), {
            "x": str(cx - half_o), "y": str(cy - half_o),
            "width": str(_FID_OUTER), "height": str(_FID_OUTER),
            "fill": FG,
        })
        ET.SubElement(root, _tag("rect"), {
            "x": str(cx - half_i), "y": str(cy - half_i),
            "width": str(_FID_INNER), "height": str(_FID_INNER),
            "fill": BG,
        })

    flat_grid = grid.reshape(-1)

    # Data-bearing cells. Emit raw <circle> so we can encode the SIZE channel
    # directly in the radius attribute (4 discrete radii), on top of the 2×2
    # jitter channel encoded in the translation. Size survives any blur; jitter
    # survives low camera resolution. Combined: 4×4 = 16 levels per cell.
    for i, pi in enumerate(data_pi.tolist()):
        bx, by = _POS[pi]
        idx = float_to_idx(float(vector[i]))
        dx, dy = idx_to_jitter(idx)
        r = idx_to_radius(idx)
        ET.SubElement(root, _tag("circle"), {
            "id": f"e{i}",
            "cx": f"{float(bx) + dx}",
            "cy": f"{float(by) + dy}",
            "r": f"{r}",
            "fill": FG,
        })

    # Decorative cells (no id, no jitter).
    for pi in filled.tolist():
        if pi in data_set:
            continue
        bx, by = _POS[pi]
        sym = SYMBOLS[int(flat_grid[pi])]
        ET.SubElement(root, _tag("use"), {
            "href": f"#{sym}",
            "transform": f"translate({float(bx)},{float(by)})",
        })

    return ET.tostring(root, encoding="unicode", xml_declaration=False)


# ── Decoder helpers (exported for shape_decoder) ────────────────────────────
_TRANSFORM_RE = re.compile(r"translate\(\s*([-\d.eE+]+)[\s,]+([-\d.eE+]+)\s*\)")


def _parse_translate(s: str) -> tuple[float, float] | None:
    m = _TRANSFORM_RE.search(s or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _snap_to_grid(x: float, y: float) -> tuple[float, float]:
    """Return the nearest grid base position for (x, y). Jitter range ±3 px
    is ~35% of CELL=8.5 px spacing; snap margin 1.25 px → unambiguous."""
    dists = (_POS[:, 0] - x) ** 2 + (_POS[:, 1] - y) ** 2
    pi = int(np.argmin(dists))
    return float(_POS[pi, 0]), float(_POS[pi, 1])
