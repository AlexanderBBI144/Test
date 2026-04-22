"""Decode SVG produced by shape_encoder (v2: size + jitter codec) → 1536-d vector.

Data cells are emitted as `<circle id="e{i}" cx cy r fill>` where:
  • r encodes the 4-level size channel (idx_to_radius)
  • (cx, cy) encodes the 2×2 jitter channel relative to the grid base
"""

import numpy as np
import xml.etree.ElementTree as ET

from .shape_encoder import (
    D,
    _NS,
    _snap_to_grid,
    idx_to_float,
    jitter_to_jit_idx,
    radius_to_size_idx,
    size_jit_to_idx,
)

ET.register_namespace("", _NS)
_CIRCLE = f"{{{_NS}}}circle"


def decode(svg: str) -> np.ndarray:
    root = ET.fromstring(svg)
    vector = np.zeros(D, dtype=np.float32)
    seen = np.zeros(D, dtype=bool)

    for el in root.iter(_CIRCLE):
        uid = el.get("id", "")
        if not (uid.startswith("e") and uid[1:].isdigit()):
            continue
        i = int(uid[1:])
        if i < 0 or i >= D:
            continue

        try:
            x = float(el.get("cx", ""))
            y = float(el.get("cy", ""))
            r = float(el.get("r", ""))
        except ValueError:
            continue

        bx, by = _snap_to_grid(x, y)
        jit_idx = jitter_to_jit_idx(x - bx, y - by)
        size_idx = radius_to_size_idx(r)
        idx = size_jit_to_idx(size_idx, jit_idx)
        vector[i] = idx_to_float(idx)
        seen[i] = True

    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError(f"decode: {missing} of {D} data cells missing from SVG")

    return vector
