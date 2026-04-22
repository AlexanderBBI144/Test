"""Overlay projected data-cell positions on the photo for visual debugging."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import shape_encoder as enc
from . import shape_scan_decoder as sdec

PHOTO = Path(__file__).resolve().parents[1] / "photo_2026-04-22_01-04-37.jpg"
SVG_PATH = Path(__file__).parent / "out" / "scan_i_love_geese.svg"
OUT = Path(__file__).parent / "out" / "scan_overlay.png"


def cyan_mask(rgb):
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (g + b) * 0.5 - r


def main():
    svg = SVG_PATH.read_text()
    rgb = np.array(Image.open(PHOTO).convert("RGB"))
    cyan = cyan_mask(rgb)
    binary = cyan > 20.0

    img_fid = sdec._detect_fiducials(binary)
    svg_fid = np.array(enc._FID_CENTERS, dtype=np.float64)
    H = sdec._compute_homography(svg_fid, img_fid)

    cells = sdec._parse_data_cells(svg)
    base = np.array([(bx, by) for _, bx, by in cells], dtype=np.float64)
    projected = sdec._apply_homography(H, base)

    # Also mark: canvas outline via forward-mapped corners.
    corners_svg = np.array([[0, 0], [enc.CANVAS_W, 0], [enc.CANVAS_W, enc.CANVAS_H], [0, enc.CANVAS_H]], dtype=np.float64)
    corners_img = sdec._apply_homography(H, corners_svg)

    # Bright the photo so we can see the cyan clearly.
    out = Image.fromarray(rgb).convert("RGB").copy()
    dr = ImageDraw.Draw(out)

    # Red: projected cell positions.
    r = 2
    for x, y in projected:
        dr.ellipse([x - r, y - r, x + r, y + r], outline="red")

    # Yellow: canvas outline.
    pts = [tuple(p) for p in corners_img] + [tuple(corners_img[0])]
    dr.line(pts, fill="yellow", width=2)

    # Green: fiducials detected.
    for cx, cy in img_fid:
        dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline="lime", width=2)

    out.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
