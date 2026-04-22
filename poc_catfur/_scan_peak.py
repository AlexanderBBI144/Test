"""Decode by local-peak (centroid) detection instead of 16-argmax."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from scipy.ndimage import gaussian_filter

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI

from . import shape_decoder as dec
from . import shape_encoder as enc
from . import shape_scan_decoder as sdec

PHOTO = Path(__file__).resolve().parents[1] / "photo_2026-04-22_01-04-37.jpg"
SVG_PATH = Path(__file__).parent / "out" / "shape_chocolate_is_delicious.svg"


def cyan_mask(rgb):
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (g + b) * 0.5 - r


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def decode_by_peak(svg: str, photo_path: Path) -> np.ndarray:
    """Decode with per-cell centroid detection."""
    rgb = np.array(Image.open(photo_path).convert("RGB"))
    cyan = cyan_mask(rgb)
    binary = cyan > 20.0

    # Fiducials → homography.
    img_fid = sdec._detect_fiducials(binary)
    svg_fid = np.array(enc._FID_CENTERS, dtype=np.float64)
    H = sdec._compute_homography(svg_fid, img_fid)
    H_inv = np.linalg.inv(H)

    # Mild Gaussian smoothing on the cyan score (not the binary) so small dots
    # yield a clean single-peak response.
    cyan_smooth = gaussian_filter(np.clip(cyan, 0, None), sigma=0.8)

    # Scale estimate.
    svg_unit = sdec._apply_homography(H, np.array([[0.0, 0.0], [1.0, 0.0]]))
    px_per_svg = float(np.linalg.norm(svg_unit[1] - svg_unit[0]))

    # Sample radius around each cell base (image px).
    # Jitter range is ±3 SVG px; DON'T exceed that or we pick up neighboring
    # decorative symbols and get systematic bias.
    jitter_max_svg = enc._JITTER_STEP * (enc._JITTER_LEVELS - 1) / 2   # = 3.0 SVG px
    r_img = max(2, int(round(jitter_max_svg * px_per_svg)))

    cells = sdec._parse_data_cells(svg)
    vector = np.zeros(enc.D, dtype=np.float32)
    h, w = cyan_smooth.shape

    for cell_i, bx, by in cells:
        # Project base to image.
        [[cx, cy]] = sdec._apply_homography(H, np.array([[bx, by]]))
        x0 = max(0, int(cx) - r_img)
        x1 = min(w, int(cx) + r_img + 1)
        y0 = max(0, int(cy) - r_img)
        y1 = min(h, int(cy) + r_img + 1)
        patch = cyan_smooth[y0:y1, x0:x1]

        if patch.size == 0 or patch.max() < 5:
            # No detectable ink → fall back to jitter=0 (center).
            vector[cell_i] = enc.idx_to_float(int(enc._N_LEVELS // 2))
            continue

        # Peak position (integer pixel) → refine with ±1-px weighted centroid.
        py, px = np.unravel_index(np.argmax(patch), patch.shape)
        # 3×3 centroid around the peak for sub-pixel accuracy.
        py0 = max(0, py - 1); py1 = min(patch.shape[0], py + 2)
        px0 = max(0, px - 1); px1 = min(patch.shape[1], px + 2)
        sub = patch[py0:py1, px0:px1].astype(np.float64)
        sub = np.clip(sub - sub.min(), 0, None)
        s = sub.sum()
        if s > 1e-6:
            yy, xx = np.indices(sub.shape)
            cy_sub = (yy * sub).sum() / s + py0
            cx_sub = (xx * sub).sum() / s + px0
        else:
            cy_sub, cx_sub = float(py), float(px)

        # Image-space peak coords.
        peak_img = np.array([[x0 + cx_sub, y0 + cy_sub]])
        # Back-project to SVG space.
        [[sx_peak, sy_peak]] = sdec._apply_homography(H_inv, peak_img)
        dx = sx_peak - bx
        dy = sy_peak - by
        idx = enc.jitter_to_idx(dx, dy)
        vector[cell_i] = enc.idx_to_float(idx)

    return vector


def main():
    svg = SVG_PATH.read_text()
    v_scan = decode_by_peak(svg, PHOTO)
    v_svg = dec.decode(svg)

    client = OpenAI()
    r = client.embeddings.create(model="text-embedding-ada-002", input=["chocolate is delicious"])
    v_orig = np.array(r.data[0].embedding, dtype=np.float32); v_orig /= np.linalg.norm(v_orig)

    print(f"cos(svg_decoder, orig) = {cosine(v_svg, v_orig):.4f}")
    print(f"cos(scan, svg_decoder) = {cosine(v_scan, v_svg):.4f}")
    print(f"cos(scan, orig)        = {cosine(v_scan, v_orig):.4f}")

    idx_svg = np.array([enc.float_to_idx(float(f)) for f in v_svg])
    idx_scan = np.array([enc.float_to_idx(float(f)) for f in v_scan])
    agree = int((idx_svg == idx_scan).sum())
    print(f"idx agreement: {agree}/{len(idx_svg)}  ({agree/len(idx_svg)*100:.1f}%)")
    u, c = np.unique(idx_scan, return_counts=True)
    print(f"scan idx histogram: {dict(zip(u.tolist(), c.tolist()))}")


if __name__ == "__main__":
    main()
