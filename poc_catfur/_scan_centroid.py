"""Decode by full weighted centroid of ink within the jitter zone."""
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


def decode_centroid(svg: str, photo_path: Path, sigma: float = 0.6) -> np.ndarray:
    rgb = np.array(Image.open(photo_path).convert("RGB"))
    cyan = np.clip(cyan_mask(rgb), 0, None)
    binary = cyan > 20.0

    img_fid = sdec._detect_fiducials(binary)
    svg_fid = np.array(enc._FID_CENTERS, dtype=np.float64)
    H = sdec._compute_homography(svg_fid, img_fid)
    H_inv = np.linalg.inv(H)

    cyan_smooth = gaussian_filter(cyan, sigma=sigma)

    svg_unit = sdec._apply_homography(H, np.array([[0.0, 0.0], [1.0, 0.0]]))
    px_per_svg = float(np.linalg.norm(svg_unit[1] - svg_unit[0]))

    jitter_max_svg = enc._JITTER_STEP * (enc._JITTER_LEVELS - 1) / 2
    r_img = max(2, int(round(jitter_max_svg * px_per_svg)))

    cells = sdec._parse_data_cells(svg)
    vector = np.zeros(enc.D, dtype=np.float32)
    h, w = cyan_smooth.shape

    for cell_i, bx, by in cells:
        [[cx, cy]] = sdec._apply_homography(H, np.array([[bx, by]]))
        x0 = max(0, int(round(cx)) - r_img)
        x1 = min(w, int(round(cx)) + r_img + 1)
        y0 = max(0, int(round(cy)) - r_img)
        y1 = min(h, int(round(cy)) + r_img + 1)
        patch = cyan_smooth[y0:y1, x0:x1]

        if patch.size == 0:
            vector[cell_i] = enc.idx_to_float(enc._N_LEVELS // 2)
            continue

        # Subtract a per-cell background (min of patch) to suppress uniform bleed
        # from adjacent decoration. Then any remaining signal is the local dot.
        bg = np.percentile(patch, 20)
        weights = np.clip(patch - bg, 0, None).astype(np.float64)
        s = weights.sum()
        if s < 1e-3:
            vector[cell_i] = enc.idx_to_float(enc._N_LEVELS // 2)
            continue

        yy, xx = np.indices(patch.shape)
        cy_sub = (yy * weights).sum() / s + y0
        cx_sub = (xx * weights).sum() / s + x0

        [[sx_peak, sy_peak]] = sdec._apply_homography(H_inv, np.array([[cx_sub, cy_sub]]))
        dx = sx_peak - bx
        dy = sy_peak - by
        idx = enc.jitter_to_idx(dx, dy)
        vector[cell_i] = enc.idx_to_float(idx)

    return vector


def main():
    svg = SVG_PATH.read_text()

    client = OpenAI()
    r = client.embeddings.create(model="text-embedding-ada-002", input=["chocolate is delicious"])
    v_orig = np.array(r.data[0].embedding, dtype=np.float32); v_orig /= np.linalg.norm(v_orig)
    v_svg = dec.decode(svg)

    best = ("-", -1.0)
    for sigma in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        v_scan = decode_centroid(svg, PHOTO, sigma=sigma)
        cos_o = cosine(v_scan, v_orig)
        cos_s = cosine(v_scan, v_svg)
        idx_svg = np.array([enc.float_to_idx(float(f)) for f in v_svg])
        idx_scan = np.array([enc.float_to_idx(float(f)) for f in v_scan])
        agree = int((idx_svg == idx_scan).sum())
        print(f"sigma={sigma}  cos(scan,svg)={cos_s:.4f}  cos(scan,orig)={cos_o:.4f}  idx_agree={agree}/{len(idx_svg)} ({agree/len(idx_svg)*100:.1f}%)")
        if cos_o > best[1]:
            best = (sigma, cos_o)

    print(f"\nbest sigma={best[0]}  cos={best[1]:.4f}")


if __name__ == "__main__":
    main()
