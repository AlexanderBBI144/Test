"""Per-cell alignment diagnostic with the CORRECT SVG (shape_chocolate_is_delicious)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from scipy.ndimage import label as nd_label

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI

from . import shape_decoder as dec
from . import shape_encoder as enc
from . import shape_scan_decoder as sdec

PHOTO = Path(__file__).resolve().parents[1] / "photo_2026-04-22_01-04-37.jpg"
SVG_PATH = Path(__file__).parent / "out" / "shape_chocolate_is_delicious.svg"
OUT = Path(__file__).parent / "out" / "scan_overlay_choco.png"


def cyan_mask(rgb):
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (g + b) * 0.5 - r


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    svg = SVG_PATH.read_text()
    rgb = np.array(Image.open(PHOTO).convert("RGB"))
    cyan = cyan_mask(rgb)
    binary = cyan > 20.0
    ink_gray = (255 - np.clip(cyan * 4.0, 0, 255)).astype(np.uint8)

    img_fid = sdec._detect_fiducials(binary)
    svg_fid = np.array(enc._FID_CENTERS, dtype=np.float64)
    H = sdec._compute_homography(svg_fid, img_fid)

    # Project data cells, overlay on photo.
    cells = sdec._parse_data_cells(svg)
    base = np.array([(bx, by) for _, bx, by in cells], dtype=np.float64)
    projected = sdec._apply_homography(H, base)

    # Nearest cyan blob for each projected cell.
    lab, n = nd_label(binary)
    sizes = np.bincount(lab.ravel())[1:]
    ys, xs = np.indices(binary.shape)
    cxs = np.bincount(lab.ravel(), weights=xs.ravel())[1:] / np.maximum(sizes, 1)
    cys = np.bincount(lab.ravel(), weights=ys.ravel())[1:] / np.maximum(sizes, 1)
    # Consider small-to-medium blobs (data dots or single symbols).
    keep = (sizes >= 2) & (sizes <= 100)
    all_blob = np.column_stack([cxs[keep], cys[keep]])
    dists = np.sqrt(((projected[:, None, :] - all_blob[None, :, :]) ** 2).sum(-1))
    min_d = dists.min(axis=1)
    arg_d = dists.argmin(axis=1)

    print(f"data cells: {len(cells)}")
    print(f"candidate blobs (2-100 px): {len(all_blob)}")
    print(f"distance to nearest blob: median {np.median(min_d):.2f}  p50 {np.percentile(min_d,50):.2f}  p90 {np.percentile(min_d,90):.2f}  max {min_d.max():.2f}")
    vec = all_blob[arg_d] - projected
    print(f"systematic offset: dx_mean={vec[:,0].mean():+.2f}  dy_mean={vec[:,1].mean():+.2f}")
    print(f"offset stddev:     dx_std={vec[:,0].std():.2f}  dy_std={vec[:,1].std():.2f}")

    # By row band:
    bys = base[:, 1]
    print("\nmedian nearest-blob dist by canvas row band:")
    for lo in range(0, 1070, 107):
        hi = lo + 107
        mm = (bys >= lo) & (bys < hi)
        if mm.sum() > 10:
            print(f"  y=[{lo:4d},{hi:4d})  n={int(mm.sum()):4d}  median={np.median(min_d[mm]):5.2f}  dx={vec[mm,0].mean():+5.2f}  dy={vec[mm,1].mean():+5.2f}")

    # Overlay.
    out = Image.fromarray(rgb).convert("RGB").copy()
    dr = ImageDraw.Draw(out)
    for x, y in projected:
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], outline="red")
    for cx, cy in img_fid:
        dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline="lime", width=2)
    out.save(OUT)
    print(f"\nwrote {OUT}")

    # Decode and cos.
    tmp = Path(__file__).parent / "out" / "_diag_choco_gray.png"
    Image.fromarray(ink_gray, mode="L").save(tmp)
    v_scan = sdec.decode_image(tmp.as_posix(), svg)
    v_svg = dec.decode(svg)
    client = OpenAI()
    r = client.embeddings.create(model="text-embedding-ada-002", input=["chocolate is delicious"])
    v_orig = np.array(r.data[0].embedding, dtype=np.float32); v_orig /= np.linalg.norm(v_orig)
    print(f"\ncos(svg_decoder, orig)    = {cosine(v_svg, v_orig):.4f}")
    print(f"cos(scan, svg_decoder)   = {cosine(v_scan, v_svg):.4f}")
    print(f"cos(scan, orig)          = {cosine(v_scan, v_orig):.4f}")


if __name__ == "__main__":
    main()
