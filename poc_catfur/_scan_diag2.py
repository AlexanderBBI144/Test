"""Measure alignment between projected data-cell positions and real cyan blobs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as nd_label

from . import shape_encoder as enc
from . import shape_scan_decoder as sdec

PHOTO = Path(__file__).resolve().parents[1] / "photo_2026-04-22_01-04-37.jpg"
SVG_PATH = Path(__file__).parent / "out" / "scan_i_love_geese.svg"


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

    # All cyan blobs: label and get centroids.
    lab, n = nd_label(binary)
    sizes = np.bincount(lab.ravel())[1:]
    ys, xs = np.indices(binary.shape)
    cxs = np.bincount(lab.ravel(), weights=xs.ravel())[1:] / np.maximum(sizes, 1)
    cys = np.bincount(lab.ravel(), weights=ys.ravel())[1:] / np.maximum(sizes, 1)
    print(f"total cyan blobs: {n}")
    print(f"blob-size histogram (top buckets): {np.histogram(sizes, bins=[0,2,4,8,16,32,64,128,256,1024,10000])[0].tolist()}")

    # Small blob centres (likely data dots): size between 2 and 30 pixels.
    tiny_mask = (sizes >= 2) & (sizes <= 30)
    tiny_cx = cxs[tiny_mask]
    tiny_cy = cys[tiny_mask]
    print(f"tiny blobs (2-30 px): {tiny_mask.sum()}")

    # Fiducials & homography.
    img_fid = sdec._detect_fiducials(binary)
    svg_fid = np.array(enc._FID_CENTERS, dtype=np.float64)
    H = sdec._compute_homography(svg_fid, img_fid)

    # Project each data cell's BASE (no-jitter) position into image space.
    cells = sdec._parse_data_cells(svg)
    base = np.array([(bx, by) for _, bx, by in cells], dtype=np.float64)
    projected = sdec._apply_homography(H, base)   # (1536, 2)

    # For each projected cell, find nearest tiny blob centre.
    # Brute force but OK at n=1536 × ~thousands of blobs.
    if len(tiny_cx):
        all_blob = np.column_stack([tiny_cx, tiny_cy])
        dists = np.sqrt(((projected[:, None, :] - all_blob[None, :, :]) ** 2).sum(-1))
        min_d = dists.min(axis=1)
        arg_d = dists.argmin(axis=1)
        print(f"\ndistances from projected cell to nearest tiny blob:")
        print(f"  median {np.median(min_d):.2f}  mean {min_d.mean():.2f}  p90 {np.percentile(min_d,90):.2f}  max {min_d.max():.2f}")
        # Histogram.
        bins = [0, 1, 2, 3, 5, 8, 12, 20, 40, 100]
        hist, _ = np.histogram(min_d, bins=bins)
        print(f"  hist (bins {bins}): {hist.tolist()}")

        # Per-row (region of canvas) alignment: break into 10 horizontal bands
        # and report median distance per band to see if drift varies with position.
        print("\nmedian nearest-blob distance, by canvas row band (top→bottom):")
        bys = base[:, 1]
        for lo in range(0, 1050, 105):
            hi = lo + 105
            mm = (bys >= lo) & (bys < hi)
            if mm.sum() > 10:
                print(f"  y=[{lo:4d},{hi:4d})  n={int(mm.sum()):4d}  median={np.median(min_d[mm]):.2f}  mean={min_d[mm].mean():.2f}")

        # For each projected cell, compute the OFFSET to its nearest blob in image space.
        vectors = all_blob[arg_d] - projected   # (N, 2)
        print(f"\nmean offset (projected → nearest blob): dx={vectors[:,0].mean():+.2f}  dy={vectors[:,1].mean():+.2f}")
        print(f"stddev offset:                        dx={vectors[:,0].std():.2f}    dy={vectors[:,1].std():.2f}")

        # If there's a systematic offset (bias), fiducials might be detected at the ring
        # centre but the SVG anchor is at some slightly different location.
        # Shift the homography and re-measure.
        bias = vectors.mean(axis=0)
        shifted = projected + bias
        dists2 = np.sqrt(((shifted[:, None, :] - all_blob[None, :, :]) ** 2).sum(-1))
        min_d2 = dists2.min(axis=1)
        print(f"\nafter subtracting mean bias: median={np.median(min_d2):.2f}  mean={min_d2.mean():.2f}")


if __name__ == "__main__":
    main()
