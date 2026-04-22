"""Diagnostic: decode the real printed photo and dump intermediate state."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI

from . import shape_decoder as dec
from . import shape_encoder as enc
from . import shape_scan_decoder as sdec

PHOTO = Path(__file__).resolve().parents[1] / "photo_2026-04-22_01-04-37.jpg"
SVG_PATH = Path(__file__).parent / "out" / "scan_i_love_geese.svg"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cyan_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (g + b) * 0.5 - r


def main():
    svg = SVG_PATH.read_text()
    print(f"SVG: {SVG_PATH}  ({len(svg)} bytes)")

    rgb = np.array(Image.open(PHOTO).convert("RGB"))
    print(f"photo: {rgb.shape}")
    cyan = cyan_mask(rgb)
    print(f"cyan score: min={cyan.min():.1f} max={cyan.max():.1f} mean={cyan.mean():.2f}")

    # Threshold cyan to binary ink mask.
    thresh = 20.0
    binary = cyan > thresh
    print(f"binary (cyan>{thresh}): {binary.sum()} ink px  ({binary.mean()*100:.2f}%)")

    # Simulated-grayscale view for the decoder's _ink_score (high cyan → dark gray).
    ink_gray = (255 - np.clip(cyan * 4.0, 0, 255)).astype(np.uint8)

    # Fiducial detection — what the decoder sees.
    img_fid = sdec._detect_fiducials(binary)
    print(f"\nfiducials (img coords):")
    for (cx, cy), (name, svg_c) in zip(img_fid, zip(["TL", "TR", "BL", "BR"], enc._FID_CENTERS)):
        print(f"  {name:3s} svg={svg_c}  img=({cx:.1f}, {cy:.1f})")

    # Build homography and check forward-mapped canvas corners.
    svg_fid = np.array(enc._FID_CENTERS, dtype=np.float64)
    H = sdec._compute_homography(svg_fid, img_fid)
    corners_svg = np.array([[0, 0], [enc.CANVAS_W, 0], [enc.CANVAS_W, enc.CANVAS_H], [0, enc.CANVAS_H]], dtype=np.float64)
    corners_img = sdec._apply_homography(H, corners_svg)
    print("\nforward-mapped canvas corners:")
    for name, (x, y) in zip(["TL", "TR", "BR", "BL"], corners_img):
        print(f"  {name}: ({x:.1f}, {y:.1f})")

    # Scale at origin — how many img px per SVG px?
    svg_unit = sdec._apply_homography(H, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    sx = float(np.linalg.norm(svg_unit[1] - svg_unit[0]))
    sy = float(np.linalg.norm(svg_unit[2] - svg_unit[0]))
    print(f"\nhomography scale: {sx:.3f} img_px / svg_px (x), {sy:.3f} (y)")
    print(f"patch_r = max(2, round(0.4 * 2.0 * {sx:.3f})) = {max(2, int(round(0.4 * 2.0 * sx)))}")
    print(f"jitter candidate spacing in img: {enc._JITTER_STEP * sx:.2f} img_px")

    # Skip the direct decode (dark navy BG breaks img<128 threshold).
    # Also: decode from the reference SVG (sanity).
    v_svg = dec.decode(svg)

    # Ground truth embedding.
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-ada-002", input=["i love geese"])
    v_orig = np.array(resp.data[0].embedding, dtype=np.float32)
    v_orig /= np.linalg.norm(v_orig) + 1e-12

    print(f"\ncos(svg_decoder, orig)  = {cosine(v_svg, v_orig):.4f}")

    # Try with a CYAN-based decode path: rebuild the grayscale view and push through decode_image,
    # but decode_image opens the file itself. Monkey-patch path: we pass the cyan ink_gray
    # through by saving a temp PNG.
    tmp = PHOTO.parent / "_diag_cyan_gray.png"
    Image.fromarray(ink_gray, mode="L").save(tmp)
    v_scan_cyan = sdec.decode_image(tmp.as_posix(), svg)
    print(f"cos(scan_cyan,    orig) = {cosine(v_scan_cyan, v_orig):.4f}")
    print(f"cos(scan_cyan,    svg)  = {cosine(v_scan_cyan, v_svg):.4f}")

    # Agreement at the index level: where do the 16-level winners match?
    from .shape_encoder import float_to_idx
    idx_svg = np.array([float_to_idx(float(f)) for f in v_svg])
    idx_scan = np.array([float_to_idx(float(f)) for f in v_scan_cyan])
    agree = int((idx_svg == idx_scan).sum())
    print(f"\nidx agreement (scan_cyan vs svg): {agree} / {len(idx_svg)}  ({agree/len(idx_svg)*100:.1f}%)")

    # What's the distribution of scan idx?
    u, c = np.unique(idx_scan, return_counts=True)
    print(f"scan idx histogram: {dict(zip(u.tolist(), c.tolist()))}")
    u, c = np.unique(idx_svg, return_counts=True)
    print(f"svg  idx histogram: {dict(zip(u.tolist(), c.tolist()))}")

    # Per-cell ink analysis: project each data cell's expected (winning) position
    # into image space and check ink there.
    cells = sdec._parse_data_cells(svg)
    # Use expected GROUND TRUTH jitter from the SVG.
    offsets = np.array(
        [(ix * enc._JITTER_STEP + enc._JITTER_BASE, iy * enc._JITTER_STEP + enc._JITTER_BASE)
         for iy in range(enc._JITTER_LEVELS)
         for ix in range(enc._JITTER_LEVELS)],
        dtype=np.float64,
    )
    ink_gray_arr = ink_gray  # alias
    img_arr = np.array(Image.open(tmp).convert("L"), dtype=np.uint8)

    patch_r = 2
    gt_scores = []
    max_scores = []
    n_candidates_with_ink = []
    best_vs_gt = []
    for cell_i, bx, by in cells:
        gt_idx = int(idx_svg[cell_i])
        svg_cands = np.column_stack([
            np.full(len(offsets), bx) + offsets[:, 0],
            np.full(len(offsets), by) + offsets[:, 1],
        ])
        img_cands = sdec._apply_homography(H, svg_cands)
        scores = np.array([
            sdec._ink_score(img_arr, cx, cy, patch_r)
            for cx, cy in img_cands
        ])
        gt_scores.append(scores[gt_idx])
        max_scores.append(scores.max())
        n_candidates_with_ink.append(int((scores > 0.05).sum()))
        best_vs_gt.append(scores[int(np.argmax(scores))] - scores[gt_idx])

    gt_scores = np.asarray(gt_scores)
    max_scores = np.asarray(max_scores)
    n_cand_ink = np.asarray(n_candidates_with_ink)

    print("\n── per-cell diagnostic ─────────────────────────────────────")
    print(f"cells w/ ZERO max ink  : {int((max_scores < 0.01).sum())} / {len(cells)}")
    print(f"cells w/ ground-truth < 0.05 ink : {int((gt_scores < 0.05).sum())} / {len(cells)}")
    print(f"mean max_score across cells     : {max_scores.mean():.3f}")
    print(f"mean gt_score  across cells     : {gt_scores.mean():.3f}")
    print(f"cells where argmax == gt        : "
          f"{int(((max_scores - gt_scores) < 1e-6).sum())} / {len(cells)}")

    # Among cells WHERE MAX INK EXISTS (> 0.05), how often is argmax == gt?
    mask = max_scores > 0.05
    if mask.sum():
        agree_when_inked = int(((max_scores[mask] - gt_scores[mask]) < 1e-6).sum())
        print(f"among {int(mask.sum())} inked cells: argmax==gt in {agree_when_inked} ({agree_when_inked/int(mask.sum())*100:.1f}%)")

    print(f"\nhist of n_candidates_with_ink (>0.05): {dict(zip(*[x.tolist() for x in np.unique(n_cand_ink, return_counts=True)]))}")


if __name__ == "__main__":
    main()
