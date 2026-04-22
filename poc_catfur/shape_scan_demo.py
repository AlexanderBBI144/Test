"""End-to-end simulation of the print + phone-photo scan pipeline.

Run:  uv run --extra poc python -m poc_catfur.shape_scan_demo

Pipeline
--------
1. Embed "i love geese" via ada-002 → v_orig
2. shape_encoder.encode(v_orig) → svg
3. Render svg → PNG at 300 DPI (svglib + reportlab.graphics.renderPM)
4. Invert grayscale so dark symbols sit on a white background
5. Apply mild perspective warp + Gaussian blur + pixel noise (simulated photo)
6. Save PNG to out/scan_i_love_geese.png
7. shape_scan_decoder.decode_image(path, svg) → v_rt
8. Compare cos(v_rt, v_orig) to target ≥ 0.85
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI
from PIL import Image, ImageFilter
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

from . import shape_decoder as dec
from . import shape_encoder as enc
from . import shape_scan_decoder as sdec

OUT = Path(__file__).parent / "out"
ADA = "text-embedding-ada-002"
PROBE = "i love geese"
DPI = 300
WARP_FRAC = 0.012   # corner jitter as fraction of image dimension
BLUR_SIGMA_FRAC = 0.0005   # ~1.5 px at 300 DPI; phone cameras at this resolution
NOISE_STD = 3.0     # graylevels
SEED = 0


def embed_ada(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=ADA, input=[text])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


def render_svg_to_array(svg: str, dpi: int) -> np.ndarray:
    """Rasterise SVG → grayscale uint8 array at the given DPI."""
    drawing = svg2rlg(io.StringIO(svg))
    buf = io.BytesIO()
    renderPM.drawToFile(drawing, buf, fmt="PNG", dpi=dpi)
    buf.seek(0)
    img = Image.open(buf).convert("L")
    return np.array(img, dtype=np.uint8)


def perspective_coefs(src_corners, dst_corners):
    """Return PIL.Image.transform PERSPECTIVE coefficients mapping dst→src."""
    A = []
    b = []
    for (x_s, y_s), (x_d, y_d) in zip(src_corners, dst_corners):
        A.append([x_d, y_d, 1, 0, 0, 0, -x_d * x_s, -y_d * x_s])
        A.append([0, 0, 0, x_d, y_d, 1, -x_d * y_s, -y_d * y_s])
        b.append(x_s)
        b.append(y_s)
    coefs, *_ = np.linalg.lstsq(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64), rcond=None)
    return tuple(coefs.tolist())


def simulate_photo(img_arr: np.ndarray, rng: np.random.Generator) -> Image.Image:
    """Apply perspective warp + blur + noise to simulate a flat phone photo.

    The output image is padded by `jitter_px` on each side so that corner
    fiducials cannot spill outside the frame and get clipped — clipping shifts
    their detected centroids by up to tens of pixels and breaks the homography
    recovered by the scan decoder.
    """
    h, w = img_arr.shape
    jitter_px = WARP_FRAC * min(w, h)
    pad = int(np.ceil(jitter_px)) + 2
    out_w = w + 2 * pad
    out_h = h + 2 * pad

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (pad + rng.uniform(-jitter_px, jitter_px), pad + rng.uniform(-jitter_px, jitter_px)),
        (pad + w + rng.uniform(-jitter_px, jitter_px), pad + rng.uniform(-jitter_px, jitter_px)),
        (pad + w + rng.uniform(-jitter_px, jitter_px), pad + h + rng.uniform(-jitter_px, jitter_px)),
        (pad + rng.uniform(-jitter_px, jitter_px), pad + h + rng.uniform(-jitter_px, jitter_px)),
    ]
    coefs = perspective_coefs(src, dst)

    pil = Image.fromarray(img_arr, mode="L")
    warped = pil.transform(
        (out_w, out_h),
        Image.Transform.PERSPECTIVE,
        coefs,
        resample=Image.Resampling.BILINEAR,
        fillcolor=255,
    )

    blurred = warped.filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA_FRAC * w))

    arr = np.array(blurred, dtype=np.float32)
    arr += rng.normal(0.0, NOISE_STD, size=arr.shape)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Embedding '{PROBE}' via ada-002…")
    client = OpenAI()
    v_orig = embed_ada(client, PROBE)

    print("Encoding to SVG…")
    svg = enc.encode(v_orig)
    svg_path = OUT / "scan_i_love_geese.svg"
    svg_path.write_text(svg, encoding="utf-8")

    # Sanity: SVG round-trip (no print path).
    v_rt_digital = dec.decode(svg)
    cos_digital = cosine(v_orig, v_rt_digital)

    print(f"Rendering SVG at {DPI} DPI…")
    arr_render = render_svg_to_array(svg, DPI)
    # Encoder puts BG=#0E142C (dark) with FG=#1FD9FE (bright symbols). The scan
    # decoder expects dark symbols on a light background, so invert.
    arr_ink = 255 - arr_render

    print("Simulating photo (perspective + blur + noise)…")
    photo = simulate_photo(arr_ink, rng)
    photo_path = OUT / "scan_i_love_geese.png"
    photo.save(photo_path)

    # Diagnostic: dark-pixel fraction pre- and post-simulation.
    dark_pre = float((arr_ink < 128).mean())
    dark_post = float((np.array(photo) < 128).mean())
    print(f"  dark-px fraction: render={dark_pre:.4f}  photo={dark_post:.4f}")

    print("Decoding scan…")
    v_rt_scan = sdec.decode_image(photo_path)
    cos_scan_vs_orig = cosine(v_orig, v_rt_scan)
    cos_scan_vs_digital = cosine(v_rt_digital, v_rt_scan)

    print()
    print(f"{'phrase':<24} {PROBE}")
    print(f"{'cos_digital':<24} {cos_digital:.4f}   (SVG → shape_decoder)")
    print(f"{'cos_scan_vs_digital':<24} {cos_scan_vs_digital:.4f}   (PNG decode vs. clean decode)")
    print(f"{'cos_scan_vs_orig':<24} {cos_scan_vs_orig:.4f}   (PNG decode vs. original ada-002)")
    target = 0.85
    verdict = "PASS" if cos_scan_vs_orig >= target else "FAIL"
    print(f"{'target':<24} ≥ {target}   → {verdict}")
    print()
    print(f"Artifacts: {svg_path}  {photo_path}")


if __name__ == "__main__":
    main()
