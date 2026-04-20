"""Simulate what a photographed T-shirt image looks like: folds, perspective,
blur, lighting, sensor noise and JPEG compression.

These are the nuisances the decoder has to be robust against.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def _fold(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vertical fold: introduce a ~20px horizontal shift along a random x-line,
    imitating fabric bunched up over a seam."""
    H, W = arr.shape
    fold_x = int(rng.uniform(0.3, 0.7) * W)
    shift = int(rng.uniform(15, 30))
    y, x = np.indices(arr.shape)
    new_x = np.where(x > fold_x, np.clip(x + shift, 0, W - 1), x)
    return arr[y, new_x]


def _perspective(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Apply a random mild perspective warp (quad corners jittered)."""
    W, H = img.size
    m = 0.06  # max corner displacement as fraction of side
    jitter = (rng.uniform(-m, m, size=(4, 2)) * np.array([W, H])).astype(np.float32)
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32) + jitter
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    # PIL wants coefficients for the inverse map dst -> src.
    A = []
    b = []
    for (x, y), (xp, yp) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y]); b.append(xp)
        A.append([0, 0, 0, x, y, 1, -yp * x, -yp * y]); b.append(yp)
    coeffs = np.linalg.solve(np.asarray(A), np.asarray(b))
    return img.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=255)


def distort(
    img: Image.Image,
    seed: int = 0,
    level: str = "medium",
) -> Image.Image:
    """level: 'light' | 'medium' | 'heavy'."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("L"))

    if level in ("medium", "heavy"):
        arr = _fold(arr, rng)

    pimg = Image.fromarray(arr)

    if level in ("medium", "heavy"):
        pimg = _perspective(pimg, rng)

    angle = rng.uniform(-10, 10) if level == "light" else rng.uniform(-20, 20)
    pimg = pimg.rotate(angle, fillcolor=255, resample=Image.BICUBIC)

    blur_r = {"light": 1.0, "medium": 2.0, "heavy": 3.0}[level]
    pimg = pimg.filter(ImageFilter.GaussianBlur(radius=blur_r))

    arr = np.asarray(pimg, dtype=np.float32)
    # uneven lighting: low-freq multiplicative gradient
    H, W = arr.shape
    y_grad = np.linspace(0.8, 1.2, H)[:, None]
    x_grad = np.linspace(1.1, 0.9, W)[None, :]
    arr = arr * (y_grad * x_grad)

    noise_sigma = {"light": 4.0, "medium": 8.0, "heavy": 14.0}[level]
    arr = arr + rng.normal(0, noise_sigma, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    pimg = Image.fromarray(arr)

    # JPEG compression round-trip
    quality = {"light": 85, "medium": 65, "heavy": 45}[level]
    buf = io.BytesIO()
    pimg.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L")
