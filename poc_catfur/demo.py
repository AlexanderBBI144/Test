"""End-to-end demo: text -> vector -> blob image -> distort -> decode -> text.

Run:  python3 -m poc_catfur.demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import decoder, distort, embed, encoder
from .corpus import CORPUS

OUT = Path(__file__).parent / "out"


def _roundtrip(
    text: str, bundle: dict, level: str | None
) -> tuple[str, float, float, list[tuple[str, float]], Image.Image, dict]:
    v = embed.project(text, bundle)
    img = encoder.encode(v)
    if level is not None:
        img = distort.distort(img, seed=abs(hash((text, level))) % 2**32, level=level)
    v_hat, diag = decoder.decode(img)
    matches = embed.nearest(v_hat, bundle, k=3)
    cos_to_orig = float(np.dot(v_hat, v))
    top_text, top_sim = matches[0]
    return top_text, top_sim, cos_to_orig, matches, img, diag


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building / loading embedding projection...")
    bundle = embed.load_or_build(CORPUS)

    probes = [
        "я люблю гусей",
        "обожаю шоколад",
        "я ненавижу пауков",
        "светит солнце",
        "слушаю тихую музыку",
    ]

    levels = [None, "light", "medium", "heavy"]

    print()
    header = f"{'phrase':<28} {'level':<7} {'cos(orig)':<10} {'top match':<30} {'sim'}"
    print(header)
    print("-" * len(header))

    for text in probes:
        for level in levels:
            top_text, top_sim, cos_orig, matches, img, diag = _roundtrip(text, bundle, level)
            tag = level or "clean"
            safe = text.replace(" ", "_")[:25]
            img.save(OUT / f"{safe}__{tag}.png")
            print(f"{text:<28} {tag:<7} {cos_orig:>8.3f}   {top_text:<30} {top_sim:.3f}")
        print()


if __name__ == "__main__":
    main()
