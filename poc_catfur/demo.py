"""End-to-end demo: text -> blob image -> distort -> decode -> phrase.

Run:  uv run --extra poc python -m poc_catfur.demo
"""

from pathlib import Path

import numpy as np
from PIL import Image

from . import decoder, distort, encoder, tokenize, vocab

OUT = Path(__file__).parent / "out"

PROBES = [
    "i love geese",
    "chocolate is delicious",
    "i hate spiders",
    "the sun is shining",
    "listening to quiet music",
    "the train is late",
    "mountains are beautiful",
    "she writes a long letter",
]
LEVELS: list[str | None] = [None, "light", "medium", "heavy"]


def _roundtrip(
    text: str, v: np.ndarray, vb: dict, level: str | None
) -> tuple[list[tuple[str, float]], float, Image.Image]:
    img = encoder.encode(v, vb["v_lo"], vb["v_hi"])
    if level is not None:
        img = distort.distort(img, seed=abs(hash((text, level))) % 2**32, level=level)
    v_hat, _ = decoder.decode(img, vb["v_lo"], vb["v_hi"])
    return vocab.lookup_phrases(v_hat, vb, k=3), float(np.dot(v_hat, v)), img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading / building phrase bank...")
    vb = vocab.load_or_build()
    print(f"phrases: {len(vb['phrases'])}, dim={vocab.DIM}\n")

    for text in PROBES:
        norm = tokenize.normalize(text)
        print(f"\ninput:       {text!r}")
        print(f"normalized:  {norm!r}")
        print(f"subwords:    {tokenize.subword_tokens(text)}")
        print(f"words:       {tokenize.words(text)}")

        v = vocab.project(norm, vb)
        for level in LEVELS:
            matches, cos_orig, img = _roundtrip(norm, v, vb, level)
            tag = level or "clean"
            safe = "".join(c if c.isalnum() else "_" for c in norm)[:30]
            img.save(OUT / f"{safe}__{tag}.png")
            top3 = ", ".join(f"{w}({s:.2f})" for w, s in matches)
            print(f"  {tag:<7} cos={cos_orig:.3f}  {top3}")


if __name__ == "__main__":
    main()
