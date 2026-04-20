"""End-to-end demo: English text -> blob image -> distort -> decode -> word.

Run:  uv run --extra poc python -m poc_catfur.demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import decoder, distort, encoder, tokenize, vocab

OUT = Path(__file__).parent / "out"


def _roundtrip(
    text: str, v: np.ndarray, vocab_bundle: dict, level: str | None
) -> tuple[list[tuple[str, float]], float, Image.Image, dict]:
    img = encoder.encode(v, vocab_bundle["v_lo"], vocab_bundle["v_hi"])
    if level is not None:
        img = distort.distort(img, seed=abs(hash((text, level))) % 2**32, level=level)
    v_hat, diag = decoder.decode(img, vocab_bundle["v_lo"], vocab_bundle["v_hi"])
    matches = vocab.lookup(v_hat, vocab_bundle, k=3)
    cos_orig = float(np.dot(v_hat, v))
    return matches, cos_orig, img, diag


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading / building 20k-word English vocabulary...")
    vb = vocab.load_or_build()
    print(f"vocab: {len(vb['words'])} words, PCA dim={vocab.DIM}")
    print()

    probes = [
        "i love geese",
        "chocolate is delicious",
        "i hate spiders",
        "the sun is shining",
        "listening to quiet music",
        "the train is late",
        "mountains are beautiful",
        "she writes a long letter",
    ]

    levels = [None, "light", "medium", "heavy"]

    print(f"{'phrase':<32} {'level':<7} {'cos':<7} {'top-3 decoded words'}")
    print("-" * 96)

    for text in probes:
        norm = tokenize.normalize(text)
        toks = tokenize.subword_tokens(text)
        wrds = tokenize.words(text)
        print(f"\ninput:       {text!r}")
        print(f"normalized:  {norm!r}")
        print(f"subwords:    {toks}")
        print(f"words:       {wrds}")

        v = vocab.project(norm, vb)

        for level in levels:
            matches, cos_orig, img, diag = _roundtrip(norm, v, vb, level)
            tag = level or "clean"
            safe = "".join(c if c.isalnum() else "_" for c in norm)[:30]
            img.save(OUT / f"{safe}__{tag}.png")
            top3 = ", ".join(f"{w}({s:.2f})" for w, s in matches)
            print(f"  {tag:<7} cos={cos_orig:.3f}  {top3}")


if __name__ == "__main__":
    main()
