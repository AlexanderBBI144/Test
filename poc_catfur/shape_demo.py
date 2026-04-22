"""
Round-trip demo for the Shape+Texture Hybrid codec.

Run:  uv run --extra poc python -m poc_catfur.shape_demo

Outputs:
  poc_catfur/out/shape_<phrase>.svg   – one SVG per probe phrase
  Summary table: per-component MAE and cosine similarity.
"""

from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI

from . import shape_encoder as enc
from . import shape_decoder as dec

OUT = Path(__file__).parent / "out"
ADA = "text-embedding-ada-002"

PROBES = [
    "i love geese",
    "chocolate is delicious",
    "the sky is blue",
    "geese are not cats but birds",
    "i hate spiders",
    "the sun is shining",
]


def embed_ada(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=ADA, input=texts)
    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    print("Embedding probes via ada-002…")
    vectors = embed_ada(client, PROBES)

    print(f"\n{'phrase':<32} {'cos_rt':>7}  {'MAE':>9}")
    print("-" * 52)

    for phrase, v_orig in zip(PROBES, vectors):
        svg = enc.encode(v_orig)

        safe = "".join(c if c.isalnum() else "_" for c in phrase)[:28]
        (OUT / f"shape_{safe}.svg").write_text(svg, encoding="utf-8")

        v_rt = dec.decode(svg)
        cos = float(np.dot(v_orig, v_rt) / (np.linalg.norm(v_rt) + 1e-12))
        mae = float(np.abs(v_orig - v_rt).mean())

        print(f"{phrase:<32} {cos:>7.5f}  {mae:>9.2e}")

    print(f"\nSVGs written to {OUT}/")
    print("\nOpen any .svg in a browser to preview the continent pattern.")


if __name__ == "__main__":
    main()
