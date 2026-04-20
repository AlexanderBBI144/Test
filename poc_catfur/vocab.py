"""English-word vocabulary + PCA projection used by both the encoder
(for channel calibration) and the decoder (for nearest-neighbour lookup).

The vocabulary is the top-N most frequent English words from `wordfreq`,
filtered to alphabetic tokens of length >= 2. For each word we cache:

  * `E`        — (N, 384) raw MiniLM embeddings
  * `mean`,
    `components` — fitted PCA (k = DIM)
  * `Z`        — (N, DIM) unit-sphere projections
  * `v_lo`,
    `v_hi`     — per-dimension 1st/99th percentiles of Z, used to
                  linearly map values to the [0, 1] area range in the
                  blob channel.

Building takes ~1–2 min on CPU. The result is cached to .cache/vocab.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from wordfreq import top_n_list

from . import embed

DIM = 24
VOCAB_SIZE = 20000
CACHE = Path(__file__).parent / ".cache" / "vocab.npz"


def _fetch_words(n: int) -> list[str]:
    # Oversample from wordfreq, then filter to clean alphabetic tokens.
    raw = top_n_list("en", int(n * 1.4), wordlist="best")
    seen = set()
    clean: list[str] = []
    for w in raw:
        if len(w) >= 2 and w.isalpha():
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                clean.append(lw)
        if len(clean) >= n:
            break
    return clean


def build(n: int = VOCAB_SIZE, dim: int = DIM, show_progress: bool = True) -> dict:
    words = _fetch_words(n)
    print(f"Embedding {len(words)} English words...")
    E = embed.embed(words, show_progress=show_progress)
    pca = PCA(n_components=dim).fit(E)
    Z = pca.transform(E).astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    v_lo = np.percentile(Z, 1.0, axis=0).astype(np.float32)
    v_hi = np.percentile(Z, 99.0, axis=0).astype(np.float32)
    return {
        "words": np.array(words),
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "Z": Z,
        "v_lo": v_lo,
        "v_hi": v_hi,
    }


def save(bundle: dict, path: Path = CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bundle)


def load(path: Path = CACHE) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "words": list(d["words"]),
        "mean": d["mean"],
        "components": d["components"],
        "Z": d["Z"],
        "v_lo": d["v_lo"],
        "v_hi": d["v_hi"],
    }


def load_or_build(**kw) -> dict:
    if CACHE.exists():
        return load()
    bundle = build(**kw)
    save(bundle)
    return bundle


def project(text: str, vocab: dict) -> np.ndarray:
    e = embed.embed([text])[0]
    z = vocab["components"] @ (e - vocab["mean"])
    return (z / (np.linalg.norm(z) + 1e-12)).astype(np.float32)


def lookup(v: np.ndarray, vocab: dict, k: int = 5) -> list[tuple[str, float]]:
    sims = vocab["Z"] @ v
    order = np.argpartition(-sims, k)[:k]
    order = order[np.argsort(-sims[order])]
    return [(str(vocab["words"][int(i)]), float(sims[int(i)])) for i in order]
