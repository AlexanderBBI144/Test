"""Embedding layer: text -> unit vector on a d-dim sphere, using a
multilingual sentence encoder + PCA whitening fit on a closed corpus.

The embedding's semantic-smoothness is what makes the whole scheme robust:
small perturbations after the round-trip through printing and photographing
stay close to the original in cosine distance, so the nearest-neighbour
lookup still returns a phrase with similar meaning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 16
CACHE = Path(__file__).parent / ".cache" / "projection.npz"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_texts(texts: list[str]) -> np.ndarray:
    return np.asarray(_get_model().encode(texts, normalize_embeddings=True))


def build(corpus: list[str]) -> dict:
    """Fit PCA on the corpus, return a dict of arrays ready to save."""
    E = encode_texts(corpus)
    pca = PCA(n_components=DIM)
    pca.fit(E)
    Z = pca.transform(E)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "corpus": np.array(corpus),
        "Z": Z.astype(np.float32),
    }


def save(bundle: dict, path: Path = CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bundle)


def load(path: Path = CACHE) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "mean": d["mean"],
        "components": d["components"],
        "corpus": list(d["corpus"]),
        "Z": d["Z"],
    }


def load_or_build(corpus: list[str], path: Path = CACHE) -> dict:
    if path.exists():
        return load(path)
    bundle = build(corpus)
    save(bundle, path)
    return bundle


def project(text: str, bundle: dict) -> np.ndarray:
    e = encode_texts([text])[0]
    z = bundle["components"] @ (e - bundle["mean"])
    return z / (np.linalg.norm(z) + 1e-12)


def nearest(v: np.ndarray, bundle: dict, k: int = 3) -> list[tuple[str, float]]:
    Z = bundle["Z"]
    sims = Z @ v
    order = np.argsort(-sims)[:k]
    return [(bundle["corpus"][int(i)], float(sims[int(i)])) for i in order]
