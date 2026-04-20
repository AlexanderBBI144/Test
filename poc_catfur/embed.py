"""Thin wrapper around the multilingual MiniLM sentence encoder.

Only produces raw 384-dim embeddings; the vocabulary/PCA layer lives in
`vocab.py`.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str], batch_size: int = 256, show_progress: bool = False) -> np.ndarray:
    """L2-normalised sentence/word embeddings, shape (N, 384)."""
    return np.asarray(
        get_model().encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        ),
        dtype=np.float32,
    )
