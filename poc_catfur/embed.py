"""Thin wrapper around OpenAI's text-embedding-3-large.

Returns L2-normalised embeddings. The `dimensions` arg asks the API for a
Matryoshka truncation — we use that to get 24-d vectors directly.
"""

import os
import time
from functools import cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL_NAME = "text-embedding-3-large"
NATIVE_DIM = 3072
BATCH_SIZE = 512
_RETRIABLE = (APIConnectionError, APITimeoutError, RateLimitError)


@cache
def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env at the repo root.")
    return OpenAI(api_key=key, timeout=60.0, max_retries=0)


def _embed_batch(batch: list[str], dimensions: int | None) -> list[list[float]]:
    kwargs: dict = {"model": MODEL_NAME, "input": batch}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    for attempt in range(5):
        try:
            resp = _client().embeddings.create(**kwargs)
            return [d.embedding for d in resp.data]
        except _RETRIABLE as e:
            if attempt == 4:
                raise
            wait = 2**attempt
            print(f"    retry in {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def embed(
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    show_progress: bool = False,
    dimensions: int | None = None,
) -> np.ndarray:
    """L2-normalised embeddings. Shape (N, dimensions or NATIVE_DIM)."""
    out: list[list[float]] = []
    total = len(texts)
    t0 = time.time()
    for i in range(0, total, batch_size):
        out.extend(_embed_batch(texts[i : i + batch_size], dimensions))
        if show_progress:
            done = min(i + batch_size, total)
            print(f"  embed {done:>6}/{total}  ({time.time() - t0:.1f}s)", flush=True)
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr
