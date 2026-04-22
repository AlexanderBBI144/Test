"""Phrase-bank bundle for the cat-fur codec.

OpenAI text-embedding-3-large at `dimensions=24` (Matryoshka truncation).
The ~5k-phrase bank is embedded once, per-dim [v_lo, v_hi] percentiles
feed the encoder's area calibration, and the bundle is cached to disk.
"""

import hashlib
from pathlib import Path

import numpy as np

from . import embed

DIM = 24
CACHE_DIR = Path(__file__).parent / ".cache"


def _cache_path(phrases: list[str]) -> Path:
    key = hashlib.sha1(
        ("|".join(phrases) + f"|dim={DIM}|{embed.MODEL_NAME}").encode("utf-8")
    ).hexdigest()[:10]
    return CACHE_DIR / f"phrases-{key}.npz"


def load_or_build() -> dict:
    from . import phrases as phrases_mod

    phrase_list = phrases_mod.generate()
    path = _cache_path(phrase_list)
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {
            "phrases": list(d["phrases"]),
            "phrases_Z": d["phrases_Z"],
            "v_lo": d["v_lo"],
            "v_hi": d["v_hi"],
        }

    print(f"Embedding {len(phrase_list)} phrases at dim={DIM}...", flush=True)
    Z = embed.embed(phrase_list, dimensions=DIM, show_progress=True)
    v_lo = np.percentile(Z, 1.0, axis=0).astype(np.float32)
    v_hi = np.percentile(Z, 99.0, axis=0).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, phrases=np.array(phrase_list), phrases_Z=Z, v_lo=v_lo, v_hi=v_hi)
    return {"phrases": phrase_list, "phrases_Z": Z, "v_lo": v_lo, "v_hi": v_hi}


def project(text: str, vocab: dict) -> np.ndarray:
    """Embed a single text at DIM dims, unit-normalised."""
    return embed.embed([text], dimensions=DIM)[0]


def lookup_phrases(v: np.ndarray, vocab: dict, k: int = 5) -> list[tuple[str, float]]:
    sims = vocab["phrases_Z"] @ v
    k = min(k, len(sims))
    order = np.argpartition(-sims, k - 1)[:k]
    order = order[np.argsort(-sims[order])]
    return [(str(vocab["phrases"][int(i)]), float(sims[int(i)])) for i in order]
