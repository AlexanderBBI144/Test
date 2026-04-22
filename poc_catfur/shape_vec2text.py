"""vec2text wrapper: approximate embedding → top-N phrase candidates.

Uses the cached corrector model (ada-002 / msmarco / msl128).
Lazy-loads on first call; subsequent calls reuse the loaded model.

Usage
-----
    from poc_catfur.shape_vec2text import top_phrases
    phrases = top_phrases(embedding_1536d, n=10)
"""

from __future__ import annotations

import numpy as np

_corrector = None
_MODEL = "jxm/vec2text__openai_ada002__msmarco__msl128__corrector"


def _get_corrector():
    global _corrector
    if _corrector is None:
        import vec2text
        _corrector = vec2text.load_corrector(_MODEL)
    return _corrector


def top_phrases(embedding: np.ndarray, n: int = 10) -> list[str]:
    """Return up to n decoded phrase candidates for a 1536-d ada-002 embedding."""
    import torch
    import vec2text

    corrector = _get_corrector()
    emb = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    results = vec2text.invert_embeddings(
        embeddings=emb,
        corrector=corrector,
        num_steps=20,
        sequence_beam_width=n,
    )
    if results and isinstance(results[0], list):
        return results[0][:n]
    return list(results)[:n]
