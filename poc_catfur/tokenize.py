"""Proper tokenizer layer.

Wraps the MiniLM tokenizer (HuggingFace `tokenizers`) for input
inspection and normalisation. The tokenizer is used in three places:

  * `normalize(text)` — strip whitespace, lowercase, collapse spaces.
  * `subword_tokens(text)` — subword pieces (for diagnostics / display).
  * `words(text)` — whitespace-and-punctuation-free list of content
    tokens, used when mapping an input sentence to its dominant word.

The encoding channel itself feeds the *full* normalised string to the
sentence encoder, so the tokenizer is run implicitly there too.
"""
from __future__ import annotations

import re

from transformers import AutoTokenizer

from .embed import MODEL_NAME

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def subword_tokens(text: str) -> list[str]:
    """Return the model's own subword pieces, minus BOS/EOS."""
    tok = get_tokenizer()
    ids = tok.encode(normalize(text), add_special_tokens=False)
    return tok.convert_ids_to_tokens(ids)


_WORD_RE = re.compile(r"[a-z]+(?:['-][a-z]+)*", re.IGNORECASE)


def words(text: str) -> list[str]:
    """Alphabetic words from the input, lowercased."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]
