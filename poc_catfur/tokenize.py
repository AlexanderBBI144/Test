"""Input inspection helpers using tiktoken's cl100k_base encoding."""

import re
from functools import cache

import tiktoken

_WS = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z]+(?:['-][a-z]+)*", re.IGNORECASE)


@cache
def _tok() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def subword_tokens(text: str) -> list[str]:
    tok = _tok()
    return [
        tok.decode_single_token_bytes(i).decode("utf-8", errors="replace")
        for i in tok.encode(normalize(text))
    ]


def words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]
