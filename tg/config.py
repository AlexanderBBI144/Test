"""Configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env(key: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Environment variable {key} is required but not set")
    return val  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Config:
    api_id: int = field(default_factory=lambda: int(_env("TG_API_ID", required=True)))
    api_hash: str = field(default_factory=lambda: _env("TG_API_HASH", required=True))
    session: str = field(default_factory=lambda: _env("TG_SESSION", "test_user"))
    bot_username: str = field(default_factory=lambda: _env("TG_BOT_USERNAME", required=True))
    default_timeout: float = field(
        default_factory=lambda: float(_env("TG_DEFAULT_TIMEOUT", "10"))
    )
    rate_limit_pause: float = field(
        default_factory=lambda: float(_env("TG_RATE_LIMIT_PAUSE", "5"))
    )


cfg = Config()
