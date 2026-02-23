"""Configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Config:
    api_id: int
    api_hash: str
    session: str
    bot_username: str
    timeout: float
    rate_limit_retries: int

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> Config:
        load_dotenv(dotenv_path or ".env")

        api_id_raw = os.environ.get("TG_API_ID", "")
        if not api_id_raw:
            raise RuntimeError("TG_API_ID is not set")

        api_hash = os.environ.get("TG_API_HASH", "")
        if not api_hash:
            raise RuntimeError("TG_API_HASH is not set")

        session = os.environ.get("TG_SESSION", "tg_test_session")

        bot_username = os.environ.get("TG_BOT_USERNAME", "")
        if not bot_username:
            raise RuntimeError("TG_BOT_USERNAME is not set")

        return cls(
            api_id=int(api_id_raw),
            api_hash=api_hash,
            session=session,
            bot_username=bot_username.lstrip("@"),
            timeout=float(os.environ.get("TG_TIMEOUT", "10")),
            rate_limit_retries=int(os.environ.get("TG_RATE_LIMIT_RETRIES", "3")),
        )
