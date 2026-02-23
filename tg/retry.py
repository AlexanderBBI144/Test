"""Reusable flood-wait retry logic for Telegram API calls.

Centralises the ``FloodWaitError`` → sleep → retry pattern so that
every call-site stays DRY.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

from telethon import errors

from tg.config import cfg

log = logging.getLogger(__name__)

_MAX_RETRIES = 4

T = TypeVar("T")


async def with_flood_retry(
    call: Callable[[], Awaitable[T]],
    *,
    retries: int = _MAX_RETRIES,
    label: str = "request",
) -> T:
    """Invoke *call()* with automatic ``FloodWaitError`` retry.

    *call* must be a **zero-argument** callable that returns an awaitable
    (typically a ``lambda`` wrapping the real coroutine)::

        result = await with_flood_retry(
            lambda: client.send_message(entity, text),
            label="send_message",
        )

    Each retry creates a fresh coroutine, so the operation is safe to
    repeat.
    """
    for attempt in range(1, retries + 1):
        try:
            return await call()
        except errors.FloodWaitError as exc:
            if attempt == retries:
                raise
            wait = max(exc.seconds, cfg.rate_limit_pause)
            log.warning(
                "FloodWait %ds on %s (attempt %d/%d) — sleeping…",
                wait,
                label,
                attempt,
                retries,
            )
            await asyncio.sleep(wait)
    # Unreachable — the last attempt either returns or re-raises.
    raise errors.FloodWaitError(request=None, capture=0)
