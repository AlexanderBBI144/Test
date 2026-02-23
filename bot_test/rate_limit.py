"""Rate-limit retry helpers for Telegram API calls."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import TypeVar, Callable, Awaitable, ParamSpec

from telethon.errors import FloodWaitError

log = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


async def retry_on_flood(
    coro_fn: Callable[P, Awaitable[T]],
    *args: P.args,
    max_retries: int = 3,
    **kwargs: P.kwargs,
) -> T:
    """Call *coro_fn* and automatically retry on ``FloodWaitError``.

    Waits the amount of seconds Telegram asks for, up to *max_retries* times.
    """
    for attempt in range(1, max_retries + 2):  # +1 for initial attempt
        try:
            return await coro_fn(*args, **kwargs)
        except FloodWaitError as exc:
            if attempt > max_retries:
                raise
            log.warning(
                "FloodWaitError: sleeping %ds (attempt %d/%d)",
                exc.seconds,
                attempt,
                max_retries,
            )
            await asyncio.sleep(exc.seconds)
    # unreachable, but keeps mypy happy
    raise RuntimeError("retry_on_flood: unreachable")


def rate_limited(max_retries: int = 3) -> Callable:
    """Decorator that wraps an async function with flood-wait retry logic."""

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await retry_on_flood(fn, *args, max_retries=max_retries, **kwargs)

        return wrapper

    return decorator
