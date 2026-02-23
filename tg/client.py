"""Rate-limit-aware Telethon client wrapper."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from telethon import TelegramClient, errors, functions

from tg.config import cfg

log = logging.getLogger(__name__)

_MAX_RETRIES = 4


class TelegramTestClient:
    """Thin wrapper around :class:`TelegramClient` that transparently retries
    on ``FloodWaitError`` and exposes a simplified API for tests.
    """

    def __init__(self, client: TelegramClient | None = None) -> None:
        self._client = client or TelegramClient(
            cfg.session, cfg.api_id, cfg.api_hash
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._client.start()
        me = await self._client.get_me()
        log.info("Logged in as %s (id=%s)", me.first_name, me.id)

    async def stop(self) -> None:
        await self._client.disconnect()

    # ------------------------------------------------------------------
    # Rate-limited request helper
    # ------------------------------------------------------------------

    async def __call__(self, request: Any, *, retries: int = _MAX_RETRIES) -> Any:
        """Execute a raw Telethon *request* with automatic flood-wait retry.

        Usage::

            result = await tc(functions.messages.GetBotCallbackAnswerRequest(...))
        """
        for attempt in range(1, retries + 1):
            try:
                return await self._client(request)
            except errors.FloodWaitError as exc:
                wait = max(exc.seconds, cfg.rate_limit_pause)
                log.warning(
                    "FloodWait %ds (attempt %d/%d) — sleeping…",
                    wait,
                    attempt,
                    retries,
                )
                await asyncio.sleep(wait)
        raise errors.FloodWaitError(
            request=request,
            capture=0,
        )

    # ------------------------------------------------------------------
    # Convenience send with retry
    # ------------------------------------------------------------------

    async def send(self, entity: Any, text: str, **kwargs: Any) -> Any:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await self._client.send_message(entity, text, **kwargs)
            except errors.FloodWaitError as exc:
                wait = max(exc.seconds, cfg.rate_limit_pause)
                log.warning("FloodWait on send (%ds, attempt %d)", wait, attempt)
                await asyncio.sleep(wait)
        raise errors.FloodWaitError(request=None, capture=0)

    # ------------------------------------------------------------------
    # Direct access to underlying client
    # ------------------------------------------------------------------

    @property
    def raw(self) -> TelegramClient:
        return self._client
