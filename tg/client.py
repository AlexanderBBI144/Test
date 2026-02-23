"""Rate-limit-aware Telethon client wrapper."""

from __future__ import annotations

import logging
from typing import Any

from telethon import TelegramClient

from tg.config import cfg
from tg.retry import with_flood_retry

log = logging.getLogger(__name__)


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

    async def __call__(self, request: Any, *, retries: int = 4) -> Any:
        """Execute a raw Telethon *request* with automatic flood-wait retry.

        Usage::

            result = await tc(functions.messages.GetBotCallbackAnswerRequest(...))
        """
        return await with_flood_retry(
            lambda: self._client(request),
            retries=retries,
            label="raw_request",
        )

    # ------------------------------------------------------------------
    # Convenience send with retry
    # ------------------------------------------------------------------

    async def send(self, entity: Any, text: str, **kwargs: Any) -> Any:
        """Send a text message with automatic flood-wait retry."""
        return await with_flood_retry(
            lambda: self._client.send_message(entity, text, **kwargs),
            label="send_message",
        )

    async def send_file(self, entity: Any, file: Any, **kwargs: Any) -> Any:
        """Send a file/photo with automatic flood-wait retry."""
        return await with_flood_retry(
            lambda: self._client.send_file(entity, file, **kwargs),
            label="send_file",
        )

    # ------------------------------------------------------------------
    # Direct access to underlying client
    # ------------------------------------------------------------------

    @property
    def raw(self) -> TelegramClient:
        return self._client
