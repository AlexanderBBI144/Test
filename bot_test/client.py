"""Thin wrapper around TelegramClient tailored for bot testing."""

from __future__ import annotations

import logging
from typing import Any

from telethon import TelegramClient
from telethon.tl.types import InputPeerUser, User

from bot_test.config import Config
from bot_test.rate_limit import retry_on_flood

log = logging.getLogger(__name__)


class TelegramTestClient:
    """High-level test-oriented Telegram client.

    Wraps :class:`TelegramClient` and adds:
    * automatic flood-wait retries on every API call,
    * convenient ``bot`` property for the entity under test,
    * simplified ``send`` / ``send_command`` helpers.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._client = TelegramClient(
            config.session,
            config.api_id,
            config.api_hash,
        )
        self._bot_entity: User | None = None

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        await self._client.start()
        self._bot_entity = await self._client.get_entity(self.cfg.bot_username)
        log.info("Connected. Bot entity: %s (id=%d)", self.bot.username, self.bot.id)

    async def stop(self) -> None:
        await self._client.disconnect()
        log.info("Disconnected.")

    async def __aenter__(self) -> TelegramTestClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # -- properties -----------------------------------------------------------

    @property
    def raw(self) -> TelegramClient:
        """Access the underlying ``TelegramClient`` when you need low-level API."""
        return self._client

    @property
    def bot(self) -> User:
        assert self._bot_entity is not None, "Client not started"
        return self._bot_entity

    @property
    def bot_peer(self) -> InputPeerUser:
        return InputPeerUser(self.bot.id, self.bot.access_hash)

    # -- messaging helpers (flood-safe) ---------------------------------------

    async def send(self, text: str) -> Any:
        """Send *text* to the bot under test (with flood-wait retry)."""
        return await retry_on_flood(
            self._client.send_message,
            self.bot,
            text,
            max_retries=self.cfg.rate_limit_retries,
        )

    async def send_command(self, command: str) -> Any:
        """Send a ``/command`` to the bot.  Leading ``/`` is added if missing."""
        if not command.startswith("/"):
            command = f"/{command}"
        return await self.send(command)
