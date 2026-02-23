"""Test-scoped fixtures: per-test bot conversation."""

from __future__ import annotations

import pytest

from tg.client import TelegramTestClient
from tg.config import cfg
from tg.interact import BotConversation


@pytest.fixture
async def conv(tc: TelegramTestClient) -> BotConversation:
    """A fresh :class:`BotConversation` opened against the configured bot.

    Usage in tests::

        async def test_start(conv: BotConversation):
            reply = await conv.send_and_wait("/start")
            assert "Welcome" in reply.text
    """
    async with BotConversation(tc, cfg.bot_username) as c:
        yield c  # type: ignore[misc]
