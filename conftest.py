"""Root conftest — session-scoped Telethon client fixture.

The client logs in **once** per test session and is shared across all
tests (Telegram auth is expensive; reuse is intentional).
"""

from __future__ import annotations

import logging

import pytest

from tg.client import TelegramTestClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@pytest.fixture(scope="session")
async def tc() -> TelegramTestClient:
    """Session-wide :class:`TelegramTestClient` (started & stopped
    automatically)."""
    client = TelegramTestClient()
    await client.start()
    yield client  # type: ignore[misc]
    await client.stop()
