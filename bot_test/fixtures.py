"""Reusable pytest fixtures for Telegram bot testing.

Import this module's fixtures into your ``conftest.py``::

    from bot_test.fixtures import *          # noqa: F401,F403
    # or cherry-pick:
    from bot_test.fixtures import tg_config, tg_client, bot
"""

from __future__ import annotations

import pytest

from bot_test.config import Config
from bot_test.client import TelegramTestClient


# ---------------------------------------------------------------------------
# Session-scoped: one Telegram connection per test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tg_config() -> Config:
    """Load Telegram test configuration from ``.env``."""
    return Config.from_env()


@pytest.fixture(scope="session")
async def tg_client(tg_config: Config):
    """Session-scoped :class:`TelegramTestClient` — connects once, reuses."""
    tc = TelegramTestClient(tg_config)
    await tc.start()
    yield tc
    await tc.stop()


@pytest.fixture(scope="session")
def bot(tg_client: TelegramTestClient):
    """Shortcut for the bot entity (``User``)."""
    return tg_client.bot


# ---------------------------------------------------------------------------
# Function-scoped helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def timeout(tg_config: Config) -> float:
    """Default per-test timeout pulled from config."""
    return tg_config.timeout
