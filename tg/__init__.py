"""Telethon + pytest toolkit for Telegram bot testing."""

from tg.asserts import (
    assert_alert,
    assert_has_buttons,
    assert_has_media,
    assert_has_photo,
    assert_has_reply_keyboard,
    assert_text,
    assert_toast,
)
from tg.client import TelegramTestClient
from tg.config import cfg
from tg.interact import BotConversation
from tg.retry import with_flood_retry
from tg.waiters import EventTimeout

__all__ = [
    # Core
    "TelegramTestClient",
    "BotConversation",
    "EventTimeout",
    "cfg",
    # Retry
    "with_flood_retry",
    # Assertion helpers
    "assert_alert",
    "assert_has_buttons",
    "assert_has_media",
    "assert_has_photo",
    "assert_has_reply_keyboard",
    "assert_text",
    "assert_toast",
]
