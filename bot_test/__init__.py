"""Telethon + pytest toolkit for Telegram bot integration testing."""

from bot_test.client import TelegramTestClient
from bot_test.config import Config
from bot_test.waiters import wait_for_message, wait_for_edit, wait_for_callback_answer
from bot_test.actions import (
    send_command,
    click_inline_button,
    click_inline_button_await_edit,
    click_inline_button_await_alert,
)
from bot_test.selectors import (
    get_inline_rows,
    get_inline_buttons_flat,
    find_inline_button,
    get_reply_keyboard_rows,
    get_text,
)
from bot_test.rate_limit import rate_limited

__all__ = [
    "TelegramTestClient",
    "Config",
    # waiters
    "wait_for_message",
    "wait_for_edit",
    "wait_for_callback_answer",
    # actions
    "send_command",
    "click_inline_button",
    "click_inline_button_await_edit",
    "click_inline_button_await_alert",
    # selectors
    "get_inline_rows",
    "get_inline_buttons_flat",
    "find_inline_button",
    "get_reply_keyboard_rows",
    "get_text",
    # rate limit
    "rate_limited",
]
