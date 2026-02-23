"""Telethon + pytest toolkit for Telegram bot testing."""

from tg.client import TelegramTestClient
from tg.config import cfg
from tg.interact import BotConversation
from tg.waiters import EventTimeout

__all__ = [
    "TelegramTestClient",
    "BotConversation",
    "EventTimeout",
    "cfg",
]
