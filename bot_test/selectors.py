"""Selectors — extract buttons, text, and other data from messages."""

from __future__ import annotations

from telethon.tl.types import (
    Message,
    ReplyInlineMarkup,
    ReplyKeyboardMarkup,
    KeyboardButtonRow,
    KeyboardButton,
    KeyboardButtonCallback,
    KeyboardButtonUrl,
)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def get_text(message: Message) -> str:
    """Return the plain-text content of a message (empty string if absent)."""
    return message.message or ""


# ---------------------------------------------------------------------------
# Inline keyboard
# ---------------------------------------------------------------------------

def get_inline_rows(message: Message) -> list[KeyboardButtonRow]:
    """Return rows of the inline keyboard, or ``[]`` if there is none."""
    markup = message.reply_markup
    if not isinstance(markup, ReplyInlineMarkup):
        return []
    return list(markup.rows)


def get_inline_buttons_flat(
    message: Message,
) -> list[KeyboardButtonCallback | KeyboardButtonUrl | KeyboardButton]:
    """Return all inline buttons in a flat list (row order preserved)."""
    buttons: list = []
    for row in get_inline_rows(message):
        buttons.extend(row.buttons)
    return buttons


def find_inline_button(
    message: Message,
    text: str,
) -> KeyboardButtonCallback | KeyboardButtonUrl | KeyboardButton | None:
    """Find the first inline button whose label matches *text* (exact)."""
    for btn in get_inline_buttons_flat(message):
        if btn.text == text:
            return btn
    return None


# ---------------------------------------------------------------------------
# Reply keyboard
# ---------------------------------------------------------------------------

def get_reply_keyboard_rows(message: Message) -> list[KeyboardButtonRow]:
    """Return rows of the reply keyboard, or ``[]`` if there is none."""
    markup = message.reply_markup
    if not isinstance(markup, ReplyKeyboardMarkup):
        return []
    return list(markup.rows)


def get_reply_buttons_flat(message: Message) -> list[KeyboardButton]:
    """Return all reply-keyboard buttons in a flat list."""
    buttons: list[KeyboardButton] = []
    for row in get_reply_keyboard_rows(message):
        buttons.extend(row.buttons)
    return buttons


def find_reply_button(message: Message, text: str) -> KeyboardButton | None:
    """Find the first reply-keyboard button whose label matches *text*."""
    for btn in get_reply_buttons_flat(message):
        if btn.text == text:
            return btn
    return None
