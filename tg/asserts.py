"""Telegram-specific assertion helpers for concise, readable tests.

Usage::

    from tg.asserts import assert_text, assert_has_buttons, assert_alert

    reply = await conv.send_and_wait("/start")
    assert_text(reply, contains="Welcome")
    assert_has_buttons(reply, min_count=2)

Every helper raises a plain ``AssertionError`` with a descriptive message
so pytest diffs stay useful.
"""

from __future__ import annotations

from telethon import types

from tg.waiters import read_buttons, read_reply_keyboard


# ------------------------------------------------------------------
# Text assertions
# ------------------------------------------------------------------

def assert_text(
    message: types.Message,
    *,
    equals: str | None = None,
    contains: str | None = None,
    startswith: str | None = None,
    endswith: str | None = None,
) -> None:
    """Assert properties of ``message.text``.

    Multiple keyword arguments can be combined — all must pass.
    """
    text = message.text or ""
    if equals is not None:
        assert text == equals, f"Expected text {equals!r}, got {text!r}"
    if contains is not None:
        assert contains in text, f"Expected {contains!r} in {text!r}"
    if startswith is not None:
        assert text.startswith(startswith), (
            f"Expected text starting with {startswith!r}, got {text!r}"
        )
    if endswith is not None:
        assert text.endswith(endswith), (
            f"Expected text ending with {endswith!r}, got {text!r}"
        )


# ------------------------------------------------------------------
# Inline keyboard assertions
# ------------------------------------------------------------------

def assert_has_buttons(
    message: types.Message,
    *,
    min_count: int = 1,
    labels: list[str] | None = None,
) -> None:
    """Assert the message has an inline keyboard.

    *min_count* — minimum total number of buttons expected.
    *labels* — specific labels that must be present.
    """
    rows = read_buttons(message)
    flat = [b for row in rows for b in row]
    assert len(flat) >= min_count, (
        f"Expected >= {min_count} inline buttons, found {len(flat)}"
    )
    if labels:
        actual = [b.text for b in flat]
        for lbl in labels:
            assert lbl in actual, (
                f"Button {lbl!r} not found; available: {actual}"
            )


# ------------------------------------------------------------------
# Reply keyboard assertions
# ------------------------------------------------------------------

def assert_has_reply_keyboard(
    message: types.Message,
    *,
    min_count: int = 1,
    labels: list[str] | None = None,
) -> None:
    """Assert the message has a reply keyboard."""
    rows = read_reply_keyboard(message)
    flat = [b for row in rows for b in row]
    assert len(flat) >= min_count, (
        f"Expected >= {min_count} reply keyboard buttons, found {len(flat)}"
    )
    if labels:
        actual = [b.text for b in flat]
        for lbl in labels:
            assert lbl in actual, (
                f"Reply button {lbl!r} not found; available: {actual}"
            )


# ------------------------------------------------------------------
# Media assertions
# ------------------------------------------------------------------

def assert_has_media(message: types.Message) -> None:
    """Assert the message contains some media (photo, document, etc.)."""
    assert message.media is not None, "Expected message to have media"


def assert_has_photo(message: types.Message) -> None:
    """Assert the message contains a photo."""
    assert message.photo is not None, "Expected message to have a photo"


# ------------------------------------------------------------------
# Callback answer assertions
# ------------------------------------------------------------------

def assert_alert(
    answer: types.messages.BotCallbackAnswer,
    *,
    contains: str | None = None,
) -> None:
    """Assert the callback answer is an alert popup (not a toast)."""
    assert answer.alert, "Expected an alert popup, got a toast notification"
    if contains:
        text = answer.message or ""
        assert contains in text, f"Expected {contains!r} in alert {text!r}"


def assert_toast(
    answer: types.messages.BotCallbackAnswer,
    *,
    contains: str | None = None,
) -> None:
    """Assert the callback answer is a toast notification (not an alert)."""
    assert not answer.alert, "Expected a toast notification, got an alert popup"
    if contains:
        text = answer.message or ""
        assert contains in text, f"Expected {contains!r} in toast {text!r}"
