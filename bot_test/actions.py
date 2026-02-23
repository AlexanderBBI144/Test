"""High-level bot interaction actions.

Every action combines sending / clicking with waiting for the bot's reaction,
so test code stays concise.
"""

from __future__ import annotations

import asyncio
import logging

from telethon.tl.types import Message

from bot_test.client import TelegramTestClient
from bot_test.rate_limit import retry_on_flood
from bot_test.waiters import wait_for_message, wait_for_edit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Send a command and wait for the bot's reply
# ---------------------------------------------------------------------------

async def send_command(
    tc: TelegramTestClient,
    command: str,
    *,
    timeout: float | None = None,
) -> Message:
    """Send ``/command`` to the bot and return the bot's first reply.

    This is the most common pattern in bot tests:
    ``response = await send_command(tc, "/start")``
    """
    timeout = timeout or tc.cfg.timeout
    await tc.send_command(command)
    return await wait_for_message(
        tc.raw,
        tc.bot,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Inline button clicks
# ---------------------------------------------------------------------------

async def click_inline_button(
    tc: TelegramTestClient,
    message: Message,
    *,
    text: str | None = None,
    index: int | None = None,
    row: int | None = None,
    col: int | None = None,
) -> bytes | None:
    """Click an inline button on *message*.

    Locate the button by **one** of:
    * ``text``  — exact label match,
    * ``index`` — flat 0-based index across all rows,
    * ``row`` + ``col`` — grid coordinates (both 0-based).

    Returns the raw callback result (``bytes``) or ``None``.
    """
    # message.click() is Telethon's built-in convenience method.
    kwargs: dict = {}
    if text is not None:
        kwargs["text"] = text
    elif index is not None:
        kwargs["i"] = index
    elif row is not None and col is not None:
        kwargs["i"] = row
        kwargs["j"] = col
    else:
        raise ValueError("Provide text=, index=, or (row=, col=)")

    return await retry_on_flood(
        message.click,
        max_retries=tc.cfg.rate_limit_retries,
        **kwargs,
    )


async def click_inline_button_await_edit(
    tc: TelegramTestClient,
    message: Message,
    *,
    text: str | None = None,
    index: int | None = None,
    row: int | None = None,
    col: int | None = None,
    timeout: float | None = None,
) -> Message:
    """Click an inline button and wait for the **same message** to be edited.

    Returns the updated :class:`Message`.
    """
    timeout = timeout or tc.cfg.timeout
    edit_future = asyncio.ensure_future(
        wait_for_edit(
            tc.raw,
            tc.bot,
            message_id=message.id,
            timeout=timeout,
        )
    )
    await click_inline_button(tc, message, text=text, index=index, row=row, col=col)
    return await edit_future


async def click_inline_button_await_alert(
    tc: TelegramTestClient,
    message: Message,
    *,
    text: str | None = None,
    index: int | None = None,
    row: int | None = None,
    col: int | None = None,
    timeout: float | None = None,
) -> str | None:
    """Click an inline button and return the callback-query answer/alert text.

    Returns ``None`` if the bot doesn't answer the callback query.
    """
    timeout = timeout or tc.cfg.timeout
    result = await asyncio.wait_for(
        click_inline_button(tc, message, text=text, index=index, row=row, col=col),
        timeout=timeout,
    )
    if result is None:
        return None
    # BotCallbackAnswer
    return result.message if hasattr(result, "message") else str(result)
