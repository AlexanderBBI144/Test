"""Async waiters — await specific Telegram events with timeout."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from telethon import TelegramClient, events
from telethon.tl.types import Message, User

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# wait_for_message
# ---------------------------------------------------------------------------

async def wait_for_message(
    client: TelegramClient,
    from_user: User | int,
    *,
    chat: User | int | None = None,
    filter_fn: Callable[[Message], bool] | None = None,
    timeout: float = 10,
) -> Message:
    """Wait until a **new** message from *from_user* arrives.

    Parameters
    ----------
    client:
        The running :class:`TelegramClient`.
    from_user:
        The user/bot whose message we're waiting for (entity or id).
    chat:
        Restrict to a specific chat.  Defaults to ``from_user`` (DM with bot).
    filter_fn:
        Optional predicate ``(Message) -> bool`` for additional filtering.
    timeout:
        Seconds to wait before raising :class:`asyncio.TimeoutError`.
    """
    sender_id = from_user if isinstance(from_user, int) else from_user.id
    chat_id = chat if isinstance(chat, int) else (chat.id if chat else sender_id)
    future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def _handler(event: events.NewMessage.Event) -> None:
        msg: Message = event.message
        if msg.sender_id != sender_id:
            return
        if filter_fn and not filter_fn(msg):
            return
        if not future.done():
            future.set_result(msg)

    handler = client.on(events.NewMessage(chats=chat_id))(_handler)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    finally:
        client.remove_event_handler(handler, events.NewMessage)


# ---------------------------------------------------------------------------
# wait_for_edit
# ---------------------------------------------------------------------------

async def wait_for_edit(
    client: TelegramClient,
    from_user: User | int,
    *,
    message_id: int | None = None,
    chat: User | int | None = None,
    filter_fn: Callable[[Message], bool] | None = None,
    timeout: float = 10,
) -> Message:
    """Wait until a message from *from_user* is **edited**.

    If *message_id* is provided, only that specific message's edit is awaited.
    """
    sender_id = from_user if isinstance(from_user, int) else from_user.id
    chat_id = chat if isinstance(chat, int) else (chat.id if chat else sender_id)
    future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def _handler(event: events.MessageEdited.Event) -> None:
        msg: Message = event.message
        if msg.sender_id != sender_id:
            return
        if message_id is not None and msg.id != message_id:
            return
        if filter_fn and not filter_fn(msg):
            return
        if not future.done():
            future.set_result(msg)

    handler = client.on(events.MessageEdited(chats=chat_id))(_handler)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    finally:
        client.remove_event_handler(handler, events.MessageEdited)


# ---------------------------------------------------------------------------
# wait_for_callback_answer
# ---------------------------------------------------------------------------

async def wait_for_callback_answer(
    client: TelegramClient,
    click_coro,
    *,
    timeout: float = 10,
) -> str | None:
    """Click a button (via *click_coro*) and capture the ``answer`` / alert.

    Telethon's ``button.click()`` returns the raw callback-query answer.
    This helper wraps it with a timeout.

    Returns the answer text or ``None`` if the bot didn't respond.
    """
    result = await asyncio.wait_for(click_coro, timeout=timeout)
    if result is None:
        return None
    # result is a BotCallbackAnswer or similar
    return result.message if hasattr(result, "message") else str(result)
