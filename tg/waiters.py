"""Async waiters — wait for specific Telegram events with a timeout.

Every public function returns the matched event/message or raises
:class:`EventTimeout`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from telethon import TelegramClient, events, types

from tg.config import cfg


class EventTimeout(Exception):
    """Raised when the expected Telegram event was not received in time."""


# ======================================================================
# Internal helpers
# ======================================================================

async def _wait_event(
    client: TelegramClient,
    event_cls: Any,
    check: Callable[[Any], bool],
    timeout: float | None,
) -> Any:
    """Register a one-shot event handler and await it with *timeout*."""
    timeout = timeout if timeout is not None else cfg.default_timeout
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    async def _handler(ev: Any) -> None:
        if not future.done() and check(ev):
            future.set_result(ev)

    client.add_event_handler(_handler, event_cls)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        raise EventTimeout(
            f"Timed out after {timeout}s waiting for {event_cls.__name__}"
        ) from None
    finally:
        client.remove_event_handler(_handler, event_cls)


# ======================================================================
# Public waiters — single events
# ======================================================================

async def wait_new_message(
    client: TelegramClient,
    from_user: int | str,
    chat: int | str | None = None,
    *,
    filter: Callable[[events.NewMessage.Event], bool] | None = None,
    timeout: float | None = None,
) -> events.NewMessage.Event:
    """Wait for a **new** incoming message from *from_user*.

    *chat* restricts to a specific dialog.  *filter* is an optional extra
    predicate on the event.
    """
    from_id = await _resolve_peer_id(client, from_user)
    chat_id = await _resolve_peer_id(client, chat) if chat else None

    def _check(ev: events.NewMessage.Event) -> bool:
        if ev.sender_id != from_id:
            return False
        if chat_id and ev.chat_id != chat_id:
            return False
        return filter(ev) if filter else True

    return await _wait_event(client, events.NewMessage, _check, timeout)


async def wait_message_edited(
    client: TelegramClient,
    from_user: int | str,
    chat: int | str | None = None,
    *,
    message_id: int | None = None,
    filter: Callable[[events.MessageEdited.Event], bool] | None = None,
    timeout: float | None = None,
) -> events.MessageEdited.Event:
    """Wait for a message edit from *from_user*.

    Optionally restrict to a specific *message_id*.
    """
    from_id = await _resolve_peer_id(client, from_user)
    chat_id = await _resolve_peer_id(client, chat) if chat else None

    def _check(ev: events.MessageEdited.Event) -> bool:
        if ev.sender_id != from_id:
            return False
        if chat_id and ev.chat_id != chat_id:
            return False
        if message_id and ev.message.id != message_id:
            return False
        return filter(ev) if filter else True

    return await _wait_event(client, events.MessageEdited, _check, timeout)


async def wait_callback_query_answer(
    client: TelegramClient,
    peer: int | str,
    message_id: int,
    data: bytes,
    *,
    timeout: float | None = None,
) -> types.messages.BotCallbackAnswer:
    """Press an inline button identified by *data* and return the bot's
    :class:`BotCallbackAnswer` (may contain ``message`` or ``alert``).
    """
    from telethon import functions

    timeout = timeout if timeout is not None else cfg.default_timeout
    entity = await client.get_input_entity(peer)
    try:
        return await asyncio.wait_for(
            client(
                functions.messages.GetBotCallbackAnswerRequest(
                    peer=entity,
                    msg_id=message_id,
                    data=data,
                )
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise EventTimeout(
            f"Timed out after {timeout}s waiting for callback answer"
        ) from None


# ======================================================================
# Public waiters — multi-message
# ======================================================================

async def wait_messages(
    client: TelegramClient,
    from_user: int | str,
    count: int,
    *,
    filter: Callable[[events.NewMessage.Event], bool] | None = None,
    timeout: float | None = None,
) -> list[types.Message]:
    """Wait for *count* new messages from *from_user*.

    Useful when a single command triggers multiple reply messages (e.g. a
    bot that sends a text message followed by a photo).
    """
    timeout = timeout if timeout is not None else cfg.default_timeout
    from_id = await _resolve_peer_id(client, from_user)

    collected: list[types.Message] = []
    future: asyncio.Future[list[types.Message]] = (
        asyncio.get_running_loop().create_future()
    )

    async def _handler(ev: events.NewMessage.Event) -> None:
        if future.done():
            return
        if ev.sender_id != from_id:
            return
        if filter and not filter(ev):
            return
        collected.append(ev.message)
        if len(collected) >= count:
            future.set_result(list(collected))

    client.add_event_handler(_handler, events.NewMessage)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        raise EventTimeout(
            f"Timed out after {timeout}s waiting for {count} messages "
            f"(received {len(collected)})"
        ) from None
    finally:
        client.remove_event_handler(_handler, events.NewMessage)


# ======================================================================
# Public waiters — deletion
# ======================================================================

async def wait_message_deleted(
    client: TelegramClient,
    message_ids: int | list[int],
    *,
    timeout: float | None = None,
) -> events.MessageDeleted.Event:
    """Wait for specific message(s) to be deleted.

    *message_ids* can be a single id or a list.  Resolves as soon as
    **any** of the given ids appears in a ``MessageDeleted`` event.
    """
    timeout = timeout if timeout is not None else cfg.default_timeout
    ids = {message_ids} if isinstance(message_ids, int) else set(message_ids)

    future: asyncio.Future[events.MessageDeleted.Event] = (
        asyncio.get_running_loop().create_future()
    )

    async def _handler(ev: events.MessageDeleted.Event) -> None:
        if not future.done() and ids & set(ev.deleted_ids):
            future.set_result(ev)

    client.add_event_handler(_handler, events.MessageDeleted)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        raise EventTimeout(
            f"Timed out after {timeout}s waiting for deletion of message(s) {ids}"
        ) from None
    finally:
        client.remove_event_handler(_handler, events.MessageDeleted)


# ======================================================================
# Negative testing
# ======================================================================

async def wait_no_event(
    client: TelegramClient,
    event_cls: Any,
    check: Callable[[Any], bool],
    *,
    timeout: float = 3.0,
) -> None:
    """Assert that **no** matching event arrives within *timeout* seconds.

    Raises ``AssertionError`` if the event fires.  Useful for negative
    tests ("the bot should NOT reply to this unknown command").
    """
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    async def _handler(ev: Any) -> None:
        if not future.done() and check(ev):
            future.set_result(ev)

    client.add_event_handler(_handler, event_cls)
    try:
        event = await asyncio.wait_for(future, timeout=timeout)
        raise AssertionError(
            f"Expected no {event_cls.__name__} event, but received: {event}"
        )
    except asyncio.TimeoutError:
        pass  # Good — nothing arrived
    finally:
        client.remove_event_handler(_handler, event_cls)


# ======================================================================
# Inline keyboard reader
# ======================================================================

def read_buttons(message: types.Message) -> list[list[Any]]:
    """Extract inline keyboard rows from a message.

    Returns a 2D list ``rows × buttons``.  Each button can be any inline
    button type (``KeyboardButtonCallback``, ``KeyboardButtonUrl``,
    ``KeyboardButtonSwitchInline``, etc.).

    Returns ``[]`` if the message has no inline keyboard.
    """
    markup = message.reply_markup
    if not isinstance(markup, types.ReplyInlineMarkup):
        return []
    return [list(row.buttons) for row in markup.rows]


def find_button(
    message: types.Message,
    *,
    label: str | None = None,
    data: bytes | None = None,
    url: str | None = None,
    pos: tuple[int, int] | None = None,
) -> Any | None:
    """Find a single inline button by *label* text, callback *data*,
    *url*, or grid *pos* ``(row, col)`` (0-indexed).
    """
    rows = read_buttons(message)
    if not rows:
        return None
    if pos is not None:
        r, c = pos
        if r < len(rows) and c < len(rows[r]):
            return rows[r][c]
        return None
    for row in rows:
        for btn in row:
            if label is not None and btn.text == label:
                return btn
            if data is not None and getattr(btn, "data", None) == data:
                return btn
            if url is not None and getattr(btn, "url", None) == url:
                return btn
    return None


# ======================================================================
# Reply keyboard reader
# ======================================================================

def read_reply_keyboard(
    message: types.Message,
) -> list[list[Any]]:
    """Extract reply keyboard rows from a message.

    Returns a 2D list ``rows × buttons``.  Returns ``[]`` if the message
    has no reply keyboard markup.
    """
    markup = message.reply_markup
    if not isinstance(markup, types.ReplyKeyboardMarkup):
        return []
    return [list(row.buttons) for row in markup.rows]


def find_reply_button(
    message: types.Message,
    *,
    label: str,
) -> Any | None:
    """Find a reply keyboard button by its *label* text."""
    for row in read_reply_keyboard(message):
        for btn in row:
            if btn.text == label:
                return btn
    return None


# ======================================================================
# Peer ID resolution cache
# ======================================================================

_peer_cache: dict[str | int, int] = {}


async def _resolve_peer_id(client: TelegramClient, peer: int | str) -> int:
    if isinstance(peer, int):
        return peer
    if peer in _peer_cache:
        return _peer_cache[peer]
    entity = await client.get_entity(peer)
    _peer_cache[peer] = entity.id
    return entity.id
