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
# Public waiters
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
# Inline keyboard reader
# ======================================================================

def read_buttons(message: types.Message) -> list[list[types.KeyboardButtonCallback]]:
    """Extract inline keyboard rows from a message.

    Returns a 2D list ``rows × buttons``.  Returns ``[]`` if the message
    has no inline keyboard.
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
    pos: tuple[int, int] | None = None,
) -> types.KeyboardButtonCallback | None:
    """Find a single inline button by *label* text, callback *data*, or
    grid *pos* ``(row, col)`` (0-indexed).
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
