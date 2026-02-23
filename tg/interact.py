"""High-level bot interaction helpers.

:class:`BotConversation` is the main entry-point used in tests — it wraps
a Telethon client + target bot and provides a concise API for the most
common test operations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from telethon import errors, events, types

from tg.client import TelegramTestClient
from tg.config import cfg
from tg.waiters import (
    EventTimeout,
    find_button,
    read_buttons,
    wait_callback_query_answer,
    wait_message_edited,
    wait_new_message,
)

log = logging.getLogger(__name__)


class BotConversation:
    """Stateful helper bound to a single bot dialog.

    Typical usage inside a test::

        async with bot_conv as conv:
            reply = await conv.send_and_wait("/start")
            assert "Welcome" in reply.text
    """

    def __init__(self, tc: TelegramTestClient, bot: str | int) -> None:
        self._tc = tc
        self._bot = bot
        self._bot_id: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle (usable as async context manager)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BotConversation:
        entity = await self._tc.raw.get_entity(self._bot)
        self._bot_id = entity.id
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass  # nothing to tear down per-conversation

    # ------------------------------------------------------------------
    # Core: send → wait for reply
    # ------------------------------------------------------------------

    async def send(self, text: str, **kw: Any) -> types.Message:
        """Send *text* to the bot (with automatic flood-wait retry)."""
        return await self._tc.send(self._bot, text, **kw)

    async def send_and_wait(
        self,
        text: str,
        *,
        timeout: float | None = None,
        filter: Callable[[events.NewMessage.Event], bool] | None = None,
    ) -> types.Message:
        """Send *text* and wait for the bot's reply message."""
        await self.send(text)
        ev = await wait_new_message(
            self._tc.raw,
            from_user=self._bot_id,
            filter=filter,
            timeout=timeout,
        )
        return ev.message

    # ------------------------------------------------------------------
    # Wait helpers (delegate to waiters module)
    # ------------------------------------------------------------------

    async def wait_reply(
        self,
        *,
        timeout: float | None = None,
        filter: Callable[[events.NewMessage.Event], bool] | None = None,
    ) -> types.Message:
        """Wait for the next new message from the bot."""
        ev = await wait_new_message(
            self._tc.raw,
            from_user=self._bot_id,
            filter=filter,
            timeout=timeout,
        )
        return ev.message

    async def wait_edit(
        self,
        message_id: int | None = None,
        *,
        timeout: float | None = None,
        filter: Callable[[events.MessageEdited.Event], bool] | None = None,
    ) -> types.Message:
        """Wait for the bot to *edit* a message (optionally a specific one)."""
        ev = await wait_message_edited(
            self._tc.raw,
            from_user=self._bot_id,
            message_id=message_id,
            filter=filter,
            timeout=timeout,
        )
        return ev.message

    # ------------------------------------------------------------------
    # Inline keyboard interaction
    # ------------------------------------------------------------------

    async def click(
        self,
        message: types.Message,
        *,
        label: str | None = None,
        data: bytes | None = None,
        pos: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> types.messages.BotCallbackAnswer:
        """Press an inline button on *message*.

        Identify the button by its *label* text, raw callback *data*, or
        grid position *pos* ``(row, col)`` — exactly one must be given.

        Returns the :class:`BotCallbackAnswer` which may contain:
        - ``answer.message`` — toast notification text
        - ``answer.alert`` — alert popup text
        """
        btn = find_button(message, label=label, data=data, pos=pos)
        if btn is None:
            available = [
                b.text for row in read_buttons(message) for b in row
            ]
            raise LookupError(
                f"Button not found (label={label!r}, data={data!r}, pos={pos!r}). "
                f"Available buttons: {available}"
            )
        return await self._click_with_retry(message, btn.data, timeout)

    async def click_and_wait_edit(
        self,
        message: types.Message,
        *,
        label: str | None = None,
        data: bytes | None = None,
        pos: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> tuple[types.messages.BotCallbackAnswer, types.Message]:
        """Press an inline button and concurrently wait for the message to be
        edited.  Returns ``(callback_answer, edited_message)``."""
        timeout = timeout or cfg.default_timeout

        async def _do_click() -> types.messages.BotCallbackAnswer:
            return await self.click(
                message, label=label, data=data, pos=pos, timeout=timeout
            )

        async def _do_wait() -> types.Message:
            return await self.wait_edit(message.id, timeout=timeout)

        answer, edited = await asyncio.gather(_do_click(), _do_wait())
        return answer, edited

    async def click_and_wait_reply(
        self,
        message: types.Message,
        *,
        label: str | None = None,
        data: bytes | None = None,
        pos: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> tuple[types.messages.BotCallbackAnswer, types.Message]:
        """Press an inline button and concurrently wait for a **new** message
        from the bot.  Returns ``(callback_answer, new_message)``."""
        timeout = timeout or cfg.default_timeout

        async def _do_click() -> types.messages.BotCallbackAnswer:
            return await self.click(
                message, label=label, data=data, pos=pos, timeout=timeout
            )

        async def _do_wait() -> types.Message:
            return await self.wait_reply(timeout=timeout)

        answer, reply = await asyncio.gather(_do_click(), _do_wait())
        return answer, reply

    # ------------------------------------------------------------------
    # Reading helpers
    # ------------------------------------------------------------------

    async def get_last_messages(self, limit: int = 5) -> list[types.Message]:
        """Fetch the last *limit* messages from the bot dialog."""
        return [
            m
            async for m in self._tc.raw.iter_messages(self._bot, limit=limit)
        ]

    @staticmethod
    def get_buttons(message: types.Message) -> list[list[types.KeyboardButtonCallback]]:
        """Shortcut to :func:`waiters.read_buttons`."""
        return read_buttons(message)

    @staticmethod
    def get_button_labels(message: types.Message) -> list[list[str]]:
        """Return inline button labels as a 2D list of strings."""
        return [[b.text for b in row] for row in read_buttons(message)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _click_with_retry(
        self,
        message: types.Message,
        data: bytes,
        timeout: float | None,
    ) -> types.messages.BotCallbackAnswer:
        """Click with automatic FloodWaitError retry."""
        for attempt in range(1, 4):
            try:
                return await wait_callback_query_answer(
                    self._tc.raw,
                    peer=self._bot,
                    message_id=message.id,
                    data=data,
                    timeout=timeout,
                )
            except errors.FloodWaitError as exc:
                wait = max(exc.seconds, cfg.rate_limit_pause)
                log.warning("FloodWait on click (%ds, attempt %d)", wait, attempt)
                await asyncio.sleep(wait)
        raise errors.FloodWaitError(request=None, capture=0)
