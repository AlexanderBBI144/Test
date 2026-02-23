"""High-level bot interaction helpers.

:class:`BotConversation` is the main entry-point used in tests — it wraps
a Telethon client + target bot and provides a concise API for the most
common test operations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from telethon import events, types

from tg.client import TelegramTestClient
from tg.config import cfg
from tg.retry import with_flood_retry
from tg.waiters import (
    EventTimeout,
    find_button,
    find_reply_button,
    read_buttons,
    read_reply_keyboard,
    wait_callback_query_answer,
    wait_message_deleted,
    wait_message_edited,
    wait_messages,
    wait_new_message,
    wait_no_event,
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

    async def send_and_wait_multiple(
        self,
        text: str,
        count: int,
        *,
        timeout: float | None = None,
        filter: Callable[[events.NewMessage.Event], bool] | None = None,
    ) -> list[types.Message]:
        """Send *text* and wait for *count* reply messages.

        Useful when a command triggers multiple bot responses (e.g. a text
        message followed by a photo).
        """
        await self.send(text)
        return await wait_messages(
            self._tc.raw,
            from_user=self._bot_id,
            count=count,
            filter=filter,
            timeout=timeout,
        )

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

    async def wait_replies(
        self,
        count: int,
        *,
        timeout: float | None = None,
        filter: Callable[[events.NewMessage.Event], bool] | None = None,
    ) -> list[types.Message]:
        """Wait for *count* new messages from the bot."""
        return await wait_messages(
            self._tc.raw,
            from_user=self._bot_id,
            count=count,
            filter=filter,
            timeout=timeout,
        )

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

    async def wait_deletion(
        self,
        message_ids: int | list[int],
        *,
        timeout: float | None = None,
    ) -> None:
        """Wait for specific message(s) to be deleted."""
        await wait_message_deleted(
            self._tc.raw,
            message_ids=message_ids,
            timeout=timeout,
        )

    async def assert_no_reply(
        self,
        *,
        timeout: float = 3.0,
    ) -> None:
        """Assert the bot does NOT send a new message within *timeout*.

        Raises ``AssertionError`` if the bot replies.
        """
        await wait_no_event(
            self._tc.raw,
            events.NewMessage,
            lambda ev: ev.sender_id == self._bot_id,
            timeout=timeout,
        )

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
        """Press an inline callback button on *message*.

        Identify the button by its *label* text, raw callback *data*, or
        grid position *pos* ``(row, col)`` — exactly one must be given.

        Returns the :class:`BotCallbackAnswer` which may contain:
        - ``answer.message`` — toast notification text
        - ``answer.alert`` — alert popup flag (text is in ``message``)
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
        btn_data = getattr(btn, "data", None)
        if btn_data is None:
            raise TypeError(
                f"Button {btn.text!r} is not a callback button (no data). "
                f"URL and switch-inline buttons cannot be clicked via the API."
            )
        return await self._click_with_retry(message, btn_data, timeout)

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

    async def click_and_wait_alert(
        self,
        message: types.Message,
        *,
        label: str | None = None,
        data: bytes | None = None,
        pos: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Click a button, assert the answer is an **alert** popup.

        Returns the alert text.
        """
        answer = await self.click(
            message, label=label, data=data, pos=pos, timeout=timeout
        )
        assert answer.alert, "Expected alert popup, got toast notification"
        return answer.message or ""

    async def click_and_wait_toast(
        self,
        message: types.Message,
        *,
        label: str | None = None,
        data: bytes | None = None,
        pos: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Click a button, assert the answer is a **toast** notification.

        Returns the toast text.
        """
        answer = await self.click(
            message, label=label, data=data, pos=pos, timeout=timeout
        )
        assert not answer.alert, "Expected toast notification, got alert popup"
        return answer.message or ""

    # ------------------------------------------------------------------
    # Reply keyboard interaction
    # ------------------------------------------------------------------

    async def send_reply_button(
        self,
        message: types.Message,
        label: str,
        **kw: Any,
    ) -> types.Message:
        """Send text matching a reply keyboard button label.

        Raises ``LookupError`` if the button is not found.
        """
        btn = find_reply_button(message, label=label)
        if btn is None:
            available = [
                b.text for row in read_reply_keyboard(message) for b in row
            ]
            raise LookupError(
                f"Reply keyboard button {label!r} not found. "
                f"Available: {available}"
            )
        return await self.send(btn.text, **kw)

    async def send_reply_button_and_wait(
        self,
        message: types.Message,
        label: str,
        *,
        timeout: float | None = None,
        filter: Callable[[events.NewMessage.Event], bool] | None = None,
    ) -> types.Message:
        """Press a reply keyboard button and wait for the bot's response."""
        await self.send_reply_button(message, label)
        ev = await wait_new_message(
            self._tc.raw,
            from_user=self._bot_id,
            filter=filter,
            timeout=timeout,
        )
        return ev.message

    # ------------------------------------------------------------------
    # File / media helpers
    # ------------------------------------------------------------------

    async def send_file(self, file: Any, **kw: Any) -> types.Message:
        """Send a file/photo to the bot (with automatic flood-wait retry)."""
        return await self._tc.send_file(self._bot, file, **kw)

    async def send_file_and_wait(
        self,
        file: Any,
        *,
        timeout: float | None = None,
        filter: Callable[[events.NewMessage.Event], bool] | None = None,
        **kw: Any,
    ) -> types.Message:
        """Send a file/photo and wait for the bot's reply."""
        await self.send_file(file, **kw)
        ev = await wait_new_message(
            self._tc.raw,
            from_user=self._bot_id,
            filter=filter,
            timeout=timeout,
        )
        return ev.message

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
    def get_buttons(message: types.Message) -> list[list[Any]]:
        """Shortcut to :func:`waiters.read_buttons`."""
        return read_buttons(message)

    @staticmethod
    def get_button_labels(message: types.Message) -> list[list[str]]:
        """Return inline button labels as a 2D list of strings."""
        return [[b.text for b in row] for row in read_buttons(message)]

    @staticmethod
    def get_button_urls(message: types.Message) -> list[tuple[str, str]]:
        """Return ``(label, url)`` pairs for all URL buttons."""
        result: list[tuple[str, str]] = []
        for row in read_buttons(message):
            for btn in row:
                url = getattr(btn, "url", None)
                if url:
                    result.append((btn.text, url))
        return result

    @staticmethod
    def get_reply_keyboard(
        message: types.Message,
    ) -> list[list[Any]]:
        """Shortcut to :func:`waiters.read_reply_keyboard`."""
        return read_reply_keyboard(message)

    @staticmethod
    def get_reply_keyboard_labels(message: types.Message) -> list[list[str]]:
        """Return reply keyboard labels as a 2D list of strings."""
        return [[b.text for b in row] for row in read_reply_keyboard(message)]

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
        return await with_flood_retry(
            lambda: wait_callback_query_answer(
                self._tc.raw,
                peer=self._bot,
                message_id=message.id,
                data=data,
                timeout=timeout,
            ),
            retries=3,
            label="click",
        )
