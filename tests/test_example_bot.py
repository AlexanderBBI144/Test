"""Example tests showcasing every capability of the bot_test framework.

Replace ``/start``, ``/help``, button labels etc. with real commands
supported by the bot you're testing.
"""

from __future__ import annotations

import asyncio

import pytest

from bot_test import (
    TelegramTestClient,
    send_command,
    click_inline_button_await_edit,
    click_inline_button_await_alert,
    get_inline_buttons_flat,
    find_inline_button,
    get_text,
    get_reply_buttons_flat,
)
from bot_test.waiters import wait_for_message, wait_for_edit


# ---------------------------------------------------------------------------
# /start — basic command → response
# ---------------------------------------------------------------------------

class TestStartCommand:

    async def test_start_returns_message(self, tg_client: TelegramTestClient):
        response = await send_command(tg_client, "/start")
        assert response is not None
        assert len(get_text(response)) > 0

    async def test_start_has_inline_keyboard(self, tg_client: TelegramTestClient):
        response = await send_command(tg_client, "/start")
        buttons = get_inline_buttons_flat(response)
        # Uncomment & adapt once you know the expected buttons:
        # assert len(buttons) >= 1
        # assert buttons[0].text == "Open menu"


# ---------------------------------------------------------------------------
# Inline button → message edit
# ---------------------------------------------------------------------------

class TestInlineButtonEdit:

    async def test_button_click_edits_message(self, tg_client: TelegramTestClient):
        """Click an inline button and assert the message gets edited."""
        original = await send_command(tg_client, "/start")
        buttons = get_inline_buttons_flat(original)
        if not buttons:
            pytest.skip("Bot did not return inline buttons for /start")

        edited = await click_inline_button_await_edit(
            tg_client,
            original,
            index=0,
        )
        assert edited.id == original.id
        # The text should have changed:
        # assert get_text(edited) != get_text(original)


# ---------------------------------------------------------------------------
# Inline button → callback alert / toast
# ---------------------------------------------------------------------------

class TestInlineButtonAlert:

    async def test_button_shows_alert(self, tg_client: TelegramTestClient):
        """Click a button that triggers an alert / toast popup."""
        original = await send_command(tg_client, "/start")
        buttons = get_inline_buttons_flat(original)
        if not buttons:
            pytest.skip("Bot did not return inline buttons for /start")

        alert_text = await click_inline_button_await_alert(
            tg_client,
            original,
            index=0,
        )
        # alert_text is None when the bot answers without show_alert.
        # Adapt the assertion to your bot's behavior:
        # assert alert_text is not None
        # assert "Success" in alert_text


# ---------------------------------------------------------------------------
# Reply keyboard
# ---------------------------------------------------------------------------

class TestReplyKeyboard:

    async def test_reply_keyboard_buttons(self, tg_client: TelegramTestClient):
        """Send a command that returns a reply keyboard and press a button."""
        response = await send_command(tg_client, "/menu")
        reply_buttons = get_reply_buttons_flat(response)
        if not reply_buttons:
            pytest.skip("Bot did not return a reply keyboard for /menu")

        # "Press" a reply-keyboard button by sending its label as text.
        await tg_client.send(reply_buttons[0].text)
        reply = await wait_for_message(
            tg_client.raw,
            tg_client.bot,
            timeout=tg_client.cfg.timeout,
        )
        assert reply is not None


# ---------------------------------------------------------------------------
# Edited messages (bot edits a message after some time)
# ---------------------------------------------------------------------------

class TestMessageEdit:

    async def test_wait_for_delayed_edit(self, tg_client: TelegramTestClient):
        """Some bots send a placeholder and then edit with the real content."""
        response = await send_command(tg_client, "/slow")
        edited = await wait_for_edit(
            tg_client.raw,
            tg_client.bot,
            message_id=response.id,
            timeout=tg_client.cfg.timeout,
        )
        assert get_text(edited) != get_text(response)


# ---------------------------------------------------------------------------
# Custom filter on wait_for_message
# ---------------------------------------------------------------------------

class TestCustomFilter:

    async def test_filter_by_content(self, tg_client: TelegramTestClient):
        """Use a filter function to wait for a message containing a keyword."""
        await tg_client.send_command("/help")
        msg = await wait_for_message(
            tg_client.raw,
            tg_client.bot,
            filter_fn=lambda m: "help" in (m.message or "").lower(),
            timeout=tg_client.cfg.timeout,
        )
        assert "help" in get_text(msg).lower()


# ---------------------------------------------------------------------------
# Parallel waiter pattern (wait for edit while interacting)
# ---------------------------------------------------------------------------

class TestParallelWaiter:

    async def test_send_then_listen_for_edit(self, tg_client: TelegramTestClient):
        """Demonstrate starting a waiter *before* triggering the action."""
        response = await send_command(tg_client, "/counter")
        buttons = get_inline_buttons_flat(response)
        if not buttons:
            pytest.skip("No buttons on /counter")

        # Start listening before clicking.
        edit_task = asyncio.create_task(
            wait_for_edit(
                tg_client.raw,
                tg_client.bot,
                message_id=response.id,
                timeout=tg_client.cfg.timeout,
            )
        )
        await tg_client.raw.send_read_acknowledge(tg_client.bot)
        await response.click(0)
        edited = await edit_task
        assert edited.id == response.id
