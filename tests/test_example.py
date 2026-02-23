"""Example tests demonstrating every major framework capability.

These are **templates** — adapt bot usernames, commands, and expected
texts to your actual bot under test.
"""

from __future__ import annotations

import pytest

from tg.interact import BotConversation
from tg.waiters import EventTimeout


# ==============================================================
# 1. Basic command → reply
# ==============================================================

class TestBasicCommands:
    async def test_start_returns_welcome(self, conv: BotConversation):
        reply = await conv.send_and_wait("/start")
        assert reply.text  # bot replied with *something*

    async def test_help_returns_text(self, conv: BotConversation):
        reply = await conv.send_and_wait("/help")
        assert len(reply.text) > 0


# ==============================================================
# 2. Inline keyboard — reading buttons
# ==============================================================

class TestInlineKeyboard:
    async def test_message_has_buttons(self, conv: BotConversation):
        reply = await conv.send_and_wait("/menu")
        buttons = conv.get_buttons(reply)
        assert len(buttons) > 0, "Expected inline keyboard"

    async def test_button_labels_readable(self, conv: BotConversation):
        reply = await conv.send_and_wait("/menu")
        labels = conv.get_button_labels(reply)
        # labels is [[str, ...], ...] — flatten and check non-empty
        flat = [lbl for row in labels for lbl in row]
        assert all(isinstance(lbl, str) and lbl for lbl in flat)


# ==============================================================
# 3. Clicking a button → callback answer (toast / alert)
# ==============================================================

class TestButtonClick:
    async def test_click_by_label(self, conv: BotConversation):
        reply = await conv.send_and_wait("/menu")
        answer = await conv.click(reply, label="Info")
        # answer.message is toast text; answer.alert indicates a popup
        assert answer.message or answer.alert

    async def test_click_by_position(self, conv: BotConversation):
        reply = await conv.send_and_wait("/menu")
        answer = await conv.click(reply, pos=(0, 0))
        assert answer.message or answer.alert


# ==============================================================
# 4. Click → bot edits the message
# ==============================================================

class TestClickAndEdit:
    async def test_click_triggers_edit(self, conv: BotConversation):
        reply = await conv.send_and_wait("/menu")
        answer, edited = await conv.click_and_wait_edit(reply, pos=(0, 0))
        assert edited.text != reply.text  # content changed


# ==============================================================
# 5. Click → bot sends a NEW message
# ==============================================================

class TestClickAndReply:
    async def test_click_triggers_new_message(self, conv: BotConversation):
        reply = await conv.send_and_wait("/start")
        _, new_msg = await conv.click_and_wait_reply(reply, pos=(0, 0))
        assert new_msg.text


# ==============================================================
# 6. Waiting for an edited message standalone
# ==============================================================

class TestWaitEdit:
    async def test_wait_edit_on_known_message(self, conv: BotConversation):
        reply = await conv.send_and_wait("/counter")
        # suppose the bot auto-edits the counter every second
        edited = await conv.wait_edit(reply.id, timeout=15)
        assert edited.id == reply.id


# ==============================================================
# 7. Timeout handling
# ==============================================================

class TestTimeouts:
    async def test_no_reply_raises_timeout(self, conv: BotConversation):
        await conv.send("/nonexistent_command_xyz")
        with pytest.raises(EventTimeout):
            await conv.wait_reply(timeout=3)


# ==============================================================
# 8. Reading recent chat history
# ==============================================================

class TestHistory:
    async def test_fetch_last_messages(self, conv: BotConversation):
        await conv.send_and_wait("/start")
        history = await conv.get_last_messages(limit=3)
        assert len(history) >= 1


# ==============================================================
# 9. Custom filter on wait
# ==============================================================

class TestCustomFilter:
    async def test_wait_reply_with_filter(self, conv: BotConversation):
        reply = await conv.send_and_wait(
            "/start",
            filter=lambda ev: "Welcome" in (ev.message.text or ""),
        )
        assert "Welcome" in reply.text
