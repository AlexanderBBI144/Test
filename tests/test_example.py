"""Example tests demonstrating every major framework capability.

These are **templates** — adapt bot usernames, commands, and expected
texts to your actual bot under test.
"""

from __future__ import annotations

import pytest

from tg.asserts import (
    assert_alert,
    assert_has_buttons,
    assert_has_media,
    assert_has_reply_keyboard,
    assert_text,
    assert_toast,
)
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

    async def test_url_buttons_readable(self, conv: BotConversation):
        """Demonstrate reading URL buttons from an inline keyboard."""
        reply = await conv.send_and_wait("/links")
        urls = conv.get_button_urls(reply)
        # urls is [(label, url), ...]
        for label, url in urls:
            assert label
            assert url.startswith("http")


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

    async def test_click_returns_toast(self, conv: BotConversation):
        """Use click_and_wait_toast to assert the answer is a toast."""
        reply = await conv.send_and_wait("/menu")
        text = await conv.click_and_wait_toast(reply, pos=(0, 0))
        assert isinstance(text, str)

    async def test_click_returns_alert(self, conv: BotConversation):
        """Use click_and_wait_alert to assert the answer is an alert popup."""
        reply = await conv.send_and_wait("/confirm")
        text = await conv.click_and_wait_alert(reply, label="Delete")
        assert "sure" in text.lower() or text  # depends on bot


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
# 7. Waiting for message deletion
# ==============================================================

class TestWaitDeletion:
    async def test_bot_deletes_message(self, conv: BotConversation):
        """Wait for the bot to delete a specific message."""
        reply = await conv.send_and_wait("/temp")
        # bot should auto-delete this message shortly
        await conv.wait_deletion(reply.id, timeout=15)


# ==============================================================
# 8. Multiple reply messages
# ==============================================================

class TestMultipleReplies:
    async def test_command_triggers_multiple_messages(self, conv: BotConversation):
        """Some bots reply with several messages at once."""
        messages = await conv.send_and_wait_multiple("/gallery", count=3)
        assert len(messages) == 3

    async def test_wait_replies_standalone(self, conv: BotConversation):
        """Send first, then wait for N replies separately."""
        await conv.send("/gallery")
        messages = await conv.wait_replies(count=2, timeout=15)
        assert len(messages) == 2


# ==============================================================
# 9. Reply keyboard interaction
# ==============================================================

class TestReplyKeyboard:
    async def test_message_has_reply_keyboard(self, conv: BotConversation):
        reply = await conv.send_and_wait("/settings")
        keyboard = conv.get_reply_keyboard(reply)
        assert len(keyboard) > 0, "Expected reply keyboard"

    async def test_reply_keyboard_labels(self, conv: BotConversation):
        reply = await conv.send_and_wait("/settings")
        labels = conv.get_reply_keyboard_labels(reply)
        flat = [lbl for row in labels for lbl in row]
        assert all(isinstance(lbl, str) and lbl for lbl in flat)

    async def test_press_reply_button(self, conv: BotConversation):
        """Press a reply keyboard button and wait for the bot's response."""
        reply = await conv.send_and_wait("/settings")
        response = await conv.send_reply_button_and_wait(reply, "Language")
        assert response.text


# ==============================================================
# 10. File / media interaction
# ==============================================================

class TestMedia:
    async def test_send_photo_and_wait(self, conv: BotConversation):
        """Send a photo to the bot and wait for its reply."""
        reply = await conv.send_file_and_wait(
            "tests/fixtures/sample.jpg",
            caption="Identify this",
        )
        assert reply.text

    async def test_bot_reply_has_media(self, conv: BotConversation):
        """Check that the bot replied with media."""
        reply = await conv.send_and_wait("/photo")
        assert_has_media(reply)


# ==============================================================
# 11. Timeout handling
# ==============================================================

class TestTimeouts:
    async def test_no_reply_raises_timeout(self, conv: BotConversation):
        await conv.send("/nonexistent_command_xyz")
        with pytest.raises(EventTimeout):
            await conv.wait_reply(timeout=3)


# ==============================================================
# 12. Negative testing — assert NO reply
# ==============================================================

class TestNegative:
    async def test_unknown_command_no_reply(self, conv: BotConversation):
        """Assert the bot stays silent on unknown input."""
        await conv.send("/definitely_not_a_command_1234")
        await conv.assert_no_reply(timeout=3)


# ==============================================================
# 13. Reading recent chat history
# ==============================================================

class TestHistory:
    async def test_fetch_last_messages(self, conv: BotConversation):
        await conv.send_and_wait("/start")
        history = await conv.get_last_messages(limit=3)
        assert len(history) >= 1


# ==============================================================
# 14. Custom filter on wait
# ==============================================================

class TestCustomFilter:
    async def test_wait_reply_with_filter(self, conv: BotConversation):
        reply = await conv.send_and_wait(
            "/start",
            filter=lambda ev: "Welcome" in (ev.message.text or ""),
        )
        assert "Welcome" in reply.text

    async def test_wait_edit_with_filter(self, conv: BotConversation):
        """Wait for an edit that matches a custom predicate."""
        reply = await conv.send_and_wait("/counter")
        edited = await conv.wait_edit(
            reply.id,
            filter=lambda ev: "2" in (ev.message.text or ""),
            timeout=15,
        )
        assert "2" in edited.text


# ==============================================================
# 15. Assertion helpers (tg.asserts)
# ==============================================================

class TestAssertionHelpers:
    async def test_assert_text(self, conv: BotConversation):
        reply = await conv.send_and_wait("/start")
        assert_text(reply, contains="Welcome")

    async def test_assert_has_buttons(self, conv: BotConversation):
        reply = await conv.send_and_wait("/menu")
        assert_has_buttons(reply, min_count=2, labels=["Info", "Settings"])

    async def test_assert_has_reply_keyboard(self, conv: BotConversation):
        reply = await conv.send_and_wait("/settings")
        assert_has_reply_keyboard(reply, labels=["Language"])

    async def test_assert_callback_toast(self, conv: BotConversation):
        reply = await conv.send_and_wait("/menu")
        answer = await conv.click(reply, pos=(0, 0))
        assert_toast(answer, contains="Done")

    async def test_assert_callback_alert(self, conv: BotConversation):
        reply = await conv.send_and_wait("/confirm")
        answer = await conv.click(reply, label="Delete")
        assert_alert(answer, contains="Are you sure")


# ==============================================================
# 16. Deep link / start parameter
# ==============================================================

class TestDeepLink:
    async def test_start_with_parameter(self, conv: BotConversation):
        """Test /start with a deep-link payload."""
        reply = await conv.send_and_wait("/start ref_campaign_123")
        assert reply.text
