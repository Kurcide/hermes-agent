"""First-contact setup notices follow the receiving adapter's delivery capability."""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionStore


class RecordingAdapter(BasePlatformAdapter):
    def __init__(self, config):
        super().__init__(config, Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def get_chat_info(self, chat_id):
        return {"type": "dm"}


@pytest.mark.asyncio
@pytest.mark.parametrize("async_delivery", [True, False])
async def test_first_contact_home_prompt_requires_async_delivery(monkeypatch, tmp_path, async_delivery):
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    config = PlatformConfig(enabled=True)
    adapter = RecordingAdapter(config)
    if not async_delivery:
        adapter.supports_async_delivery = False
    source = adapter.build_source(chat_id="first-dm", chat_type="dm", user_id="person")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: config})
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.session_store.get_or_create_session(source)
    runner.session_store.get_or_create_session(
        adapter.build_source(chat_id="existing-dm", chat_type="dm", user_id="person"))

    await runner._hmwa_first_contact_notes(source, [], [])
    if async_delivery:
        assert len(adapter.sent) == 1 and "Type /sethome" in adapter.sent[0]
    else:
        assert adapter.sent == []

    # The capability gates only the irrelevant home prompt. Operational errors
    # and the model's own reply still traverse the same real send boundary.
    await runner._deliver_platform_notice(source, "The provider is unavailable.")
    await adapter.send(source.chat_id, "The requested work is complete.")
    assert adapter.sent[-2:] == ["The provider is unavailable.", "The requested work is complete."]
