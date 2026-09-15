"""Routine notices follow the configured home without diverting actionable task output."""
import asyncio
import json
from pathlib import Path

import pytest

from agent.status_output import StatusOutputMixin
from agent.conversation_compression import COMPACTION_STATUS, COMPACTION_DONE_STATUS
import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from agent.i18n import t


class RecordingAdapter(BasePlatformAdapter):
    def __init__(self, platform):
        super().__init__(PlatformConfig(enabled=True), platform)
        self.sent = []
        self.accept = True
        self.delivered = asyncio.Event()

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if self.accept:
            self.sent.append((chat_id, content, metadata))
            self.delivered.set()
        return SendResult(success=self.accept, message_id=str(len(self.sent)))

    async def get_chat_info(self, chat_id):
        return {"type": "dm"}


def setup_gateway(tmp_path, monkeypatch, *, setting=True, home=True, profile=None):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(gateway_run, "_hermes_home", root)
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
    raw = {"gateway": {}, "platforms": {"whatsapp": {"enabled": True}}}
    if setting:
        raw["gateway"]["system_notices_home_platform"] = "whatsapp"
    if home:
        raw["platforms"]["whatsapp"]["home_channel"] = {
            "platform": "whatsapp", "chat_id": "owner-home", "thread_id": "home-topic",
        }
    target = root
    if profile:
        (root / "config.yaml").write_text(json.dumps({
            "gateway": {"system_notices_home_platform": "whatsapp"},
            "platforms": {"whatsapp": {"enabled": True, "home_channel": {
                "platform": "whatsapp", "chat_id": "other-profile-home",
            }}},
        }))
        target = root / "profiles" / profile
        target.mkdir(parents=True)
    (target / "config.yaml").write_text(json.dumps(raw))
    task, owner, other = (RecordingAdapter(p) for p in (Platform.TELEGRAM, Platform.WHATSAPP, Platform.WHATSAPP))
    gateway = object.__new__(GatewayRunner)
    gateway.config = GatewayConfig(multiplex_profiles=bool(profile))
    gateway._primary_profile_name = "default"
    gateway.adapters = {Platform.TELEGRAM: task, Platform.WHATSAPP: other if profile else owner}
    gateway._profile_adapters = {profile: {Platform.TELEGRAM: task, Platform.WHATSAPP: owner}} if profile else {}
    source = task.build_source(chat_id="task-chat", chat_type="dm", user_id="person", thread_id="task-thread")
    source.profile = profile
    ctx = TurnContext(source=source, user_config=raw, _status_adapter=task,
                      _status_chat_id=source.chat_id, _status_thread_metadata={"thread_id": "task-thread"},
                      _run_still_current=lambda: True, _loop_for_step=asyncio.get_running_loop())
    return gateway, TurnRunner(gateway, ctx), source, task, owner, other


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "research"])
async def test_routine_home_delivery_preserves_task_route_and_profile(tmp_path, monkeypatch, profile):
    gateway, turn, source, task, owner, other = setup_gateway(tmp_path, monkeypatch, profile=profile)
    agent = StatusOutputMixin()
    agent.log_prefix = ""
    agent.suppress_status_output = True
    agent.status_callback = turn._status_callback_sync
    agent._emit_system_status("Forgotten content was removed; reloading this session.")
    # Exercise the real sync-worker callback scheduling boundary, with no model or network.
    await asyncio.wait_for(owner.delivered.wait(), timeout=2)
    await gateway._hmwa_hygiene_notify(source, {"thread_id": "task-thread"},
                                     t("gateway.compress.turnhold_deferred"), "deferred", routine=True)
    await turn._deliver_status("warn", "A required task file is missing; choose another file.")
    await gateway._hmwa_hygiene_notify(source, {"thread_id": "task-thread"},
                                     "Compression failed; run /compress to retry.", "failure")
    await task.send(source.chat_id, "The requested work is complete.", metadata={"thread_id": "task-thread"})
    assert [row[1] for row in owner.sent] == [
        "Forgotten content was removed; reloading this session.", t("gateway.compress.turnhold_deferred"),
    ]
    assert all(chat == "owner-home" and meta == {"thread_id": "home-topic", "_interim_send": True}
               for chat, _, meta in owner.sent)
    assert all(chat == "task-chat" and meta["thread_id"] == "task-thread" for chat, _, meta in task.sent)
    assert len(task.sent) == 3 and other.sent == []
    await turn._deliver_status("lifecycle", COMPACTION_STATUS)
    await turn._deliver_status("compacted", COMPACTION_DONE_STATUS)
    assert [row[1] for row in owner.sent[-2:]] == [COMPACTION_STATUS, COMPACTION_DONE_STATUS]
    await turn._deliver_status("system", "Maintenance token sk-" + "a" * 48)
    assert "sk-" + "a" * 48 not in owner.sent[-1][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["default", "no-home", "no-adapter", "failed-send", "missing-profile", "raw-surface"])
async def test_unavailable_home_keeps_existing_conversation(tmp_path, monkeypatch, mode):
    gateway, turn, source, task, owner, other = setup_gateway(
        tmp_path, monkeypatch, setting=mode != "default", home=mode != "no-home",
        profile="research" if mode == "missing-profile" else None,
    )
    if mode == "no-adapter":
        gateway.adapters.pop(Platform.WHATSAPP)
    if mode == "failed-send":
        owner.accept = False
    if mode == "missing-profile":
        source.profile = "missing"
    if mode == "raw-surface":
        source.platform = Platform.API_SERVER
        gateway.adapters[Platform.API_SERVER] = task
    message = "Forgotten content was removed; reloading this session."
    await turn._deliver_status("system", message)
    await gateway._hmwa_hygiene_notify(source, {"thread_id": "task-thread"},
                                     t("gateway.compress.turnhold_deferred"), "deferred", routine=True)
    assert [row[1] for row in task.sent] == [message, t("gateway.compress.turnhold_deferred")]
    assert owner.sent == other.sent == []
