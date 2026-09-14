import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
    )
    runner.adapters = {}
    runner._pending_native_image_paths_by_session = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="273403055",
        chat_type="dm",
        user_id="42",
        user_name="Maxim",
    )


def _image_event(text: str = "look") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.PHOTO,
        source=_source(),
        media_urls=["/tmp/cashback.png"],
        media_types=["image/png"],
    )


def _auto_config() -> dict:
    return {
        "agent": {"image_input_mode": "auto"},
        "auxiliary": {"vision": {"provider": "auto", "model": "", "base_url": ""}},
        "model": {"provider": "xiaomi", "default": "mimo-v2.5-pro"},
    }


def test_pre_turn_named_custom_provider_identity_selects_vision_override(monkeypatch):
    """Gateway preprocessing must use the name retained by runtime resolution."""
    runner = _make_runner()
    cfg = {
        "agent": {"image_input_mode": "auto"},
        "model": {"provider": "default-proxy", "default": "shared-model"},
        "custom_providers": [
            {
                "name": "default-proxy",
                "models": {"shared-model": {"supports_vision": False}},
            },
            {
                "name": "vision-provider",
                "models": {"shared-model": {"supports_vision": True}},
            },
        ],
    }
    monkeypatch.setattr(
        runner,
        "_resolve_session_agent_runtime",
        lambda **_: (
            "shared-model",
            {
                "provider": "custom",
                "requested_provider": "vision-provider",
            },
        ),
    )

    assert runner._decide_image_input_mode(
        source=_source(),
        user_config=cfg,
    ) == "native"


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", [True, False, None])
async def test_native_if_supported_preserves_question_pixels_and_auxiliary(
    tmp_path, monkeypatch, capability,
):
    """The gateway and registered tool use the same actual per-turn capability."""
    import asyncio
    import base64
    import json
    from types import SimpleNamespace

    from agent import image_routing
    from agent.auxiliary_client import (
        aux_probe_mode, resolve_vision_provider_client, scoped_runtime_main,
    )
    from gateway.run_turn_runner import TurnRunner
    from hermes_cli.config import load_config
    from tools import vision_tools
    from tools.registry import registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # No catalog network lookup for the deliberately unknown capability control.
    monkeypatch.setattr(image_routing, "_VISION_PROBES", ())
    cfg = {
        "agent": {"image_input_mode": "native_if_supported"},
        "model": {"provider": "default-provider", "default": "shared-model"},
        "providers": {
            "default-provider": {"models": {"shared-model": {"supports_vision": True}}},
            "session-provider": {"models": {"shared-model": {
                **({"supports_vision": capability} if capability is not None else {}),
            }}},
        },
        "auxiliary": {"vision": {
            "provider": "custom", "model": "fallback-vision",
            "base_url": "http://127.0.0.1:54321/v1", "api_key": "test-key",
        }},
    }
    # False on the session provider takes precedence over the capable default;
    # the unknown control has no declared capability in either entry.
    if capability is None:
        cfg["providers"]["default-provider"]["models"] = {}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    assert load_config()["agent"]["image_input_mode"] == "native_if_supported"
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    image = tmp_path / "view.png"
    image.write_bytes(raw)
    question = "Is this view useful for checking the work area? State its limits briefly."
    runtime = {
        "provider": "custom", "requested_provider": "session-provider",
        "model": "shared-model", "base_url": "http://127.0.0.1:54322/v1",
    }
    calls = []

    async def auxiliary_response(**kwargs):
        # Keep native media preparation and configured auxiliary resolution real;
        # replace only the completion, without opening a server or making inference.
        with aux_probe_mode():
            provider, client, model = resolve_vision_provider_client(model=kwargs.get("model"))
            assert provider == "custom"
            assert str(client.base_url).rstrip("/") == cfg["auxiliary"]["vision"]["base_url"]
            assert model == "fallback-vision"
        assert kwargs["task"] == "vision"
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Derived image description.", reasoning_content=None),
        )])

    monkeypatch.setattr(vision_tools, "async_call_llm", auxiliary_response)
    runner = _make_runner()
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", lambda **_: (
        runtime["model"], runtime,
    ))
    source = _source()
    session_key = "current-image-session"
    message = await runner._enrich_inbound_images(source, session_key, question, [str(image)])
    turn = TurnRunner(runner, SimpleNamespace(session_key=session_key, message=message))
    prepared = turn._native_image_run_message()
    with scoped_runtime_main(runtime):
        result = await asyncio.to_thread(
            registry.dispatch, "vision_analyze", {"image_url": str(image), "question": question},
        )
    if capability is True:
        assert calls == []
        assert message == question
        assert prepared[0]["type"] == "text"
        assert prepared[0]["text"].startswith(question)
        assert str(image) in prepared[0]["text"]
        assert base64.b64decode(prepared[1]["image_url"]["url"].split(",", 1)[1]) == raw
        assert result["_multimodal"] is True
        image_part = next(p for p in result["content"] if p["type"] == "image_url")
        assert base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1]) == raw
    else:
        assert len(calls) == 2
        assert isinstance(prepared, str)
        assert question in prepared and "Derived image description." in prepared
        assert json.loads(result)["analysis"] == "Derived image description."
        assert question in calls[-1]["messages"][0]["content"][0]["text"]
    assert runner._consume_pending_native_image_paths(session_key) == []
    assert image.read_bytes() == raw


@pytest.mark.asyncio
async def test_prepare_route_identity_check_keeps_event_loop_responsive(monkeypatch):
    """A slow route-identity check must not block gateway heartbeats."""
    import asyncio
    import threading
    from types import SimpleNamespace

    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="inspect @AGENTS.md",
        message_type=MessageType.TEXT,
        source=source,
    )
    started = threading.Event()
    released_by_event_loop = threading.Event()
    seen = {}
    main_thread = threading.current_thread()

    cfg = {
        "model": {
            "default": "test-model",
            "provider": "test-provider",
            "base_url": "https://example.invalid/v1",
            "context_length": 128000,
        }
    }
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: cfg)
    monkeypatch.setattr(
        runner,
        "_resolve_session_agent_runtime",
        lambda **_kwargs: (
            "test-model",
            {
                "provider": "test-provider",
                "base_url": "https://example.invalid/v1",
                "api_key": "",
            },
        ),
    )

    def blocking_route_identity_check(*_args):
        seen["thread"] = threading.current_thread()
        started.set()
        seen["event_loop_progressed"] = released_by_event_loop.wait(timeout=2)
        return False

    monkeypatch.setattr(
        "hermes_cli.route_identity.should_clear_context_pin",
        blocking_route_identity_check,
    )

    async def fake_context_length(*_args, **_kwargs):
        return 128000

    async def fake_preprocess(message, **_kwargs):
        return SimpleNamespace(
            blocked=False,
            expanded=False,
            message=message,
            warnings=[],
        )

    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length_async", fake_context_length
    )
    monkeypatch.setattr(
        "agent.context_references.preprocess_context_references_async",
        fake_preprocess,
    )

    async def heartbeat_ticker():
        while not started.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        released_by_event_loop.set()

    heartbeat = asyncio.create_task(heartbeat_ticker())
    result = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[]
    )
    await heartbeat

    assert result == "inspect @AGENTS.md"
    assert seen["event_loop_progressed"] is True
    assert seen["thread"] is not main_thread
