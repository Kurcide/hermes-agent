"""Named-provider timeout policy survives transport canonicalization."""
import json

import httpx
import pytest


def _config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "model": {"provider": "custom:local-workers", "default": "fixture-model"},
        "providers": {
            "disabled": {
                "name": "Local Workers", "enabled": False,
                "base_url": "http://disabled.fixture/v1", "request_timeout_seconds": 11,
                "stale_timeout_seconds": 7,
            },
            "endpointless": {
                "name": "Local Workers", "request_timeout_seconds": 13,
                "stale_timeout_seconds": 9,
            },
            "local-workers": {
                "name": "Local Workers", "base_url": "http://model.fixture/v1",
                "api_key": "disposable-fixture", "request_timeout_seconds": 67,
                "stale_timeout_seconds": 29,
                "models": {"fixture-model": {"timeout_seconds": 43, "stale_timeout_seconds": 19}},
            },
            "custom": {"request_timeout_seconds": 83, "stale_timeout_seconds": 37},
            "openrouter": {"request_timeout_seconds": 97, "stale_timeout_seconds": 41},
        },
        "agent": {"environment_probe": False},
        "auxiliary": {"title_generation": {"enabled": False}},
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    return config


@pytest.mark.parametrize("first_read_timeout", [False, True])
def test_named_custom_timeout_reaches_client_and_actual_stream_wire(
        tmp_path, monkeypatch, first_read_timeout):
    _config(tmp_path, monkeypatch)
    requests = []

    def respond(transport, request):
        assert request.url.host == "model.fixture"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fixture-model", "context_length": 32768}]})
        if request.url.path == "/api/show":
            return httpx.Response(404)
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(request)
        assert body.get("stream") is True
        if first_read_timeout and len(requests) == 1:
            raise httpx.ReadTimeout("Controlled first-attempt timeout", request=request)
        chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 1,
                 "model": body["model"], "choices": [{"index": 0,
                     "delta": {"role": "assistant", "content": "Configured timeout retained."},
                     "finish_reason": None}]}
        end = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(
            "data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(end) + "\n\ndata: [DONE]\n\n").encode())

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    from gateway.run import _resolve_runtime_agent_kwargs
    from run_agent import AIAgent
    from agent.chat_completion_helpers import _configured_stale_base

    runtime = _resolve_runtime_agent_kwargs()
    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == "custom:local-workers"
    agent = AIAgent(model="fixture-model", **runtime, max_iterations=2,
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    enabled_toolsets=[], session_id="named-timeout-fixture")
    try:
        assert httpx.Timeout(agent.client.timeout).read == 43
        assert agent._resolved_api_call_timeout() == 43
        assert agent._resolved_api_call_stale_timeout_base() == (19, False)
        assert agent._stale_timeout_is_explicit()
        assert _configured_stale_base(agent) == 19
        result = agent.run_conversation("Reply with one sentence; no tools needed.")
        assert result["final_response"] == "Configured timeout retained."
        assert len(requests) == (2 if first_read_timeout else 1)
        assert all(request.extensions["timeout"]["read"] == 43 for request in requests)
        assert all(request.extensions["timeout"]["write"] == 43 for request in requests)
    finally:
        agent.close()


@pytest.mark.parametrize("provider,requested,model,expected", [
    ("custom", "local-workers", "fixture-model", (43, 19)),
    ("custom", "Local Workers", "other-model", (67, 29)),
    ("custom", "custom:local-workers", "fixture-model", (43, 19)),
    ("custom:local-workers", None, "fixture-model", (43, 19)),
    ("custom:Local Workers", None, "other-model", (67, 29)),
    ("local-workers", "custom:local-workers", "fixture-model", (43, 19)),
    ("custom", "removed-endpoint", "fixture-model", (83, 37)),
    ("custom:removed-endpoint", None, "fixture-model", (83, 37)),
    ("openrouter", "local-workers", "fixture-model", (97, 41)),
])
def test_timeout_policy_uses_selected_named_identity_and_preserves_fallback(
        tmp_path, monkeypatch, provider, requested, model, expected):
    _config(tmp_path, monkeypatch)
    from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout
    assert (
        get_provider_request_timeout(provider, model, requested_provider=requested),
        get_provider_stale_timeout(provider, model, requested_provider=requested),
    ) == expected
