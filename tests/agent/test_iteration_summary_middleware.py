"""Iteration summaries must obey the request contract after tool-loop exhaustion."""

import copy
from types import SimpleNamespace

import pytest


@pytest.fixture
def summary_agent(monkeypatch):
    import httpx
    from hermes_cli import plugins
    from run_agent import AIAgent

    def no_network(_transport, request):
        raise AssertionError(f"Unexpected network request: {request.method} {request.url.host}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    agent = AIAgent(
        api_key="fixture-key", base_url="http://summary.fixture/v1",
        model="fixture-model", provider="custom", enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        session_id="summary-session", max_iterations=2, max_tokens=777,
    )
    agent._cached_system_prompt = "Stable system context."
    agent._current_task_id = "bound-task"
    agent._current_turn_id = "bound-turn"
    agent.platform = "cli"
    try:
        yield agent, manager
    finally:
        agent.close()


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages", "codex_responses"])
@pytest.mark.parametrize("policy", ["refresh", "withhold"])
def test_summary_rechecks_each_attempt_without_rebinding_the_runtime_nudge(
        summary_agent, monkeypatch, api_mode, policy):
    from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST

    agent, manager = summary_agent
    agent.api_mode = api_mode
    agent.reasoning_config = {"enabled": True, "effort": "low"}
    agent._is_anthropic_oauth = False
    agent._anthropic_base_url = "http://summary.fixture"
    requests, callbacks = [], []
    history = [
        {"role": "user", "content": "Give a status report.",
         "api_content": "Give a status report.\nRetired source: old-location."},
        {"role": "assistant", "tool_calls": [{"id": "lookup-1", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path":"old-location"}'}}]},
        {"role": "tool", "tool_call_id": "lookup-1", "content": "Retired source: old-location."},
    ]
    original = copy.deepcopy(history)

    def summarize(request):
        requests.append(copy.deepcopy(request))
        return "" if len(requests) == 1 else "Checked status."

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: summarize(kw))))
    monkeypatch.setattr(agent, "_ensure_primary_openai_client", lambda **_: client)
    monkeypatch.setattr(agent, "_anthropic_messages_create", summarize)
    monkeypatch.setattr(agent, "_run_codex_stream", summarize)
    # Provider-specific builders stay real; only response parsing is irrelevant here.
    monkeypatch.setattr(agent._get_transport(), "normalize_response",
                        lambda response, **_: SimpleNamespace(content=response))

    def reconcile(**context):
        callbacks.append(context)
        request = context["request"]
        key = "input" if api_mode == "codex_responses" else "messages"
        request[key] = [{"role": "user", "content": (
            "Source is unavailable; do not claim completion." if policy == "withhold"
            else f"Current source revision {context['retry_count']}.")} ]
        return {"request": request, "source": "fixture-source-policy"}

    manager._middleware["llm_request"] = [reconcile]
    assert agent._handle_max_iterations(history, 2) == "Checked status."
    assert len(callbacks) == len(requests) == 2
    assert [ctx["retry_count"] for ctx in callbacks] == [0, 1]
    assert len({ctx["api_request_id"] for ctx in callbacks}) == 1
    for ctx, sent in zip(callbacks, requests):
        assert (ctx["session_id"], ctx["task_id"], ctx["turn_id"]) == (
            "summary-session", "bound-task", "bound-turn")
        assert ctx["platform"] == "cli"
        assert ctx["api_mode"] == api_mode
        assert ctx["call_role"] == "iteration_summary"
        assert ctx["api_call_count"] == 2
        assert "native_user_message" not in ctx
        assert "original_user_message" not in ctx
        key = "input" if api_mode == "codex_responses" else "messages"
        assert "old-location" not in str(sent[key])
        assert MAX_ITERATIONS_SUMMARY_REQUEST not in str(sent[key])
        assert sent[key] == ctx["request"][key]
        assert "tools" not in sent
        # The request-policy replacement leaves provider controls byte-for-byte intact.
        assert {k: v for k, v in sent.items() if k != key} == {
            k: v for k, v in ctx["original_request"].items() if k != key}
        assert "old-location" in str(ctx["original_request"][key])
    assert history[:len(original)] == original
    assert history[-2]["content"] == MAX_ITERATIONS_SUMMARY_REQUEST
    assert history[-1]["content"] == "Checked status."


def test_budget_exhaustion_preserves_native_turn_scope_through_sdk_request(
        summary_agent, monkeypatch):
    import json
    import httpx

    agent, manager = summary_agent
    agent.max_iterations = 1
    agent._disable_streaming = True
    agent._skip_memory_review = True
    agent._skip_skill_review = True
    callbacks, sent = [], []

    def broken_observer(**_):
        raise RuntimeError("unrelated observer failure")

    def reconcile(**context):
        callbacks.append(context)
        if context.get("call_role") == "iteration_summary":
            # An explicit policy rejection replaces the request, exactly as in
            # the regular loop; raising in a callback deliberately does not.
            return {"request": {**context["request"], "messages": [
                {"role": "user", "content": "Current source access was rejected."}]}}

    manager._middleware["llm_request"] = [broken_observer, reconcile]

    def respond(_transport, request):
        assert request.url.host == "summary.fixture"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fixture-model", "context_length": 32768}]})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        sent.append(body)
        content = {"role": "assistant", "content": "No current source access."}
        finish = "stop"
        if len(sent) == 1:
            content = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "unavailable-tool", "type": "function", "function": {
                    "name": "unavailable_fixture_tool", "arguments": "{}"}}]}
            finish = "tool_calls"
        return httpx.Response(200, json={"id": f"attempt-{len(sent)}", "created": 1,
            "object": "chat.completion", "model": "fixture-model", "choices": [
                {"index": 0, "message": content, "finish_reason": finish}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    result = agent.run_conversation("Give a status report.", task_id="actual-summary-task")
    assert result["final_response"] == "No current source access."
    normal = [ctx for ctx in callbacks if ctx.get("call_role") != "iteration_summary"]
    summary = [ctx for ctx in callbacks if ctx.get("call_role") == "iteration_summary"]
    assert len(normal) == len(summary) == 1
    assert len(sent) == 2
    assert sent[-1]["messages"] == [{"role": "user", "content": "Current source access was rejected."}]
    for key in ("session_id", "task_id", "turn_id", "platform"):
        assert summary[0][key] == normal[0][key]
    assert summary[0]["task_id"] == "actual-summary-task"
    assert summary[0]["turn_id"] == result["turn_id"]
    assert normal[0]["original_user_message"] == "Give a status report."
    assert "native_user_message" not in summary[0]
    assert "original_user_message" not in summary[0]


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages", "codex_responses"])
@pytest.mark.parametrize("enabled", [False, True])
def test_summary_debug_dump_retains_transformed_sent_body_only_when_enabled(
        summary_agent, monkeypatch, tmp_path, api_mode, enabled):
    import json

    agent, manager = summary_agent
    agent.api_mode = api_mode
    agent._is_anthropic_oauth = False
    agent._anthropic_base_url = "http://summary.fixture"
    agent.logs_dir = tmp_path / "request-logs"
    agent.logs_dir.mkdir()
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1" if enabled else "0")
    monkeypatch.delenv("HERMES_DUMP_REQUEST_STDOUT", raising=False)
    sent, retained = [], []
    key = "input" if api_mode == "codex_responses" else "messages"

    def reconcile(**context):
        return {"request": {**context["request"], key: [
            {"role": "user", "content": f"Checked current revision {context['retry_count']}."}]}}

    manager._middleware["llm_request"] = [reconcile]

    def summarize(request):
        # A dump must already exist at the provider boundary, including retry.
        dumps = sorted(agent.logs_dir.glob("request_dump_*.json"))
        assert len(dumps) == (len(sent) + 1 if enabled else 0)
        if enabled:
            retained.append(json.loads(dumps[-1].read_text()))
        sent.append(copy.deepcopy(request))
        return "" if len(sent) == 1 else "Checked status."

    client = SimpleNamespace(api_key="fixture-key", chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: summarize(kw))))
    monkeypatch.setattr(agent, "_ensure_primary_openai_client", lambda **_: client)
    monkeypatch.setattr(agent, "_anthropic_messages_create", summarize)
    monkeypatch.setattr(agent, "_run_codex_stream", summarize)
    monkeypatch.setattr(agent._get_transport(), "normalize_response",
                        lambda response, **_: SimpleNamespace(content=response))
    history = [{"role": "user", "content": "Old context before request reconciliation."}]
    assert agent._handle_max_iterations(history, 2) == "Checked status."
    assert len(sent) == 2
    assert len(retained) == (2 if enabled else 0)
    for dump, request in zip(retained, sent):
        assert dump["reason"] == "iteration_summary"
        assert dump["session_id"] == agent.session_id
        # timeout is an SDK transport option, not part of the submitted body.
        assert dump["request"]["body"] == {k: v for k, v in request.items() if k != "timeout"}
        assert "Old context" not in json.dumps(dump["request"]["body"])
