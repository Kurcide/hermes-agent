"""A runtime handoff ends only its current, durably completed tool round."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


TOOL = "fixture_background_handoff"
ARGUMENTS = '{"request": "accepted background work"}'
TOOL_RESULT = json.dumps({"accepted": True, "task_id": "background-accepted"})
RECEIPT = "Accepted background task background-accepted."


def _response(*, calls=(), text=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=list(calls) or None),
            finish_reason="tool_calls" if calls else "stop",
        )],
        model="fixture/model",
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
    )


def _call(call_id="handoff-1", arguments=ARGUMENTS):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=TOOL, arguments=arguments,
    ))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    (tmp_path / "config.yaml").write_text(json.dumps({
        "auxiliary": {"title_generation": {"enabled": False}},
        "model": {"context_length": 131072},
    }), encoding="utf-8")
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    context = PluginContext(PluginManifest(name="handoff-fixture", source="user"), manager)
    schema = {"name": TOOL, "description": "Accept explicitly scoped background work.",
              "parameters": {"type": "object", "properties": {"request": {"type": "string"}}}}
    state = SimpleNamespace(scenario="success", executed=[], callbacks=[], api=[], transforms=[],
                            finalization_rows=[])

    def handler(args, **kwargs):
        state.executed.append((args, kwargs))
        if state.scenario == "steer":
            state.agent.steer("Also answer my new foreground question.")
        if state.scenario == "interrupt":
            state.agent.interrupt()
        return json.dumps({"error": "Admission failed"}) if state.scenario == "error" else TOOL_RESULT

    registration = context.register_tool(TOOL, "handoff-fixture", schema, handler)
    assert registration is not None
    database = SessionDB(tmp_path / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": schema}]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="fixture-key", base_url="https://fixture.invalid/v1",
                        provider="openai-compat", model="fixture/model", session_id="foreground",
                        max_iterations=3, quiet_mode=True, skip_context_files=True,
                        skip_memory=True, skip_background_review=True, session_db=database)
    state.agent, state.database, state.context = agent, database, context
    agent.client = MagicMock()
    agent._cached_system_prompt = "A stable fixture prompt."
    agent._disable_streaming = True
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.tool_delay = 0
    agent._save_trajectory = lambda *a, **kw: state.finalization_rows.append(database.get_messages(agent.session_id))

    def complete(**kwargs):
        from hermes_cli.tool_completion import FinishTurn

        state.callbacks.append(kwargs)
        rows = database.get_messages(agent.session_id)
        persisted = [row for row in rows if row.get("tool_call_id") == kwargs["tool_call_id"]]
        assert persisted and persisted[-1]["content"] == kwargs["tool_result"]
        if state.scenario == "hook_steer":
            state.agent.steer("Also answer my new foreground question.")
        if state.scenario == "hook_interrupt":
            state.agent.interrupt()
        if state.scenario == "untyped":
            return {"text": RECEIPT, "tool_call_id": kwargs["tool_call_id"]}
        return FinishTurn(text=RECEIPT, tool_call_id=(
            "a-different-call" if state.scenario == "wrong_call" else kwargs["tool_call_id"]))

    context.register_hook("post_tool_batch", complete)
    context.register_hook("post_api_request", lambda **kwargs: state.api.append(kwargs))

    def transform(**kwargs):
        state.transforms.append(kwargs)
        return "Transformed ordinary model answer."

    context.register_hook("transform_llm_output", transform)
    try:
        yield state
    finally:
        agent.close()
        manager.unload()
        database.close()


@pytest.mark.parametrize("platform,streaming", [("cli", False), ("telegram", True)])
def test_runtime_receipt_is_durable_at_budget_limit_and_next_turn_is_normal(runtime, monkeypatch, platform, streaming):
    agent = runtime.agent
    agent.platform = platform
    agent.max_iterations = agent.iteration_budget.max_total = 1
    agent._disable_streaming = not streaming
    agent.stream_delta_callback = MagicMock() if streaming else None
    responses = iter([_response(calls=[_call()]), _response(text="Ordinary model answer.")])
    provider = MagicMock(side_effect=lambda *a, **kw: next(responses))
    monkeypatch.setattr(agent, "_interruptible_api_call", provider)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", provider)
    summary = MagicMock(side_effect=AssertionError("A completed handoff must not ask for a summary"))
    monkeypatch.setattr(agent, "_handle_max_iterations", summary)

    result = agent.run_conversation("Hand off this entire request.", task_id="foreground-task")

    assert result["final_response"] == RECEIPT
    assert result["completed"] and not result["failed"] and not result["interrupted"]
    assert result["turn_exit_reason"] == "tool_handoff"
    assert result["response_origin"] == "runtime"
    assert result["api_calls"] == provider.call_count == len(runtime.api) == 1
    assert not runtime.transforms and not result["response_transformed"]
    assert len(runtime.executed) == len(runtime.callbacks) == 1
    observed = runtime.callbacks[0]
    assert observed["tool_name"] == TOOL and observed["tool_arguments"] == ARGUMENTS
    assert observed["tool_result"] == TOOL_RESULT and observed["platform"] == platform
    provenance = {key: observed[key] for key in (
        "session_id", "task_id", "turn_id", "api_request_id", "tool_name", "tool_call_id")}
    assert all(provenance.values())
    assert provenance["task_id"] == "foreground-task"
    assert result["response_provenance"] == {"type": "tool_handoff", **provenance}
    rows = runtime.database.get_messages(agent.session_id)
    assert [row["role"] for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[-1]["content"] == RECEIPT and rows[-1]["display_kind"] == "runtime_handoff"
    assert rows[-1]["display_metadata"] == {"response_origin": "runtime", "type": "tool_handoff", **provenance}
    assert runtime.finalization_rows[0][-1]["id"] == rows[-1]["id"]

    following = agent.run_conversation("Now answer a separate question.", conversation_history=result["messages"])
    assert following["final_response"] == "Transformed ordinary model answer."
    assert following.get("response_origin") != "runtime"
    assert following.get("response_provenance") != result["response_provenance"]
    assert provider.call_count == len(runtime.api) == 2 and len(runtime.transforms) == 1
    assert len(runtime.callbacks) == len(runtime.executed) == 1
    assert len([row for row in runtime.database.get_messages(agent.session_id)
                if row.get("display_kind") == "runtime_handoff"]) == 1
    summary.assert_not_called()


@pytest.mark.parametrize("scenario", [
    "mixed", "error", "steer", "interrupt", "hook_steer", "hook_interrupt",
    "wrong_call", "untyped", "persistence", "receipt_persistence",
])
def test_ineligible_or_uncommitted_round_cannot_finish_with_a_runtime_receipt(runtime, monkeypatch, scenario):
    runtime.scenario = scenario
    agent = runtime.agent
    calls = [_call(), _call("sibling-2", '{"request":"another foreground obligation"}')] if scenario == "mixed" else [_call()]
    responses = iter([_response(calls=calls), _response(text="The foreground still needs an answer.")])
    provider = MagicMock(side_effect=lambda *a, **kw: next(responses))
    monkeypatch.setattr(agent, "_interruptible_api_call", provider)
    if scenario in {"persistence", "receipt_persistence"}:
        original = runtime.database.append_messages_batch

        def fail_tool_write(*args, **kwargs):
            rows = kwargs.get("messages", args[1] if len(args) > 1 else [])
            if any(
                row.get("role") == "tool" if scenario == "persistence"
                else row.get("display_kind") == "runtime_handoff"
                for row in rows
            ):
                raise RuntimeError("Controlled SQLite receipt write failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(runtime.database, "append_messages_batch", fail_tool_write)

    result = agent.run_conversation("Perform these foreground requests.", task_id="foreground-task")

    assert result.get("response_origin") != "runtime"
    assert result["turn_exit_reason"] != "tool_handoff"
    assert result["final_response"] != RECEIPT
    assert not any(row.get("display_kind") == "runtime_handoff"
                   for row in runtime.database.get_messages(agent.session_id))
    assert len(runtime.executed) == len(calls)
    if scenario in {"interrupt", "hook_interrupt", "persistence", "receipt_persistence"}:
        assert result["interrupted"] or result["failed"]
        assert provider.call_count == 1
    else:
        assert result["completed"] and provider.call_count == 2
        persisted = runtime.database.get_messages(agent.session_id)
        assert {row["tool_call_id"] for row in persisted if row["role"] == "tool"} == {call.id for call in calls}
        if scenario in {"steer", "hook_steer"}:
            assert any("Also answer my new foreground question." in str(row.get("content")) for row in persisted)
    assert len(runtime.callbacks) == (1 if scenario in {
        "wrong_call", "untyped", "hook_steer", "hook_interrupt", "receipt_persistence",
    } else 0)
