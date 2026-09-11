"""Regression test for #8049.

When the post-loop cleanup chain in ``finalize_turn`` raises — trajectory
save (file I/O), resource teardown (remote VM/browser), or session
persistence (SQLite) — the partial ``final_response`` the caller is waiting
for must still be returned.  Previously any of those raised straight out of
``run_conversation``, so a subprocess wrapper saw an empty stdout with no
traceback and lost the whole turn.
"""

from unittest.mock import patch

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, raise_in):
        self._raise_in = set(raise_in)
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    # --- fallible cleanup surfaces -------------------------------------
    def _save_trajectory(self, *a, **k):
        if "save_trajectory" in self._raise_in:
            raise RuntimeError("trajectory disk full")

    def _cleanup_task_resources(self, *a, **k):
        if "cleanup_task_resources" in self._raise_in:
            raise RuntimeError("docker teardown EOF")

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        if "persist_session" in self._raise_in:
            raise RuntimeError("sqlite database is locked")

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _run(
    agent,
    *,
    final_response=None,
    api_call_count=3,
    turn_exit_reason="unknown",
    interrupted=False,
    failed=False,
):
    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )




@pytest.mark.parametrize(
    "step", ["save_trajectory", "cleanup_task_resources", "persist_session"]
)
def test_single_cleanup_step_raises_does_not_skip_others(step):
    agent = _StubAgent(raise_in=(step,))
    result = _run(agent)
    # Response survives.
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    # Exactly the failing step is recorded; the others ran without error.
    assert result["cleanup_errors"] == [
        next(
            e
            for e in result["cleanup_errors"]
            if e.startswith(step)
        )
    ]
    assert len(result["cleanup_errors"]) == 1


def test_clean_turn_has_no_cleanup_errors_key():
    agent = _StubAgent(raise_in=())
    result = _run(agent)
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    assert result["completed"] is False
    assert "cleanup_errors" not in result


@pytest.mark.parametrize("persist_disabled", [True, False])
@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
def test_detached_end_observation_does_not_publish_a_session_turn(
    monkeypatch, persist_disabled, outcome
):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    agent = _StubAgent(raise_in=())
    agent._persist_disabled = persist_disabled
    agent._parent_session_id = agent.session_id if persist_disabled else None
    calls = []
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="completion-observer", version="1.0.0"), manager)
    for name in ("transform_llm_output", "post_llm_call", "on_session_end", "on_detached_turn_end"):
        def capture(_name=name, **kwargs):
            calls.append((_name, kwargs))
            return {"user_message": "must not be ingested", "completed": False}
        assert ctx.register_hook(name, capture) is not None
    # Keep the real finalizer, lifecycle and plugin dispatch. Only substitute
    # this isolated manager; observer returns must not change the native result.
    monkeypatch.setattr("hermes_cli.lifecycle._plugin_hooks", manager.invoke_hook)
    final_response = None if outcome == "interrupted" else "private final response"
    result = _run(agent, final_response=final_response, api_call_count=1,
                  interrupted=outcome == "interrupted", failed=outcome == "failed",
                  turn_exit_reason="text_response(stop)" if outcome == "completed" else outcome)

    end_name = "on_detached_turn_end" if persist_disabled else "on_session_end"
    expected = [] if outcome == "interrupted" else ["transform_llm_output"]
    if not persist_disabled and outcome != "interrupted":
        expected.append("post_llm_call")
    assert [name for name, _ in calls] == expected + [end_name]
    payload = calls[-1][1]
    assert payload["session_id"] == agent.session_id
    assert payload["task_id"] == "task-1" and payload["turn_id"] == "turn-1"
    for key in ("completed", "failed", "interrupted", "turn_exit_reason"):
        assert payload[key] == result[key]
    assert result["completed"] is (outcome == "completed")
    if persist_disabled:
        assert payload["parent_session_id"] == agent.session_id
        assert set(payload) == {"session_id", "parent_session_id", "task_id", "turn_id",
                                "completed", "failed", "interrupted", "turn_exit_reason", "model", "platform",
                                "telemetry_schema_version"}
        assert all(isinstance(value, (str, bool, int)) for value in payload.values())
        assert "private final response" not in str(payload)
        assert "do a thing" not in str(payload)
