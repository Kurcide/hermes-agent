"""A delegated answer returns to its parent; only the claimed worker closes the card."""

import contextvars
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context


@pytest.fixture
def claimed_worker(tmp_path, monkeypatch):
    from hermes_cli.kanban_db import claim_task, create_task
    from hermes_cli.kanban_db_connect import connect_closing

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "bundled"))
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("The native loop fixture must not connect to a network endpoint")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    with connect_closing() as conn:
        tid = create_task(conn, title="Read the selected release", assignee="default")
        task = claim_task(conn, tid, claimer="test-dispatcher")
    assert task is not None and task.status == "running"
    for name, value in {
        "HERMES_KANBAN_TASK": tid,
        "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
        "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock,
    }.items():
        monkeypatch.setenv(name, value)
    return task


def _response(content, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason="tool_calls" if tool_calls else "stop",
        )], model="test/model", usage=None,
    )


def _agent(*, platform, disabled_toolsets=None):
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key", base_url="https://example.invalid/v1",
            provider="openai-compat", model="test/model", max_iterations=4,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            platform=platform, enabled_toolsets=["kanban"],
            disabled_toolsets=disabled_toolsets,
        )
    agent.save_trajectories = False
    agent.compression_enabled = False
    return agent


@pytest.mark.parametrize("context", [delegated_child_context, non_dispatcher_owned_context])
def test_in_process_child_returns_answer_without_claiming_parent_task(claimed_worker, context):
    from hermes_cli.kanban_db import get_task
    from hermes_cli.kanban_db_connect import connect_closing

    requests = []
    with context():
        child = _agent(platform="subagent", disabled_toolsets=["kanban"])
        assert not {"kanban_complete", "kanban_block"}.intersection(child.valid_tool_names)
        child._interruptible_api_call = lambda request: (
            requests.append(request) or _response("Selected release findings.")
        )
        try:
            # The delegate runner carries the existing ContextVar into its worker thread.
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(contextvars.copy_context().run,
                    child.run_conversation, "Return the release findings to the parent.").result(timeout=30)
        finally:
            child.close()
    assert result["completed"] is True
    assert result["final_response"] == "Selected release findings."
    assert len(requests) == 1
    assert not any(row.get("_kanban_stop_synthetic") for row in result["messages"])
    assert os.environ["HERMES_KANBAN_TASK"] == claimed_worker.id
    from agent.kanban_stop import build_kanban_stop_nudge
    assert build_kanban_stop_nudge(messages=[]) is not None
    with connect_closing() as conn:
        assert get_task(conn, claimed_worker.id).status == "running"


@pytest.mark.parametrize("terminal", ["kanban_complete", "kanban_block"])
def test_claimed_parent_must_use_its_terminal_tool(claimed_worker, terminal):
    from hermes_cli.kanban_db import get_task, latest_run
    from hermes_cli.kanban_db_connect import connect_closing

    parent = _agent(platform="cli")
    assert terminal in parent.valid_tool_names
    requests = []

    def response(request):
        requests.append(request)
        if len(requests) == 1:
            return _response("Work is done.")
        if len(requests) == 2:
            with connect_closing() as conn:
                assert get_task(conn, claimed_worker.id).status == "running"
            assert any("plain-text reply is NOT" in str(row.get("content", ""))
                       for row in request["messages"])
            args = {"summary": "Selected release verified."} if terminal == "kanban_complete" else {
                "reason": "The required release file is missing."}
            return _response("", [SimpleNamespace(id="terminal-call", type="function",
                function=SimpleNamespace(name=terminal, arguments=json.dumps(args)))])
        return _response("Retained task outcome.")

    parent._interruptible_api_call = response
    try:
        result = parent.run_conversation("Finish the assigned release inspection.")
    finally:
        parent.close()
    assert len(requests) == 3
    assert result["completed"] is True
    assert result["final_response"] == "Retained task outcome."
    with connect_closing() as conn:
        assert get_task(conn, claimed_worker.id).status == (
            "done" if terminal == "kanban_complete" else "blocked")
        assert latest_run(conn, claimed_worker.id).id == claimed_worker.current_run_id
