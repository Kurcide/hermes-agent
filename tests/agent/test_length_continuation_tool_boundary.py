"""Text fragments stop belonging to the final reply when a tool round intervenes."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_run_agent import _make_tool_defs, _mock_response, _mock_tool_call


@pytest.fixture
def loop_agent(tmp_path):
    from run_agent import AIAgent
    from hermes_state import SessionDB

    state = SessionDB(tmp_path / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("read_file")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            model="fixture-model", api_key="fixture-key", base_url="http://fixture.invalid/v1",
            max_iterations=12, quiet_mode=True, skip_context_files=True, skip_memory=True,
            skip_background_review=True, save_trajectories=False,
            session_db=state,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "Use the supplied local note."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    try:
        yield agent
    finally:
        agent.close()
        state.close()


def tool_response(arguments, call_id, name="read_file"):
    return _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call(name=name, arguments=arguments, call_id=call_id),
    ])


@pytest.mark.parametrize("rounds,arguments_valid", [(1, True), (2, True), (1, False)])
def test_completed_tool_round_ends_prior_text_segment(loop_agent, tmp_path, rounds, arguments_valid):
    notes = [tmp_path / f"note-{index}.txt" for index in range(rounds)]
    for note in notes:
        note.write_text("The instrument is ready.\n")
    fragments = [f"Inspecting note {index}." for index in range(rounds)]
    responses = []
    for index, fragment in enumerate(fragments):
        responses.extend([
            _mock_response(content=fragment, finish_reason="length"),
            tool_response(json.dumps({"path": str(notes[index])}) if arguments_valid else "[]", f"read-{index}"),
        ])
    # A later text-only continuation still belongs to the final answer.
    responses.extend([
        _mock_response(content="The observation", finish_reason="length"),
        _mock_response(content="is complete."),
    ])
    loop_agent.client.chat.completions.create.side_effect = responses

    result = loop_agent.run_conversation("Read the local note and report the observation.")

    assert result["completed"] is True
    assert result["final_response"] == "The observation\nis complete."
    for messages in (result["messages"], loop_agent._session_db.get_messages(loop_agent.session_id)):
        assert all(any(message.get("content") == fragment for message in messages) for fragment in fragments)
        tool_rows = [message for message in messages if message.get("role") == "tool"]
        assert {row["tool_call_id"] for row in tool_rows} == {f"read-{index}" for index in range(rounds)}
        assert all(("The instrument is ready." if arguments_valid else "Invalid tool arguments") in row["content"]
                   for row in tool_rows)
    assert all(note.read_text() == "The instrument is ready.\n" for note in notes)


@pytest.mark.parametrize("rejected", [None, "unknown", "malformed_json"])
def test_text_continuation_survives_without_an_accepted_tool_round(loop_agent, rejected):
    responses = [_mock_response(content="The observation", finish_reason="length")]
    if rejected == "unknown":
        responses.append(tool_response("{}", "rejected", name="unavailable_fixture_tool"))
    elif rejected == "malformed_json":
        responses.append(tool_response("{invalid}", "rejected"))
    responses.append(_mock_response(content="is complete."))
    loop_agent.client.chat.completions.create.side_effect = responses

    result = loop_agent.run_conversation("Report the observation.")

    assert result["completed"] is True
    assert result["final_response"] == "The observation\nis complete."
    assert len(loop_agent.client.chat.completions.create.call_args_list) == len(responses)
