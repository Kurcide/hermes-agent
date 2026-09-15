"""The normal-finish decision at a completed, single-tool batch boundary."""

from contextlib import nullcontext
from copy import deepcopy
import logging

from agent.message_sanitization import _sanitize_surrogates
from hermes_cli.tool_completion import FinishTurn

logger = logging.getLogger(__name__)


def _has_pending_input(agent):
    if agent._interrupt_requested or agent._has_pending_redirect():
        return True
    with getattr(agent, "_pending_steer_lock", None) or nullcontext():
        return bool(getattr(agent, "_pending_steer", None))


def get_tool_turn_completion(agent, *, tool_call, tool_result, effective_task_id):
    """Return a validated ``(text, provenance)`` or None, without mutating history.

    The caller supplies only the current executor-confirmed, successfully
    persisted result. Historical rows and tool-produced directive-shaped JSON
    never enter the completion protocol. Hooks cannot settle mixed batches.
    """
    if _has_pending_input(agent):
        return None
    call_id = tool_call.get("id")
    if not call_id or tool_result.get("tool_call_id") != call_id:
        return None
    function = tool_call["function"]
    provenance = {
        "type": "tool_handoff",
        "session_id": agent.session_id or "",
        "task_id": effective_task_id,
        "turn_id": agent._current_turn_id,
        "api_request_id": agent._current_api_request_id,
        "tool_name": function["name"],
        "tool_call_id": call_id,
    }
    try:
        from hermes_cli.lifecycle import invoke_hook

        decisions = invoke_hook(
            "post_tool_batch",
            **{key: value for key, value in provenance.items() if key != "type"},
            platform=getattr(agent, "platform", None) or "",
            tool_arguments=deepcopy(function["arguments"]),
            tool_result=deepcopy(tool_result["content"]),
        )
    except Exception:
        logger.warning("post_tool_batch hook failed", exc_info=True)
        return None
    # A hook may take time; input arriving during it keeps its normal loop path.
    if _has_pending_input(agent):
        return None
    for decision in decisions:
        if (
            isinstance(decision, FinishTurn)
            and decision.tool_call_id == call_id
            and isinstance(decision.text, str)
            and decision.text.strip()
        ):
            return _sanitize_surrogates(decision.text), provenance
    return None
