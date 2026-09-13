"""Iteration-budget summary requests, with the normal provider-specific payload shapes."""

from __future__ import annotations

import contextlib
import logging
import re
import uuid

from agent.chat_completion_helpers import (
    _merge_nous_portal_messages_extra_body, _provider_preferences_for_agent,
)
from agent.message_metadata import append_message
from agent.turn_context import substitute_api_content

logger = logging.getLogger(__name__)


# Keys outside the Chat Completions schema that strict gateways (Fireworks-backed OpenCode
# Go, Mistral, Moonshot/Kimi) reject with 422. The transport's convert_messages() drops them
# in the main loop; the summary path calls chat.completions.create() directly, so mirror it.
_SUMMARY_FOREIGN_MESSAGE_KEYS = ("reasoning", "finish_reason", "tool_name", "codex_reasoning_items",
    "codex_message_items", "timestamp", "platform_message_id")
_EMPTY_SUMMARY_RESPONSE = "I reached the iteration limit and couldn't generate a summary."


def _iteration_summary_api_messages(agent, messages: list) -> list:
    """Wire-ready messages for the summary call, mirroring the main loop's api_messages build
    (sidecar substitution, tool-call repair, thinking-only drop, underscore-key sweep)."""
    needs_sanitize = agent._should_sanitize_tool_calls()
    sanitize_model = agent.model
    if needs_sanitize and agent.provider == "moa":
        # MoA: agent.model is the virtual preset; use the real aggregator so Gemini keeps thought_signature.
        agg_slot = getattr(getattr(agent, "client", None), "last_aggregator_slot", None)
        sanitize_model = (agg_slot or {}).get("model") or sanitize_model
    api_messages = []
    for msg in messages:
        api_msg = msg.copy()
        agent._copy_reasoning_content_for_api(msg, api_msg)
        for key in _SUMMARY_FOREIGN_MESSAGE_KEYS:
            api_msg.pop(key, None)
        # Mirror of the transport's role-qualified strip: ``name`` is
        # schema-foreign on tool results only (strict providers reject with
        # "contains item with unknown key name"); it stays on user/assistant.
        if api_msg.get("role") == "tool":
            api_msg.pop("name", None)
        # api_content holds the exact bytes the main loop sent; substituting (not popping)
        # keeps the summary's prefix identical instead of re-prefilling the largest context.
        # Strict OpenAI-compatible gateways (Fireworks-backed OpenCode Go, Mistral, Moonshot/Kimi) reject
        # any message key outside the Chat Completions schema. The main loop drops these via
        # ChatCompletionsTransport.convert_messages(), but the summary path hand-builds messages and calls
        # chat.completions.create() directly, bypassing the transport — so mirror that sanitization here:
        # tool_name (SQLite FTS bookkeeping), the codex_* reasoning carriers, timestamp (preserved on
        # gateway user replay entries for the stale-confirmation expiry check — #47868 rejection class), and
        # every Hermes-internal underscore-prefixed scaffolding key.
        substitute_api_content(api_msg)
        if needs_sanitize:
            agent._sanitize_tool_calls_for_strict_api(api_msg, model=sanitize_model)
        api_messages.append(api_msg)

    effective_system = agent._cached_system_prompt or ""
    if agent.ephemeral_system_prompt:
        effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
    if effective_system:
        api_messages = [{"role": "system", "content": effective_system}] + api_messages
    for idx, pfm in enumerate(agent.prefill_messages or ()):
        api_messages.insert((1 if effective_system else 0) + idx, pfm.copy())

    # Compression/resume can orphan a tool result whose parent tool_call was summarized away.
    api_messages = agent._sanitize_api_messages(api_messages)
    # Same send-path vision eviction as the main loop (#89296).
    from agent.context_compressor import evict_stale_outbound_tool_images
    evict_stale_outbound_tool_images(api_messages)
    # Thinking-only assistant turns 400 on Anthropic-family providers; _thinking_prefill must
    # survive until here so the drop pass recognizes stubs after reasoning is stripped.
    api_messages = agent._drop_thinking_only_and_merge_users(api_messages)
    for api_msg in api_messages:  # underscore scaffolding: the transport's sweeper is bypassed here
        if isinstance(api_msg, dict):
            for internal_key in [k for k in api_msg if isinstance(k, str) and k.startswith("_")]:
                del api_msg[internal_key]
    return api_messages


def _summary_request(agent, request: dict, *, api_request_id: str, api_call_count: int, retry_count: int) -> dict:
    """Use the bound turn's policy without treating the runtime nudge as new input."""
    from hermes_cli.middleware import apply_llm_request_middleware

    return apply_llm_request_middleware(
        request, task_id=getattr(agent, "_current_task_id", "") or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "", api_request_id=api_request_id,
        session_id=agent.session_id or "", platform=agent.platform or "", model=agent.model,
        provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode,
        api_call_count=api_call_count, retry_count=retry_count, call_role="iteration_summary",
    ).payload


def _managed_summary_call(agent, api_request_id: str, request, callback, *, api_call_count: int, retry_count: int):
    from agent import relay_llm
    request = _summary_request(agent, request, api_request_id=api_request_id,
        api_call_count=api_call_count, retry_count=retry_count)
    return relay_llm.execute_current(
        request, callback,
        name=str(getattr(agent, "provider", "") or "provider"), model_name=str(getattr(agent, "model", "") or ""),
        metadata={"api_mode": str(getattr(agent, "api_mode", "") or "chat_completions"),
            "api_request_id": api_request_id, "call_role": "iteration_summary", "retry_count": retry_count},
        defer_logical_completion=True,
    )


def _iteration_summary_chat_kwargs(agent, api_messages: list) -> dict:
    """chat.completions.create kwargs for the summary, mirroring ChatCompletionsTransport.build_kwargs()."""
    try:
        from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE as _OMIT_TEMP
    except Exception:
        _fixed_temperature_for_model = _OMIT_TEMP = None
    raw_temp = _fixed_temperature_for_model(agent.model, agent.base_url) if _fixed_temperature_for_model is not None else None
    temperature = None if raw_temp is _OMIT_TEMP else raw_temp
    provider_name = (agent.provider or "").strip().lower()
    # LM Studio uses top-level `reasoning_effort` (not extra_body.reasoning).
    is_lmstudio = provider_name == "lmstudio" and agent._supports_reasoning_extra_body()
    lm_reasoning_effort = agent._resolve_lmstudio_summary_reasoning_effort() if is_lmstudio else None

    extra_body = {}
    if not is_lmstudio and agent._supports_reasoning_extra_body():
        extra_body["reasoning"] = agent.reasoning_config if agent.reasoning_config is not None else {"enabled": True, "effort": "medium"}
    if "nousresearch" in agent._base_url_lower:
        from agent.portal_tags import nous_portal_tags
        extra_body["tags"] = nous_portal_tags()

    summary_kwargs = {"model": agent.model, "messages": api_messages}
    if temperature is not None:
        summary_kwargs["temperature"] = temperature
    if agent.max_tokens is not None:
        summary_kwargs.update(agent._max_tokens_param(agent.max_tokens))
    if lm_reasoning_effort is not None:
        summary_kwargs["reasoning_effort"] = lm_reasoning_effort

    # Merge the profile's canonical body even when routing is unset (e.g. required Portal tags).
    provider_preferences = _provider_preferences_for_agent(agent)
    profile_extra_body = {}
    with contextlib.suppress(Exception):
        from providers import get_provider_profile
        provider_profile = get_provider_profile(agent.provider)
        if provider_profile is not None:
            profile_extra_body = provider_profile.build_extra_body(
                session_id=getattr(agent, "session_id", None), provider_preferences=provider_preferences or None,
                model=agent.model, base_url=agent.base_url, reasoning_config=agent.reasoning_config)
    if profile_extra_body:
        extra_body.update(profile_extra_body)

    def _is_openrouter() -> bool:
        return provider_name == "openrouter" or agent._is_openrouter_url()

    if provider_preferences and "provider" not in profile_extra_body and _is_openrouter():
        extra_body["provider"] = provider_preferences
    # Pareto Code router plugin — model-gated, same shape as the main-loop emission.
    _score = agent.openrouter_min_coding_score
    if agent.model == "openrouter/pareto-code" and _is_openrouter() and _score is not None and _score != "":
        with contextlib.suppress(TypeError, ValueError):
            _ps = float(_score)
            if 0.0 <= _ps <= 1.0:
                extra_body["plugins"] = [{"id": "pareto-router", "min_coding_score": _ps}]
    if extra_body:
        summary_kwargs["extra_body"] = extra_body
    return summary_kwargs


def _summary_text(agent, response, **normalize_kwargs) -> str:
    return (agent._get_transport().normalize_response(response, **normalize_kwargs).content or "").strip()


def _codex_summary_attempt(agent, api_messages: list, api_request_id: str, api_call_count: int):
    def _attempt(retry_count: int) -> str:
        codex_kwargs = agent._build_api_kwargs(api_messages)
        codex_kwargs.pop("tools", None)
        codex_kwargs = _summary_request(agent, codex_kwargs, api_request_id=api_request_id,
            api_call_count=api_call_count, retry_count=retry_count)
        return _summary_text(agent, agent._run_codex_stream(codex_kwargs))
    return _attempt


def _anthropic_summary_attempt(agent, api_messages: list, api_request_id: str, api_call_count: int):
    def _attempt(retry_count: int) -> str:
        ant_kw = agent._get_transport().build_kwargs(
            model=agent.model, messages=api_messages, tools=None, max_tokens=agent.max_tokens,
            reasoning_config=agent.reasoning_config, is_oauth=agent._is_anthropic_oauth,
            preserve_dots=agent._anthropic_preserve_dots(), base_url=getattr(agent, "_anthropic_base_url", None))
        ant_kw = _merge_nous_portal_messages_extra_body(agent, ant_kw)
        response = _managed_summary_call(agent, api_request_id, ant_kw, agent._anthropic_messages_create, api_call_count=api_call_count, retry_count=retry_count)
        return _summary_text(agent, response, strip_tool_prefix=agent._is_anthropic_oauth)
    return _attempt


def _chat_summary_attempt(agent, api_messages: list, api_request_id: str, api_call_count: int):
    summary_kwargs = _iteration_summary_chat_kwargs(agent, api_messages)

    def _attempt(retry_count: int) -> str:
        summary_client = agent._ensure_primary_openai_client(reason="iteration_limit_summary_retry" if retry_count else "iteration_limit_summary")
        response = _managed_summary_call(
            agent, api_request_id, summary_kwargs, lambda request: summary_client.chat.completions.create(**request), api_call_count=api_call_count, retry_count=retry_count)
        return _summary_text(agent, response)
    return _attempt


_SUMMARY_ATTEMPT_BUILDERS = {"codex_responses": _codex_summary_attempt, "anthropic_messages": _anthropic_summary_attempt}


def handle_max_iterations(agent, messages: list, api_call_count: int) -> str:
    """Request a summary when max iterations are reached. Returns the final response text."""
    warning = f"⚠️  Reached maximum iterations ({agent.max_iterations}). Requesting summary..."
    if getattr(agent, "suppress_status_output", False):
        # Strict machine-readable mode (-Q, oneshot): keep diagnostics off stdout. quiet_mode is
        # NOT the gate — the interactive CLI runs quiet_mode=True by default and must see this.
        # Strict machine-readable mode (hermes chat -Q, oneshot, background review): keep diagnostics out of
        # stdout so wrappers receive only the final assistant content (#93220 class).
        logger.warning(warning)
    else:
        agent._safe_print(warning)

    summary_api_request_id = f"iteration-summary:{uuid.uuid4()}"
    summary_call_outcome = "failed"

    # Shared constant so compaction recognizers can identify this runtime nudge by its stable
    # content after SessionDB projection strips metadata flags.
    from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST
    append_message(messages, {"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST})

    try:
        api_messages = _iteration_summary_api_messages(agent, messages)
        build_attempt = _SUMMARY_ATTEMPT_BUILDERS.get(agent.api_mode, _chat_summary_attempt)
        attempt = build_attempt(agent, api_messages, summary_api_request_id, api_call_count)

        # One retry on an empty summary; a summary empty once its <think> block is stripped is NOT retried.
        final_response = _EMPTY_SUMMARY_RESPONSE
        for retry_count in (0, 1):
            text = attempt(retry_count)
            if not text:
                continue
            if "<think>" in text:
                text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
            if text:
                summary_call_outcome = "success"
                append_message(messages, {"role": "assistant", "content": text})
                final_response = text
            break

    except Exception as e:
        logger.warning("Failed to get summary response: %s", e)
        final_response = f"I reached the maximum iterations ({agent.max_iterations}) but couldn't summarize. Error: {str(e)}"
    finally:
        from agent import relay_llm
        relay_llm.complete_logical_call(summary_api_request_id, outcome=summary_call_outcome)

    return final_response


