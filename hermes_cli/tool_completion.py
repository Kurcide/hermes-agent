"""Typed normal-turn completion returned by a trusted ``post_tool_batch`` hook."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FinishTurn:
    """Finish with runtime-authored text for the current successful tool call.

    The plugin must require an explicit terminal handoff of the current request;
    a successful background submission alone does not establish that unrelated
    foreground obligations are complete. Return this from ``post_tool_batch``,
    never from a tool handler. Hermes validates the current single-call receipt
    and persists the text before normal finalization. This is neither model
    output nor evidence of downstream delivery or background completion.
    """

    text: str
    tool_call_id: str
