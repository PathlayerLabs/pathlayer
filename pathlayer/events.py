"""Normalized event schema.

Every adapter (OpenClaw, LangGraph, CrewAI, ...) translates its framework's
hooks into these events so the rest of the pipeline stays framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    TOOL_CALL = "tool_call"
    # A pure LLM call (no tool). Carries token cost in ``cost_usd`` so
    # BudgetDetector can account for spend that happens in the model, not in
    # tools. Action detectors (loop/retry/no_progress) ignore it via their
    # ``event_kinds`` filter so a reasoning agent is not mistaken for a loop.
    LLM_CALL = "llm_call"
    # Resource / lifecycle events (class iii incidents). Detect-only in V1.
    SUBAGENT_SPAWN = "subagent_spawn"
    RUN_RESTART = "run_restart"


@dataclass(slots=True)
class ToolEvent:
    """A single tool call observed from the agent.

    `args` and `result` are raw here; canonicalization (stripping volatile
    metadata such as timestamps, UUIDs, PIDs, pagination cursors) happens in
    the extraction layer (`pathlayer.detectors`), not at capture time.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    cost_usd: float = 0.0
    ts: float = 0.0
    kind: EventKind = EventKind.TOOL_CALL
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
