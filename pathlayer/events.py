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

    Contract (PERF-2): once an event has been handed to
    `Pipeline.process`/`sink`, treat it -- and its `args`/`result`/`metadata`
    mappings -- as immutable, and treat the stream as append-only (no removal,
    no reordering of already-processed events). Nothing enforces this at
    runtime (no deep freeze/copy, to keep capture allocation-free on the hot
    path); it is a documented caller contract instead. The incremental fold
    path (`Detector.step`/`ProgressAnchor.step`, PERF-2) extracts the scalars
    it needs from an event once, at fold time, and never re-reads it -- so a
    retroactive mutation of an already-processed event silently does *not*
    propagate to already-computed incremental state, unlike the batch path
    (`evaluate`/`measure`), which re-reads every event fresh on each call and
    therefore *would* see the mutation. Violating the contract is undefined
    behavior (see
    `tests/test_perf2_incremental.py::test_mutating_a_past_event_is_undefined_behavior`
    for the specific divergence this causes), not a supported use case.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    cost_usd: float = 0.0
    ts: float = 0.0
    kind: EventKind = EventKind.TOOL_CALL
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
