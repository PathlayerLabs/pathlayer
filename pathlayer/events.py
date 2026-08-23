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
    # A structured progress summary the agent emitted through the Pathlayer
    # summary tool (ACT-2, ``PROGRESS_SUMMARY_TOOL``). The three summary fields
    # (``completed``/``blocker``/``next_action``) travel in ``args``. Action
    # detectors (loop/retry/no_progress) ignore it via their ``event_kinds``
    # filter, so forcing a summary never adds a fake stagnation step nor fools
    # a repetition detector; ``cost_usd`` stays 0.0 so BudgetDetector is
    # untouched too. Capture-only, exactly like every other event kind.
    PROGRESS_SUMMARY = "progress_summary"
    # Resource / lifecycle events (class iii incidents). Detect-only in V1.
    SUBAGENT_SPAWN = "subagent_spawn"
    RUN_RESTART = "run_restart"
    # Event-sourced runtime lifecycle used by RES-2. RUN_START carries the
    # durable Unix start time in metadata; RUN_TIMEOUT is injected exactly once
    # by a run-aware runtime sink when the configured hard limit expires.
    RUN_START = "run_start"
    RUN_TIMEOUT = "run_timeout"


# Stable name of the Pathlayer progress-summary tool (ACT-2). The adapter keys
# on it to classify a captured tool call as ``EventKind.PROGRESS_SUMMARY``, and
# the LangGraph gate keys on it to let a valid summary through. A constant (not
# a string literal scattered across layers) so capture and enforcement can never
# drift apart.
PROGRESS_SUMMARY_TOOL = "pathlayer_progress_summary"

# Synthetic, framework-agnostic action emitted by the ACT-2 gate whenever an
# agent tries to run another tool while a progress summary is still owed. It is
# deliberately a normal ``TOOL_CALL`` (not a new EventKind): the blocked attempt
# is real no-progress evidence that Loop/Retry/NoProgress must observe so Policy
# can escalate to STOP instead of the graph re-prompting forever.
SUMMARY_BLOCKED_TOOL = "pathlayer_summary_blocked"

# Stable names for the synthetic lifecycle events. Keeping these centralized
# makes persisted traces portable across adapters and replay implementations.
RUN_START_TOOL = "pathlayer_run_start"
RUN_TIMEOUT_TOOL = "pathlayer_run_timeout"


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
