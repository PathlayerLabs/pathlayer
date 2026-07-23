"""Tests for the ACT-1 actuation context: Pipeline production + Actuator plumbing.

Framework-agnostic (no LangGraph). Covers:

* the immutable :class:`ActuationContext` the ``Pipeline`` builds for every
  emitted non-NONE action (fields, event snapshot, post-mortem identity);
* that no context is built for NONE or in dry-run;
* the retrocompatible ``Actuator.apply_with_context`` default (delegates to
  ``apply``) and that ``wire()`` routes through it, reaching a context-aware
  actuator with the full context.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pytest

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies.actions import Action
from pathlayer.policies.actuation import ActuationContext, EventSnapshot, ProgressSummary
from pathlayer.policies.actuators import Actuator
from pathlayer.policies.base import HysteresisState, Policy
from pathlayer.policies.wiring import wire
from pathlayer.postmortem import PostMortem


class _FixedLoopDetector(Detector):
    """Emits one constant loop signal, so ``fuse`` is deterministic."""

    name = "loop"

    def __init__(self, confidence: float = 0.5) -> None:
        self._confidence = confidence

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        return Signal(name="loop", confidence=self._confidence, reason="repeated sequence")


class _ConcurrencyProbeDetector(Detector):
    """Records whether two ``evaluate`` calls overlap inside ``wire``."""

    name = "probe"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.01)  # release the GIL so an unlocked sink would overlap
        with self._lock:
            self._active -= 1
        return Signal(name=self.name, confidence=0.0)


class _ScriptedPolicy(Policy):
    """Returns actions from a fixed script; keeps the real ``fuse``/reasoning."""

    def __init__(self, actions: Sequence[Action]) -> None:
        super().__init__()
        self._actions = list(actions)
        self._calls = 0

    def step(
        self,
        state: HysteresisState,
        signals: Sequence[Signal],
        evidence: Mapping[str, float] = MappingProxyType({}),
    ) -> tuple[HysteresisState, Action]:
        index = min(self._calls, len(self._actions) - 1)
        self._calls += 1
        action = self._actions[index]
        return HysteresisState(action), action


class _RecordingActuator(Actuator):
    """Legacy actuator: implements only ``apply`` (the pre-ACT-1 contract)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Action, PostMortem | None]] = []

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        self.calls.append((action, post_mortem))


class _ContextAwareActuator(Actuator):
    """ACT-1 actuator: overrides ``apply_with_context`` to capture full context."""

    def __init__(self) -> None:
        self.contexts: list[ActuationContext] = []

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:  # pragma: no cover
        raise AssertionError("wire() must call apply_with_context, not apply")

    def apply_with_context(self, context: ActuationContext) -> None:
        self.contexts.append(context)


def test_pipeline_builds_context_on_warn() -> None:
    pipeline = Pipeline([_FixedLoopDetector(0.5)], _ScriptedPolicy([Action.WARN]))
    action = pipeline.process(ToolEvent(tool="search", ts=1.0))
    assert action is Action.WARN

    context = pipeline.actuation_context
    assert context is not None
    assert context.action is Action.WARN
    assert context.fused == pytest.approx(0.5)
    assert context.reason  # non-empty, sourced from the dominant signal
    assert [s.name for s in context.signals] == ["loop"]
    assert context.event == EventSnapshot(
        tool="search", kind=EventKind.TOOL_CALL, ts=1.0, index=0, cost_usd=0.0
    )
    assert context.post_mortem is None


def test_context_is_none_before_any_non_none_action() -> None:
    pipeline = Pipeline([_FixedLoopDetector(0.0)], _ScriptedPolicy([Action.NONE]))
    assert pipeline.process(ToolEvent(tool="x")) is Action.NONE
    assert pipeline.actuation_context is None


def test_context_on_stop_shares_post_mortem_identity() -> None:
    pipeline = Pipeline([_FixedLoopDetector(1.0)], _ScriptedPolicy([Action.STOP]))
    pipeline.process(ToolEvent(tool="x", ts=2.0))

    context = pipeline.actuation_context
    assert context is not None
    assert context.action is Action.STOP
    # Same object as the post-mortem, and the reason is threaded from it.
    assert context.post_mortem is pipeline.post_mortem
    assert pipeline.post_mortem is not None
    assert context.reason == pipeline.post_mortem.reason


def test_dry_run_never_builds_context() -> None:
    pipeline = Pipeline([_FixedLoopDetector(1.0)], _ScriptedPolicy([Action.STOP]), dry_run=True)
    assert pipeline.process(ToolEvent(tool="x")) is Action.NONE
    assert pipeline.actuation_context is None


def test_apply_with_context_default_delegates_to_apply() -> None:
    actuator = _RecordingActuator()
    post_mortem = PostMortem(
        action=Action.STOP,
        reason="loop: repeated sequence",
        stopped_at_index=0,
        events_processed=1,
        signals=(),
        recent_history=(),
    )
    context = ActuationContext(
        action=Action.STOP,
        fused=0.95,
        reason="loop: repeated sequence",
        signals=(),
        event=EventSnapshot(tool="x", kind=EventKind.TOOL_CALL, ts=0.0, index=0, cost_usd=0.0),
        post_mortem=post_mortem,
    )
    actuator.apply_with_context(context)
    assert actuator.calls == [(Action.STOP, post_mortem)]


def test_wire_routes_context_to_context_aware_actuator() -> None:
    pipeline = Pipeline([_FixedLoopDetector(0.5)], _ScriptedPolicy([Action.NONE, Action.WARN]))
    actuator = _ContextAwareActuator()
    sink = wire(pipeline, actuator)
    sink(ToolEvent(tool="a", ts=0.0))  # NONE -> no actuation
    sink(ToolEvent(tool="b", ts=1.0))  # WARN -> actuation with context

    assert len(actuator.contexts) == 1
    context = actuator.contexts[0]
    assert context.action is Action.WARN
    assert context.fused == pytest.approx(0.5)
    assert context.event.tool == "b"
    assert context.event.index == 1


def test_wire_serializes_concurrent_event_producers() -> None:
    # ACT-2 sends both adapter-captured events and synthetic gate refusals into
    # one sink. Pipeline is single-threaded by contract, so wire() must make that
    # shared sink the serialization boundary for both producers.
    detector = _ConcurrencyProbeDetector()
    pipeline = Pipeline([detector], Policy())
    sink = wire(pipeline, _RecordingActuator())
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def emit(index: int) -> None:
        try:
            barrier.wait()
            sink(ToolEvent(tool=f"tool_{index}"))
        except BaseException as exc:  # pragma: no cover - assertion surface
            errors.append(exc)

    threads = [threading.Thread(target=emit, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert detector.max_active == 1


# --------------------------------------------------------------------------- #
# ACT-2: ProgressSummary validation (the closed 3-field contract the gate uses).
# --------------------------------------------------------------------------- #


def test_progress_summary_parses_valid_args_and_strips() -> None:
    summary = ProgressSummary.from_tool_args(
        {"completed": "  did X ", "blocker": "Y", "next_action": "Z", "extra": "ignored"}
    )
    assert summary == ProgressSummary(completed="did X", blocker="Y", next_action="Z")


@pytest.mark.parametrize(
    "args",
    [
        {"completed": "a", "blocker": "b"},  # missing next_action
        {"completed": "a", "blocker": "b", "next_action": ""},  # blank
        {"completed": "a", "blocker": "b", "next_action": "   "},  # whitespace-only
        {"completed": "a", "blocker": "b", "next_action": 3},  # wrong type
        {},  # empty
        "not a mapping",  # non-mapping
        None,
    ],
)
def test_progress_summary_rejects_incomplete_args(args: object) -> None:
    # Fail-open contract: never raises, returns None so the gate stays closed.
    assert ProgressSummary.from_tool_args(args) is None
