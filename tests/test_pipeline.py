"""Tests for the event pipeline (INF-3).

Pin the in-memory backbone: events flow through detectors and policy, history
is preserved, the adapter wires into the pipeline, and the run is deterministic.
With stub detectors and the default policy the pipeline stays inert (NONE) --
the loop -> STOP behavior is added in ARB-2/ACT-5.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest

from pathlayer import Pipeline
from pathlayer.adapters import InMemoryAdapter
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent
from pathlayer.policies import Action, Policy


def _events() -> list[ToolEvent]:
    return [
        ToolEvent(tool="read", args={"path": "a.py"}, ts=0.0),
        ToolEvent(tool="search", args={"q": "foo"}, ts=1.0),
        ToolEvent(tool="read", args={"path": "a.py"}, ts=2.0),
    ]


def _pipeline() -> Pipeline:
    return Pipeline([LoopDetector(), RetryDetector()])


def test_process_returns_action() -> None:
    action = _pipeline().process(ToolEvent(tool="read"))
    assert action is Action.NONE


class _CountingPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()
        self.fuse_calls = 0

    def fuse(self, signals: Sequence[Signal], evidence: Mapping[str, float] | None = None) -> float:
        self.fuse_calls += 1
        return super().fuse(signals, evidence or {})


def test_last_decision_is_lazy_cached_and_carries_event_timestamp() -> None:
    policy = _CountingPolicy()
    pipeline = Pipeline([LoopDetector()], policy)

    assert pipeline.process(ToolEvent(tool="read", ts=12.5)) is Action.NONE
    calls_after_process = policy.fuse_calls

    decision = pipeline.last_decision
    assert decision is not None
    assert decision.ts == 12.5
    assert decision.reason == ""
    assert policy.fuse_calls == calls_after_process + 1
    assert pipeline.last_decision is decision
    assert policy.fuse_calls == calls_after_process + 1


def test_restore_decisions_returns_every_replayed_tick_without_logs() -> None:
    events = _events()
    pipeline = _pipeline()

    decisions = pipeline.restore_decisions(events)

    assert [decision.index for decision in decisions] == [0, 1, 2]
    assert [decision.ts for decision in decisions] == [0.0, 1.0, 2.0]
    assert [decision.emitted_action for decision in decisions] == list(pipeline.actions)


def test_history_is_preserved_in_order() -> None:
    pipeline = _pipeline()
    events = _events()
    pipeline.run(events)
    assert list(pipeline.history) == events


def test_run_returns_one_action_per_event() -> None:
    pipeline = _pipeline()
    actions = pipeline.run(_events())
    assert actions == [Action.NONE, Action.NONE, Action.NONE]
    assert pipeline.actions == tuple(actions)


def test_inert_with_stub_detectors() -> None:
    # Stub detectors emit confidence 0.0; the default policy must not intervene.
    actions = _pipeline().run(_events())
    assert all(a is Action.NONE for a in actions)


def test_adapter_wires_into_pipeline() -> None:
    pipeline = _pipeline()
    events = _events()
    adapter = InMemoryAdapter(pipeline.sink, events)

    adapter.attach()

    assert list(pipeline.history) == events
    assert len(pipeline.actions) == len(events)


def test_pipeline_is_deterministic() -> None:
    events = _events()
    first = Pipeline([LoopDetector(), RetryDetector()]).run(events)
    second = Pipeline([LoopDetector(), RetryDetector()]).run(events)
    assert first == second


def test_defaults_to_a_policy() -> None:
    # No policy passed -> a default Policy is created and used.
    pipeline = Pipeline([LoopDetector()])
    assert pipeline.process(ToolEvent(tool="read")) is Action.NONE


def test_loop_trace_drives_decision_to_stop(
    trace: Callable[[str], list[ToolEvent]],
) -> None:
    # End-to-end at the decision level: real LoopDetector + default Policy on a
    # looping trace must eventually decide STOP (execution lands in ACT-5).
    pipeline = Pipeline([LoopDetector()])
    actions = pipeline.run(trace("loop_trivial.json"))
    assert Action.STOP in actions


def test_healthy_trace_never_stops() -> None:
    pipeline = Pipeline([LoopDetector()])
    healthy = [
        ToolEvent(tool="read", args={"path": "a.py"}),
        ToolEvent(tool="edit", args={"path": "a.py"}),
        ToolEvent(tool="run_tests", args={}),
        ToolEvent(tool="read", args={"path": "b.py"}),
        ToolEvent(tool="edit", args={"path": "b.py"}),
    ]
    actions = pipeline.run(healthy)
    assert all(a is Action.NONE for a in actions)


class _RecordingDetector(Detector):
    name = "recording"

    def __init__(self) -> None:
        self.seen_types: list[type] = []

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        self.seen_types.append(type(history))
        return Signal(self.name, confidence=0.0)


def test_detectors_receive_immutable_history() -> None:
    # A snapshot is passed so a buggy detector cannot mutate pipeline state.
    detector = _RecordingDetector()
    Pipeline([detector]).run(_events())
    assert detector.seen_types
    assert all(seen is tuple for seen in detector.seen_types)


class _BoomDetector(Detector):
    name = "boom"

    def __init__(self, fail_on: int) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("boom")
        return Signal(self.name, confidence=0.0)


def test_exception_rolls_back_event_and_keeps_state_in_sync() -> None:
    pipeline = Pipeline([_BoomDetector(fail_on=2)])
    assert pipeline.process(ToolEvent(tool="a")) is Action.NONE

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.process(ToolEvent(tool="b"))

    # The failed event is rolled back; history and actions stay aligned.
    assert len(pipeline.history) == 1
    assert len(pipeline.actions) == 1
