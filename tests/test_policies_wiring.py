"""Tests for wire() (ACT-0): connect Pipeline decisions to an Actuator.

Uses scripted test-double policies (an established pattern in this codebase,
see test_stop.py's _AlwaysStopPolicy) to control exactly which Action each
pipeline.process() call returns, independently of any real detector.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pytest

from pathlayer.adapters.memory import InMemoryAdapter
from pathlayer.detectors.base import Signal
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies.actions import Action
from pathlayer.policies.actuators import Actuator
from pathlayer.policies.base import HysteresisState, Policy
from pathlayer.policies.wiring import wire
from pathlayer.postmortem import PostMortem


class _ScriptedPolicy(Policy):
    """Returns actions from a fixed script, one per call; repeats the last."""

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
    """Fake actuator recording every apply() call, for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[Action, PostMortem | None]] = []

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        self.calls.append((action, post_mortem))


class _RaisingActuator(Actuator):
    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        raise RuntimeError("boom")


def _events(n: int) -> list[ToolEvent]:
    return [ToolEvent(tool="read", args={"i": i}, ts=float(i)) for i in range(n)]


def test_wire_ignores_none_actions() -> None:
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.NONE, Action.NONE]))
    actuator = _RecordingActuator()
    sink = wire(pipeline, actuator)
    for event in _events(2):
        sink(event)
    assert actuator.calls == []


def test_wire_calls_actuator_on_transition_to_non_none() -> None:
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.NONE, Action.WARN]))
    actuator = _RecordingActuator()
    sink = wire(pipeline, actuator)
    for event in _events(2):
        sink(event)
    assert [call[0] for call in actuator.calls] == [Action.WARN]


def test_wire_calls_actuator_again_on_transition_between_non_none_tiers() -> None:
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.WARN, Action.ESCALATE]))
    actuator = _RecordingActuator()
    sink = wire(pipeline, actuator)
    for event in _events(2):
        sink(event)
    assert [call[0] for call in actuator.calls] == [Action.WARN, Action.ESCALATE]


def test_wire_does_not_recall_actuator_for_sustained_same_tier() -> None:
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.WARN, Action.WARN, Action.WARN]))
    actuator = _RecordingActuator()
    sink = wire(pipeline, actuator)
    for event in _events(3):
        sink(event)
    assert [call[0] for call in actuator.calls] == [Action.WARN]


def test_wire_reapplies_after_deescalating_through_none() -> None:
    # Regression: WARN -> NONE -> WARN must call apply() twice, not once --
    # a de-escalation to NONE must reset the transition tracking, otherwise
    # the second WARN looks "unchanged" from the last *applied* tier.
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.WARN, Action.NONE, Action.WARN]))
    actuator = _RecordingActuator()
    sink = wire(pipeline, actuator)
    for event in _events(3):
        sink(event)
    assert [call[0] for call in actuator.calls] == [Action.WARN, Action.WARN]


def test_wire_passes_post_mortem_only_on_stop() -> None:
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.WARN, Action.STOP]))
    actuator = _RecordingActuator()
    sink = wire(pipeline, actuator)
    for event in _events(2):
        sink(event)
    assert actuator.calls[0] == (Action.WARN, None)
    stop_action, post_mortem = actuator.calls[1]
    assert stop_action is Action.STOP
    assert post_mortem is not None
    assert post_mortem is pipeline.post_mortem


def test_wire_dedups_repeated_stop_after_pipeline_halts() -> None:
    # Pipeline itself short-circuits process() to STOP for every event after
    # the first STOP (see Pipeline._stopped), without re-invoking the policy.
    # wire() must still only have called apply() once.
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.STOP]))
    actuator = _RecordingActuator()
    sink = wire(pipeline, actuator)
    for event in _events(5):
        sink(event)
    assert [call[0] for call in actuator.calls] == [Action.STOP]
    assert len(pipeline.history) < 5


def test_wire_works_as_an_adapter_sink() -> None:
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.NONE, Action.STOP]))
    actuator = _RecordingActuator()
    adapter = InMemoryAdapter(wire(pipeline, actuator), _events(2))
    adapter.attach()
    assert [call[0] for call in actuator.calls] == [Action.STOP]


def test_wire_raises_on_dry_run_pipeline() -> None:
    pipeline = Pipeline([], dry_run=True)
    with pytest.raises(ValueError, match="dry_run"):
        wire(pipeline, _RecordingActuator())


def test_wire_is_fail_open_on_actuator_exception(caplog: pytest.LogCaptureFixture) -> None:
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.WARN]))
    sink = wire(pipeline, _RaisingActuator())
    with caplog.at_level(logging.ERROR):
        sink(ToolEvent(tool="read"))  # must not raise
    assert len(pipeline.history) == 1
    assert any("actuator.apply failed" in record.message for record in caplog.records)


def test_wire_default_logger_is_module_logger() -> None:
    # No explicit logger passed: wire() must still not raise on actuator failure.
    pipeline = Pipeline([], policy=_ScriptedPolicy([Action.WARN]))
    sink = wire(pipeline, _RaisingActuator())
    sink(ToolEvent(tool="read"))  # must not raise
    assert len(pipeline.history) == 1
