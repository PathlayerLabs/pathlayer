"""Skeleton wiring tests: every detector, the adapter contract, the action
ladder and the default policy config behave as the architecture expects.

These pin the V1 contracts so future implementations can't silently break
them (deterministic detectors, passive adapters, graded action ordering).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from pathlayer.adapters import Adapter
from pathlayer.adapters.base import EventSink
from pathlayer.adapters.openclaw import OpenClawAdapter
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.resource import ResourceDetector
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.policies import Action, Policy
from pathlayer.policies.base import PolicyConfig

# Factories so detectors that need wiring (NoProgressDetector takes anchors) are
# built ready-to-use; the contract is identical across all of them.
ALL_DETECTORS: list[tuple[Callable[[], Detector], str]] = [
    (LoopDetector, "loop"),
    (RetryDetector, "retry"),
    (BudgetDetector, "budget"),
    (lambda: NoProgressDetector(coding_anchors()), "no_progress"),
    (ResourceDetector, "resource"),
]


@pytest.mark.parametrize(
    ("make_detector", "expected_name"), ALL_DETECTORS, ids=[name for _, name in ALL_DETECTORS]
)
def test_detector_contract(make_detector: Callable[[], Detector], expected_name: str) -> None:
    detector = make_detector()
    assert isinstance(detector, Detector)
    assert detector.name == expected_name

    signal = detector.evaluate([])
    assert isinstance(signal, Signal)
    assert signal.name == expected_name
    assert 0.0 <= signal.confidence <= 1.0


@pytest.mark.parametrize(
    ("make_detector", "_name"), ALL_DETECTORS, ids=[name for _, name in ALL_DETECTORS]
)
def test_detector_is_deterministic(make_detector: Callable[[], Detector], _name: str) -> None:
    history = [ToolEvent(tool="read", args={"path": "a.py"}, ts=float(i)) for i in range(5)]
    detector = make_detector()
    first = detector.evaluate(history)
    second = detector.evaluate(history)
    assert first == second


def test_action_ladder_is_ordered() -> None:
    assert (
        Action.NONE
        < Action.WARN
        < Action.SUMMARIZE
        < Action.FALLBACK
        < Action.ESCALATE
        < Action.STOP
    )


def test_default_policy_config_matches_architecture() -> None:
    config = PolicyConfig()
    assert config.stop_threshold == 0.9
    assert config.advisory_signals == frozenset()
    assert config.weights == {}


def test_policy_uses_provided_config() -> None:
    policy = Policy(config=PolicyConfig(stop_threshold=0.5))
    assert policy.config.stop_threshold == 0.5
    # A low-confidence signal must not trigger any action.
    assert policy.decide([Signal("loop", confidence=0.1)]) is Action.NONE


def test_tool_event_defaults() -> None:
    event = ToolEvent(tool="search")
    assert event.tool == "search"
    assert event.args == {}
    assert event.cost_usd == 0.0
    assert event.kind is EventKind.TOOL_CALL


def test_adapter_is_passive_observer() -> None:
    captured: list[ToolEvent] = []
    sink: EventSink = captured.append

    class _DummyAdapter(Adapter):
        def attach(self) -> None:
            self.emit(ToolEvent(tool="probe"))

        def detach(self) -> None:
            pass

    adapter = _DummyAdapter(sink)
    adapter.attach()
    adapter.detach()

    assert len(captured) == 1
    assert captured[0].tool == "probe"


def test_openclaw_adapter_not_wired_yet() -> None:
    adapter = OpenClawAdapter(lambda _event: None)
    with pytest.raises(NotImplementedError):
        adapter.attach()
