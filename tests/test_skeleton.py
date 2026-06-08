"""Skeleton wiring tests: every detector, the adapter contract, the action
ladder and the default policy config behave as the architecture expects.

These pin the V1 contracts so future implementations can't silently break
them (deterministic detectors, passive adapters, graded action ordering).
"""

from __future__ import annotations

import pytest

from pathlayer.adapters import Adapter
from pathlayer.adapters.base import EventSink
from pathlayer.adapters.openclaw import OpenClawAdapter
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.resource import ResourceDetector
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.policies import Action, Policy
from pathlayer.policies.base import PolicyConfig

ALL_DETECTORS = [
    (LoopDetector, "loop"),
    (RetryDetector, "retry"),
    (BudgetDetector, "budget"),
    (NoProgressDetector, "no_progress"),
    (ResourceDetector, "resource"),
]


@pytest.mark.parametrize(("detector_cls", "expected_name"), ALL_DETECTORS)
def test_detector_contract(detector_cls: type[Detector], expected_name: str) -> None:
    detector = detector_cls()
    assert isinstance(detector, Detector)
    assert detector.name == expected_name

    signal = detector.evaluate([])
    assert isinstance(signal, Signal)
    assert signal.name == expected_name
    assert 0.0 <= signal.confidence <= 1.0


@pytest.mark.parametrize(("detector_cls", "_name"), ALL_DETECTORS)
def test_detector_is_deterministic(detector_cls: type[Detector], _name: str) -> None:
    history = [ToolEvent(tool="read", args={"path": "a.py"}, ts=float(i)) for i in range(5)]
    detector = detector_cls()
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
    assert config.max_same_error == 4
    assert config.max_tool_retries == 3
    assert config.max_cost_usd == 5.0
    assert config.no_progress_steps == 5


def test_policy_uses_provided_config() -> None:
    policy = Policy(config=PolicyConfig(max_cost_usd=1.0))
    assert policy.config.max_cost_usd == 1.0
    assert policy.decide([Signal("loop", confidence=0.9)]) is Action.NONE


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
