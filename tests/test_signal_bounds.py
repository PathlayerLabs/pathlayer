"""Contract: every built-in detector keeps ``Signal.confidence`` in [0, 1].

The 2026-07-01 audit (finding M-2) flagged that ``Signal.confidence`` is never validated. We
enforce the invariant with tests rather than a runtime ``raise`` inside the
detection loop: a range violation is a *detector bug*, and Pathlayer must never
crash the agent it observes. The companion defensive guard lives in
``Policy.fuse`` (see ``tests/test_policy_arbitration.py``), which clamps stray
custom-detector values so the fused score stays in [0, 1].

These tests exercise each built-in detector across empty, repetitive, long and
extreme histories and assert the emitted confidence never leaves [0, 1].
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from pathlayer.detectors.base import Detector
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent


def _failing(tool: str, error: str = "boom") -> ToolEvent:
    return ToolEvent(tool=tool, result={"error": error})


def _stalling_test(passed: int = 1, failed: int = 1) -> ToolEvent:
    # Flat, non-green test result: level = passed - failed stays constant and
    # is never satisfied (failed != 0), so the no-progress CUSUM accumulates.
    return ToolEvent(tool="pytest", result={"passed": passed, "failed": failed})


def _histories() -> dict[str, list[ToolEvent]]:
    """Varied histories, incl. extremes that push each detector to saturation."""
    return {
        "empty": [],
        "single": [ToolEvent(tool="read", args={"path": "a"})],
        "repetitive": [ToolEvent(tool="read", args={"path": "a"}) for _ in range(50)],
        "varied": [ToolEvent(tool=f"tool_{i}", args={"i": i}) for i in range(50)],
        "retries": [_failing("search") for _ in range(50)],
        "over_budget": [ToolEvent(tool="llm", cost_usd=1000.0) for _ in range(50)],
        "stalling_tests": [_stalling_test() for _ in range(50)],
    }


def _detectors() -> dict[str, Detector]:
    return {
        "loop": LoopDetector(),
        "retry": RetryDetector(),
        "budget": BudgetDetector(),
        "no_progress": NoProgressDetector(coding_anchors()),
    }


# Static name tuples so parametrization does not construct detectors or build the
# (large) histories at collection time -- the fixtures are built inside the test.
_DETECTOR_NAMES = ("loop", "retry", "budget", "no_progress")
_HISTORY_NAMES = (
    "empty",
    "single",
    "repetitive",
    "varied",
    "retries",
    "over_budget",
    "stalling_tests",
)


@pytest.mark.parametrize("detector_name", _DETECTOR_NAMES)
@pytest.mark.parametrize("history_name", _HISTORY_NAMES)
def test_detector_confidence_stays_in_unit_interval(detector_name: str, history_name: str) -> None:
    detector = _detectors()[detector_name]
    history = _histories()[history_name]
    signal = detector.evaluate(history)
    assert 0.0 <= signal.confidence <= 1.0


def test_saturating_histories_actually_reach_the_upper_bound() -> None:
    # Guard against a vacuous test: confirm the "extreme" histories really do
    # push detectors to (clamped) 1.0, so the [0, 1] assertion above exercises
    # the upper edge and not just the trivial 0.0 abstain path.
    history: Sequence[ToolEvent]

    history = [ToolEvent(tool="read", args={"path": "a"}) for _ in range(50)]
    assert LoopDetector().evaluate(history).confidence == pytest.approx(1.0)

    history = [_failing("search") for _ in range(50)]
    assert RetryDetector().evaluate(history).confidence == pytest.approx(1.0)

    history = [ToolEvent(tool="llm", cost_usd=1000.0)]
    assert BudgetDetector().evaluate(history).confidence == pytest.approx(1.0)

    history = [_stalling_test() for _ in range(50)]
    assert NoProgressDetector(coding_anchors()).evaluate(history).confidence == pytest.approx(1.0)
