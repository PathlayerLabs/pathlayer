"""RET-1: tests for the RetryDetector."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.validation import load_trace

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def _fail(error: str = "boom", **args: object) -> ToolEvent:
    return ToolEvent(tool="call", args=args, result={"error": error})


def _ok(**args: object) -> ToolEvent:
    return ToolEvent(tool="call", args=args, result={"ok": True})


def _confidence(detector: RetryDetector, *events: ToolEvent) -> float:
    return detector.evaluate(list(events)).confidence


# --------------------------------------------------------------------------- #
# firing
# --------------------------------------------------------------------------- #
def test_fires_at_threshold() -> None:
    signal = RetryDetector(threshold=3).evaluate([_fail(), _fail(), _fail()])
    assert signal.confidence == 1.0
    assert "3x identical failure" in signal.reason


def test_ramps_below_threshold() -> None:
    # 2nd identical failure with threshold 3 -> halfway.
    assert _confidence(RetryDetector(threshold=3), _fail(), _fail()) == pytest.approx(0.5)


def test_single_failure_is_silent() -> None:
    assert _confidence(RetryDetector(), _fail()) == 0.0


def test_empty_history_is_silent() -> None:
    assert RetryDetector().evaluate([]).confidence == 0.0


def test_error_marker_from_metadata() -> None:
    events = [ToolEvent(tool="call", metadata={"error": "429"}) for _ in range(3)]
    assert RetryDetector(threshold=3).evaluate(events).confidence == 1.0


def test_volatile_args_still_count_as_identical() -> None:
    # Different request_id each time, but canonicalization masks it -> a retry.
    events = [_fail("429", request_id=f"req-{i}") for i in range(3)]
    assert RetryDetector(threshold=3).evaluate(events).confidence == 1.0


# --------------------------------------------------------------------------- #
# staying silent
# --------------------------------------------------------------------------- #
def test_silent_when_last_attempt_succeeds() -> None:
    assert _confidence(RetryDetector(threshold=2), _fail(), _fail(), _ok()) == 0.0


def test_silent_when_args_change() -> None:
    # Streak broken by a different call -> the tail run is length 1.
    assert _confidence(RetryDetector(threshold=2), _fail(x=1), _fail(x=1), _fail(x=2)) == 0.0


def test_silent_when_error_changes() -> None:
    assert _confidence(RetryDetector(threshold=2), _fail("a"), _fail("b")) == 0.0


@pytest.mark.parametrize("falsy", [None, ""])
def test_falsy_error_marker_is_not_a_failure(falsy: object) -> None:
    # {"error": null} / {"error": ""} is a success, not a retry.
    events = [ToolEvent(tool="call", result={"error": falsy}) for _ in range(3)]
    assert RetryDetector(threshold=3).evaluate(events).confidence == 0.0


@pytest.mark.parametrize("threshold", [0, 1])
def test_invalid_threshold_raises(threshold: int) -> None:
    with pytest.raises(ValueError):
        RetryDetector(threshold=threshold)


# --------------------------------------------------------------------------- #
# arbitration + golden fixture
# --------------------------------------------------------------------------- #
def test_pipeline_stops_on_retry_storm() -> None:
    pipeline = Pipeline([RetryDetector(threshold=3)], Policy())
    pipeline.run([_fail(), _fail(), _fail(), _fail()])
    assert pipeline.stopped


def test_golden_retry_fixture_fires() -> None:
    trace = load_trace(_TRACES_DIR / "loop" / "retry_same_error.json")
    signal = RetryDetector(threshold=3).evaluate(trace.events)
    assert signal.confidence == 1.0
    assert Policy().decide([signal]) is Action.STOP
