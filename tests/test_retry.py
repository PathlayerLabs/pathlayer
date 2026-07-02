"""RET-1: tests for the RetryDetector."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathlayer.detectors.base import Signal
from pathlayer.detectors.canonical import ToolMask
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.validation import load_trace

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _fail(error: str = "boom", **args: object) -> ToolEvent:
    return ToolEvent(tool="call", args=args, result={"error": error})


def _ok(**args: object) -> ToolEvent:
    return ToolEvent(tool="call", args=args, result={"ok": True})


def _search(query: str, error: str | None = None) -> ToolEvent:
    """A ``search`` call with a genuinely distinct arg and optional error.

    Distinct ``query`` values keep the canonical token different every time, so
    RET-1 (exact-match streak) and the LoopDetector stay silent -- this is the
    RET-2 niche: same tool, real argument variation, mixed results.
    """
    result: dict[str, object] = {"error": error} if error else {"ok": True}
    return ToolEvent(tool="search", args={"query": query}, result=result)


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


def test_per_tool_mask_restores_retry_detection_for_attempt_id() -> None:
    # CAN-2 decisive case, tied to the CAN-3 trade-off: an incrementing
    # ``attempt_id`` is visible by default (no ``_id`` suffix masking since
    # CAN-3), so each failed deploy looks like a different call and the retry
    # storm goes unnoticed. Declaring ``attempt_id`` in ``mask`` for that tool
    # collapses the attempts back to one token -> the retry is detected again.
    events = [
        ToolEvent(tool="deploy", args={"service": "api", "attempt_id": i}, result={"error": "boom"})
        for i in range(3)
    ]
    assert RetryDetector(threshold=3).evaluate(events).confidence == 0.0
    masks = {"deploy": ToolMask(mask=frozenset({"attempt_id"}))}
    assert RetryDetector(threshold=3, tool_masks=masks).evaluate(events).confidence == 1.0


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


# --------------------------------------------------------------------------- #
# RET-2: same tool regardless of error (sliding-window dominance)
# --------------------------------------------------------------------------- #
def test_dominance_fires_on_tool_storm_with_distinct_args() -> None:
    # Same tool, genuinely different args, mixed success/failure: RET-1 and the
    # loop detector stay silent, but one tool clearly dominates the window.
    events = [_search(f"q{i}", error="boom" if i % 2 else None) for i in range(8)]
    signal = RetryDetector().evaluate(events)
    assert signal.confidence == pytest.approx(0.85)
    assert "'search' dominates" in signal.reason


def test_dominance_is_capped_below_stop_threshold() -> None:
    # The point of the cap: a tool storm alone must never STOP a run.
    events = [_search(f"q{i}", error="boom") for i in range(50)]
    signal = RetryDetector().evaluate(events)
    assert signal.confidence == pytest.approx(0.85)
    assert Policy().decide([signal]) is Action.NONE


def test_dominance_reinforces_a_stop_via_fusion() -> None:
    # Capped alone, but it tips a moderate no-progress signal over stop_threshold
    # through the noisy-OR fusion -- exactly what RET-2 is for.
    events = [_search(f"q{i}", error="boom" if i % 2 else None) for i in range(8)]
    retry_signal = RetryDetector().evaluate(events)
    fused = Policy().decide([retry_signal, Signal("no_progress", confidence=0.5)])
    assert fused is Action.STOP


def test_dominance_silent_on_all_success_burst() -> None:
    # A legitimate exploration/poll of one tool with no failures stays silent:
    # the failure gate keeps RET-2 from stopping healthy monotool workflows.
    events = [_search(f"q{i}") for i in range(10)]
    assert _confidence(RetryDetector(), *events) == 0.0


def test_dominance_silent_on_varied_multitool_workflow() -> None:
    # No single tool dominates the window -> silent, even with some failures.
    tools = ["list_dir", "read_file", "edit_file", "run_tests", "git_commit"]
    events = [
        ToolEvent(
            tool=tools[i % len(tools)],
            args={"i": i},
            result={"error": "x"} if i == 0 else None,
        )
        for i in range(10)
    ]
    assert _confidence(RetryDetector(), *events) == 0.0


def test_dominance_silent_below_ratio() -> None:
    # One tool reaches dominance_min_calls but not the window share threshold.
    storm = [_search(f"q{i}", error="boom") for i in range(6)]
    filler = [ToolEvent(tool=f"other_{i}", args={}) for i in range(6)]
    assert _confidence(RetryDetector(), *storm, *filler) == 0.0


def test_dominance_silent_below_min_calls() -> None:
    # Fewer than dominance_min_calls of the tool -> silent.
    events = [_search(f"q{i}", error="boom") for i in range(4)]
    assert _confidence(RetryDetector(), *events) == 0.0


def test_dominance_only_counts_recent_window() -> None:
    # A storm that has scrolled out of the window no longer dominates.
    storm = [_search(f"q{i}", error="boom") for i in range(6)]
    tail = [ToolEvent(tool=f"step_{i}", args={}) for i in range(16)]
    assert _confidence(RetryDetector(window=8), *storm, *tail) == 0.0


def test_ret1_streak_wins_over_dominance_when_both_fire() -> None:
    # Identical failing calls trip both components; RET-1 keeps full authority,
    # so the reported confidence is RET-1's and the reason names it.
    events = [_fail() for _ in range(5)]
    signal = RetryDetector(threshold=3).evaluate(events)
    assert signal.confidence == 1.0
    assert "identical failure" in signal.reason


def test_dominance_does_not_regress_ret1_success_tail() -> None:
    # A monotool storm whose last call succeeds: RET-1 is silent (last is ok),
    # but RET-2 still fires on the dominance -- and never exceeds its cap.
    events = [_search(f"q{i}", error="boom") for i in range(7)] + [_search("done")]
    signal = RetryDetector().evaluate(events)
    assert signal.confidence == pytest.approx(0.85)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dominance_min_calls": 1},
        {"window": 3, "dominance_min_calls": 5},
        {"dominance_ratio": 0.0},
        {"dominance_ratio": 1.5},
        {"dominance_max_confidence": 1.5},
        {"dominance_max_confidence": -0.1},
        {"min_failures": 0},
    ],
)
def test_invalid_dominance_params_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RetryDetector(**kwargs)  # type: ignore[arg-type]


def test_storm_fixture_fires_where_loop_and_ret1_are_silent() -> None:
    # The RET-2 niche, on a realistic trace: one tool hammered with genuinely
    # distinct args + mixed results. RET-2 fires (capped), while both the exact
    # RET-1 streak and the LoopDetector stay silent -- proving RET-2 catches a
    # storm the repetition detectors miss.
    trace = load_trace(_FIXTURES_DIR / "retry_tool_storm.json")
    retry = RetryDetector().evaluate(trace.events)
    assert retry.confidence == pytest.approx(0.85)
    assert "dominates" in retry.reason
    assert LoopDetector().evaluate(trace.events).confidence == 0.0
    # Capped below stop_threshold: the storm alone never stops the run.
    assert Policy().decide([retry]) is Action.NONE
