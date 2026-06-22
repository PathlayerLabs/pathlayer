"""NOP-1: tests for the coding progress anchors and the NoProgressDetector."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    TestProgressAnchor,
    coding_anchors,
)
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Policy
from pathlayer.validation import load_trace

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def _tests(passed: int, failed: int) -> ToolEvent:
    return ToolEvent(tool="run_tests", result={"passed": passed, "failed": failed})


def _edit(patch: str = "tweak") -> ToolEvent:
    return ToolEvent(tool="edit_file", args={"patch": patch})


def _shell(exit_code: int) -> ToolEvent:
    return ToolEvent(tool="shell", result={"exit_code": exit_code})


def _email(to: str) -> ToolEvent:
    return ToolEvent(tool="send_email", args={"to": to})


def _detector(**kwargs: int) -> NoProgressDetector:
    return NoProgressDetector(coding_anchors(), **kwargs)


def _confidence(detector: NoProgressDetector, *events: ToolEvent) -> float:
    return detector.evaluate(list(events)).confidence


# --------------------------------------------------------------------------- #
# TestProgressAnchor: auto-gating + level
# --------------------------------------------------------------------------- #
def test_test_anchor_abstains_without_evidence() -> None:
    assert TestProgressAnchor().measure([_email("a"), _email("b")]) is None


def test_test_anchor_abstains_on_empty_history() -> None:
    assert TestProgressAnchor().measure([]) is None


def test_test_anchor_level_is_passed_minus_failed() -> None:
    assert TestProgressAnchor().measure([_tests(10, 3)]) == 7.0


def test_test_anchor_uses_latest_test_event() -> None:
    assert TestProgressAnchor().measure([_tests(5, 5), _tests(8, 2)]) == 6.0


def test_test_anchor_reads_metadata_evidence() -> None:
    event = ToolEvent(tool="ci", metadata={"passed": 0, "failed": 2})
    assert TestProgressAnchor().measure([event]) == -2.0


# --------------------------------------------------------------------------- #
# ExitCodeProgressAnchor: fallback that defers to tests
# --------------------------------------------------------------------------- #
def test_exit_code_anchor_counts_exit_zero() -> None:
    history = [_shell(0), _shell(1), _shell(0)]
    assert ExitCodeProgressAnchor().measure(history) == 2.0


def test_exit_code_anchor_abstains_without_evidence() -> None:
    assert ExitCodeProgressAnchor().measure([_email("a")]) is None


def test_exit_code_anchor_abstains_when_tests_present() -> None:
    # Tests are the stronger ground truth: the exit-code anchor steps aside so a
    # never-decreasing success count cannot mask a real test stall.
    history = [_shell(0), _tests(1, 0)]
    assert ExitCodeProgressAnchor().measure(history) is None
    assert TestProgressAnchor().measure(history) == 1.0


def test_exit_code_anchor_tolerates_non_int_exit_code() -> None:
    # A malformed exit code is treated as "not a success", not a crash.
    event = ToolEvent(tool="shell", result={"exit_code": "boom"})
    assert ExitCodeProgressAnchor().measure([event]) == 0.0


# --------------------------------------------------------------------------- #
# NoProgressDetector: construction guards
# --------------------------------------------------------------------------- #
def test_requires_at_least_one_anchor() -> None:
    with pytest.raises(ValueError):
        NoProgressDetector([])


@pytest.mark.parametrize(
    "kwargs",
    [{"no_progress_steps": 0}, {"window": 0}, {"no_progress_steps": 10, "window": 5}],
)
def test_invalid_params_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        _detector(**kwargs)


# --------------------------------------------------------------------------- #
# NoProgressDetector: staying silent
# --------------------------------------------------------------------------- #
def test_silent_without_applicable_anchor() -> None:
    # An email agent carries no coding evidence -> the coding anchor abstains and
    # the detector must not flag it. This is the anti-false-positive guarantee.
    assert _confidence(_detector(), *(_email(f"u{i}") for i in range(8))) == 0.0


def test_silent_on_empty_history() -> None:
    assert _detector().evaluate([]).confidence == 0.0


def test_silent_on_single_event() -> None:
    # One test event establishes a baseline but has no earlier point to compare.
    assert _confidence(_detector(), _tests(11, 3)) == 0.0


def test_silent_on_real_progress() -> None:
    history = [_tests(11, 3), _edit(), _tests(11, 2), _edit(), _tests(11, 0)]
    assert _confidence(_detector(), *history) == 0.0


def test_resets_on_improvement() -> None:
    # Flat for a while, then a genuine improvement clears the stagnation.
    history = [_tests(11, 3), _tests(11, 3), _tests(11, 3), _tests(11, 0)]
    assert _confidence(_detector(), *history) == 0.0


# --------------------------------------------------------------------------- #
# NoProgressDetector: firing
# --------------------------------------------------------------------------- #
def test_ramps_with_stagnation() -> None:
    detector = _detector(no_progress_steps=5)
    # 3 identical test results -> 2 flat transitions -> 2/5.
    assert _confidence(detector, _tests(11, 3), _tests(11, 3), _tests(11, 3)) == pytest.approx(0.4)


def test_fires_at_threshold() -> None:
    detector = _detector(no_progress_steps=5)
    # 6 identical test results -> 5 flat transitions -> saturates at 1.0.
    signal = detector.evaluate([_tests(11, 3) for _ in range(6)])
    assert signal.confidence == 1.0
    assert "without progress" in signal.reason


def test_flat_with_interleaved_edits_fires() -> None:
    # Genuinely different edits each turn (no repetition) but the test result
    # never moves: high activity, flat progress.
    history: list[ToolEvent] = [_tests(11, 3)]
    for i in range(5):
        history += [_edit(f"patch {i}"), _tests(11, 3)]
    assert _confidence(_detector(no_progress_steps=5), *history) == 1.0


def test_fires_on_exit_code_stall_without_tests() -> None:
    # A shell-only agent (no test results) is measured by the exit-code anchor:
    # one success then a run of failures leaves the success count flat -> fires.
    history = [_shell(0)] + [_shell(1) for _ in range(6)]
    assert _confidence(_detector(no_progress_steps=5), *history) == 1.0


def test_exit_code_success_does_not_mask_test_stall() -> None:
    # Every shell succeeds (a rising success count on its own), yet the tests
    # never move. The exit-code anchor abstains because tests are present, so the
    # test stall is not masked and the detector still fires.
    history: list[ToolEvent] = [_tests(11, 3)]
    for _ in range(5):
        history += [_shell(0), _tests(11, 3)]
    assert _confidence(_detector(no_progress_steps=5), *history) == 1.0


def test_window_bounds_stagnation() -> None:
    # The window caps how far back stagnation is counted: with window=5 and 10
    # flat events, the streak is reported as 5, not 9.
    detector = _detector(no_progress_steps=3, window=5)
    signal = detector.evaluate([_tests(11, 3) for _ in range(10)])
    assert signal.confidence == 1.0
    assert "5 actions without progress" in signal.reason


# --------------------------------------------------------------------------- #
# arbitration + golden fixtures
# --------------------------------------------------------------------------- #
def _run(events: Sequence[ToolEvent]) -> Pipeline:
    pipeline = Pipeline([_detector(no_progress_steps=5)], Policy())
    pipeline.run(events)
    return pipeline


def test_pipeline_stops_on_coding_stall() -> None:
    trace = load_trace(_TRACES_DIR / "no_progress" / "failing_edit_churn.json")
    assert _run(trace.events).stopped


def test_pipeline_does_not_stop_on_healthy_progress() -> None:
    trace = load_trace(_TRACES_DIR / "healthy" / "varied_long_run.json")
    assert not _run(trace.events).stopped


def test_pipeline_does_not_stop_email_agent() -> None:
    # The coding anchor abstains end to end -> the run is never stopped.
    assert not _run([_email(f"u{i}") for i in range(12)]).stopped
