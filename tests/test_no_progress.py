"""NOP-2: tests for the coding progress anchors and the CUSUM NoProgressDetector."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    GitProgressAnchor,
    TestProgressAnchor,
    coding_anchors,
)
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Policy
from pathlayer.policies.base import PolicyConfig
from pathlayer.validation import load_trace

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"

# Policy that lets the (otherwise advisory) no_progress signal STOP, for
# measuring detection quality directly.
_ACTIVE = Policy(PolicyConfig(advisory_signals=frozenset()))


def _tests(passed: int, failed: int, target: str | None = None) -> ToolEvent:
    args = {"target": target} if target is not None else {}
    return ToolEvent(tool="run_tests", args=args, result={"passed": passed, "failed": failed})


def _edit(patch: str = "tweak") -> ToolEvent:
    return ToolEvent(tool="edit_file", args={"patch": patch})


def _shell(exit_code: int) -> ToolEvent:
    return ToolEvent(tool="shell", result={"exit_code": exit_code})


def _email(to: str) -> ToolEvent:
    return ToolEvent(tool="send_email", args={"to": to})


def _git(insertions: int, deletions: int) -> ToolEvent:
    # Cumulative diff stat vs the run's frozen baseline (NOP-4 capture contract).
    return ToolEvent(tool="edit_file", result={"insertions": insertions, "deletions": deletions})


def _detector(**kwargs: float) -> NoProgressDetector:
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


def test_test_anchor_uses_latest_result_per_target() -> None:
    # Same suite re-run -> latest result replaces the earlier one.
    assert TestProgressAnchor().measure([_tests(5, 5), _tests(8, 2)]) == 6.0


def test_test_anchor_sums_across_targets() -> None:
    # Switching to a second suite sums each suite's latest passed-failed instead
    # of comparing the small suite's level against the big one's (which would
    # manufacture a huge spurious drop).
    big = _tests(50, 0, target="tests/test_big.py")
    small = _tests(2, 8, target="tests/test_small.py")
    assert TestProgressAnchor().measure([big]) == 50.0
    assert TestProgressAnchor().measure([big, small]) == 44.0  # 50 + (2 - 8)


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
# GitProgressAnchor (NOP-4): no-test fallback on the working diff size
# --------------------------------------------------------------------------- #
def test_git_anchor_level_is_diff_size() -> None:
    # size = insertions + deletions, so in-place edits and deletions count.
    assert GitProgressAnchor().measure([_git(3, 3)]) == 6.0


def test_git_anchor_abstains_without_evidence() -> None:
    assert GitProgressAnchor().measure([_email("a"), _edit()]) is None


def test_git_anchor_uses_latest_cumulative_stat() -> None:
    assert GitProgressAnchor().measure([_git(4, 4), _git(10, 2)]) == 12.0


def test_git_anchor_defers_to_tests() -> None:
    # Tests are the stronger ground truth: git steps aside so a growing diff
    # cannot mask a real test stall under optimistic aggregation.
    history = [_git(3, 3), _tests(1, 0)]
    assert GitProgressAnchor().measure(history) is None
    assert TestProgressAnchor().measure(history) == 1.0


def test_git_anchor_reads_net_lines_shortcut() -> None:
    # Degraded capture: only a signed net is available -> size is its magnitude.
    event = ToolEvent(tool="edit_file", result={"net_lines": -7})
    assert GitProgressAnchor().measure([event]) == 7.0


def test_git_anchor_reads_metadata_evidence() -> None:
    event = ToolEvent(tool="edit_file", metadata={"insertions": 5, "deletions": 1})
    assert GitProgressAnchor().measure([event]) == 6.0


def test_git_anchor_never_satisfied() -> None:
    # A non-empty diff means "code changed", not "task done".
    assert GitProgressAnchor().is_satisfied([_git(3, 3)]) is False


def test_detector_fires_on_same_zone_churn_without_tests() -> None:
    # No tests, no exit codes: the agent rewrites the same zone so the diff size
    # stays flat -> the git anchor is the only one applicable and it fires.
    history = [_git(4, 4) for _ in range(13)]
    assert _confidence(_detector(), *history) == 1.0


def test_detector_silent_on_growing_git_footprint_without_tests() -> None:
    # In-place fixing across new lines grows the diff size every step -> records
    # -> silence. This is the additive-bias fix the size scalar gives over net.
    history = [_git(3 * i, 3 * i) for i in range(1, 13)]
    assert _confidence(_detector(), *history) == 0.0


def test_git_anchor_does_not_mask_test_stall() -> None:
    # A growing diff alongside a flat test suite: git defers to tests, so the
    # test stall still fires (no masking) -- the validated deferring behavior.
    history: list[ToolEvent] = []
    for i in range(1, 13):
        history += [_git(5 * i, 0), _tests(2, 2)]
    assert _confidence(_detector(), *history) == 1.0


# --------------------------------------------------------------------------- #
# NoProgressDetector: construction guards
# --------------------------------------------------------------------------- #
def test_requires_at_least_one_anchor() -> None:
    with pytest.raises(ValueError):
        NoProgressDetector([])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_rate": 0.0},
        {"target_rate": -1.0},
        {"threshold": 0.0},
        {"window": 10},  # one short of ceil(2.0/0.2) + 1 = 11 -> STOP unreachable
    ],
)
def test_invalid_params_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _detector(**kwargs)


def test_min_window_is_reachable_boundary() -> None:
    # window = ceil(h/k) + 1 is the smallest that keeps a STOP reachable: a full
    # window of stagnant steps must push confidence to the default stop_threshold.
    detector = _detector(window=11)  # defaults k=0.2, h=2.0
    flat = [_tests(11, 3) for _ in range(11)]
    assert detector.evaluate(flat).confidence >= 0.9


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


def test_tolerates_noisy_but_upward_progress() -> None:
    # Upward zigzag (+2, -1, +2, ...): every climb sets a new record and resets
    # the CUSUM, so genuine red/green churn must not be flagged.
    history = [
        _tests(11, 4),  # 7
        _tests(13, 4),  # 9  record
        _tests(12, 4),  # 8  dip -1
        _tests(14, 4),  # 10 record
        _tests(13, 4),  # 9  dip -1
        _tests(15, 4),  # 11 record
    ]
    assert _confidence(_detector(), *history) == 0.0


def test_does_not_fire_on_isolated_regression() -> None:
    # A single -2 drop is just "no new record" (one stagnant step), never an
    # instant stop -- the full patience must elapse before firing.
    conf = _confidence(_detector(), _tests(11, 1), _tests(11, 3))  # 10 -> 8
    assert conf == pytest.approx(0.1)


def test_resets_on_new_record() -> None:
    # Flat for a while, then a genuine improvement sets a record -> cleared.
    history = [_tests(11, 3), _tests(11, 3), _tests(11, 3), _tests(11, 0)]
    assert _confidence(_detector(), *history) == 0.0


def test_cusum_resets_on_none_gap() -> None:
    # Defensive: a None mid-series (anchor lost its baseline) resets the sum.
    detector = _detector()
    assert detector._cusum([8.0, 8.0, 8.0, 8.0]) == pytest.approx(0.6)  # 3 stagnant steps
    assert detector._cusum([8.0, None, 8.0, 8.0]) == pytest.approx(0.2)  # gap resets, 1 step


# --------------------------------------------------------------------------- #
# NoProgressDetector: firing
# --------------------------------------------------------------------------- #
def test_ramps_with_stagnation() -> None:
    # 3 identical results -> 2 stagnant steps -> 2 * 0.2 / 2.0 = 0.2.
    assert _confidence(_detector(), _tests(11, 3), _tests(11, 3), _tests(11, 3)) == pytest.approx(
        0.2
    )


def test_fires_at_threshold() -> None:
    detector = _detector()  # target_rate=0.2, threshold=2.0
    # 13 identical results -> 12 stagnant steps -> C past threshold -> 1.0.
    signal = detector.evaluate([_tests(11, 3) for _ in range(13)])
    assert signal.confidence == 1.0
    assert "without new progress" in signal.reason


def test_flat_with_interleaved_edits_fires() -> None:
    # Genuinely different edits each turn (no repetition) but the test result
    # never moves: high activity, flat progress -- the differentiating case.
    history: list[ToolEvent] = [_tests(11, 3)]
    for i in range(10):
        history += [_edit(f"patch {i}"), _tests(11, 3)]
    assert _confidence(_detector(), *history) == 1.0


def test_fires_on_exit_code_stall_without_tests() -> None:
    # A shell-only agent (no test results) is measured by the exit-code anchor:
    # one success then a run of failures leaves the success count flat -> fires.
    history = [_shell(0)] + [_shell(1) for _ in range(12)]
    assert _confidence(_detector(), *history) == 1.0


def test_exit_code_success_does_not_mask_test_stall() -> None:
    # Every shell succeeds (a rising success count on its own), yet the tests
    # never move. The exit-code anchor abstains because tests are present, so the
    # test stall is not masked and the detector still fires.
    history: list[ToolEvent] = [_tests(11, 3)]
    for _ in range(10):
        history += [_shell(0), _tests(11, 3)]
    assert _confidence(_detector(), *history) == 1.0


def test_window_bounds_stagnation() -> None:
    # The window caps how far back stagnation accumulates: over 20 flat events a
    # window of 11 only counts 10 stagnant steps, where window 64 counts all 19.
    flat = [_tests(11, 3) for _ in range(20)]
    assert "10 steps without new progress" in _detector(window=11).evaluate(flat).reason
    assert "19 steps without new progress" in _detector(window=64).evaluate(flat).reason


# --------------------------------------------------------------------------- #
# success suppression (is_satisfied): "done" vs "stuck"
# --------------------------------------------------------------------------- #
def test_test_anchor_satisfied_when_all_green() -> None:
    assert TestProgressAnchor().is_satisfied([_tests(14, 0)]) is True


def test_test_anchor_not_satisfied_with_failures() -> None:
    assert TestProgressAnchor().is_satisfied([_tests(11, 3)]) is False


def test_test_anchor_not_satisfied_without_passed() -> None:
    # An empty "0 passed / 0 failed" run is no signal, not a success.
    assert TestProgressAnchor().is_satisfied([_tests(0, 0)]) is False


def test_test_anchor_not_satisfied_without_evidence() -> None:
    assert TestProgressAnchor().is_satisfied([_email("a")]) is False
    assert TestProgressAnchor().is_satisfied([]) is False


def test_test_anchor_satisfied_requires_every_suite_green() -> None:
    # Per-suite aggregation: a green re-run of A must not mask a still-red B.
    green_a = _tests(5, 0, target="tests/test_a.py")
    red_b = _tests(2, 8, target="tests/test_b.py")
    assert TestProgressAnchor().is_satisfied([green_a]) is True
    assert TestProgressAnchor().is_satisfied([green_a, red_b]) is False


def test_test_anchor_not_satisfied_when_a_sibling_suite_is_empty() -> None:
    # A passing suite must not mask a sibling at 0/0 ("no signal"): checking
    # per suite, the empty suite is not satisfied, so the whole is not either.
    green_a = _tests(5, 0, target="tests/test_a.py")
    empty_b = _tests(0, 0, target="tests/test_b.py")
    assert TestProgressAnchor().is_satisfied([green_a, empty_b]) is False


def test_exit_code_anchor_never_satisfied() -> None:
    # exit_code 0 means "this command succeeded", not "the task is done".
    assert ExitCodeProgressAnchor().is_satisfied([_shell(0), _shell(0)]) is False


def test_detector_silent_once_suite_goes_green() -> None:
    # A stall that ends green is "fixed", not stuck: success suppression returns
    # 0 even though the earlier steps were flat-red.
    history = [_tests(11, 3) for _ in range(5)] + [_tests(14, 0)]
    assert _confidence(_detector(), *history) == 0.0


def test_detector_stays_active_while_a_suite_is_red() -> None:
    # Suite A is green but suite B keeps failing flat: total failed > 0, so the
    # detector is not suppressed and the stall on B still fires.
    history = [_tests(5, 0, target="tests/test_a.py")] + [
        _tests(2, 8, target="tests/test_b.py") for _ in range(13)
    ]
    assert _confidence(_detector(), *history) == 1.0


# --------------------------------------------------------------------------- #
# arbitration: no_progress is actionable by default (NOP-3) + golden fixtures
# --------------------------------------------------------------------------- #
def _run(events: Sequence[ToolEvent], policy: Policy | None = None) -> Pipeline:
    pipeline = Pipeline([_detector()], policy if policy is not None else Policy())
    pipeline.run(events)
    return pipeline


def test_default_policy_stops_on_coding_stall() -> None:
    # Since NOP-3 no_progress is actionable: the anchored stall that used to be
    # observe-only now STOPs under the default policy.
    trace = load_trace(_TRACES_DIR / "no_progress" / "failing_edit_churn.json")
    assert _detector().evaluate(trace.events).confidence >= 0.9
    assert _run(trace.events).stopped


def test_active_policy_also_stops_on_coding_stall() -> None:
    # Pinned explicitly with an empty advisory set (equivalent to the default).
    trace = load_trace(_TRACES_DIR / "no_progress" / "failing_edit_churn.json")
    assert _run(trace.events, _ACTIVE).stopped


@pytest.mark.parametrize(
    "name",
    [
        "varied_long_run.json",
        "noisy_progress.json",
        "retry_then_succeed.json",
        "plateau_after_success.json",
        "slow_but_real_progress.json",
        "finished_without_tests.json",
    ],
)
def test_default_policy_silent_on_healthy(name: str) -> None:
    trace = load_trace(_TRACES_DIR / "healthy" / name)
    assert not _run(trace.events).stopped


def test_pipeline_does_not_stop_email_agent() -> None:
    # The coding anchor abstains end to end -> the run is never stopped.
    assert not _run([_email(f"u{i}") for i in range(12)]).stopped
