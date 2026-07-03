"""LOOP-2 precision: tests for the positive-progress signal (`is_advancing`).

`is_advancing` is deliberately distinct from `NoProgressDetector` staying
silent: silence there means "record just set" *or* "no anchor applicable yet"
(F1b), and the two are indistinguishable from that signal alone. This function
collapses only the first case to `True`, so a caller (the arbitration
suppression, LOOP-2 precision) never mistakes "no evidence" for "progressing".
"""

from __future__ import annotations

from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    GitProgressAnchor,
    TestProgressAnchor,
    is_advancing,
    loop_suppression_anchors,
)
from pathlayer.events import ToolEvent

_TEST_ANCHORS = [TestProgressAnchor(), ExitCodeProgressAnchor()]


def _tests(passed: int, failed: int) -> ToolEvent:
    return ToolEvent(tool="run_tests", args={}, result={"passed": passed, "failed": failed})


def _edit(patch: str = "tweak") -> ToolEvent:
    return ToolEvent(tool="edit_file", args={"patch": patch})


def _shell(exit_code: int) -> ToolEvent:
    return ToolEvent(tool="shell", result={"exit_code": exit_code})


def test_empty_history_is_not_advancing() -> None:
    assert is_advancing(_TEST_ANCHORS, []) is False


def test_first_ever_test_result_is_not_advancing() -> None:
    # No prior measurement to be strictly greater than: `measure(prefix)` is
    # `None` (no evidence at all yet), which abstains rather than reading as a
    # baseline of zero. A single data point is not (yet) a trend.
    history = [_tests(1, 0)]
    assert is_advancing(_TEST_ANCHORS, history) is False


def test_rising_test_count_is_advancing_on_every_repeat() -> None:
    # The F1a scenario: identical (tool, args), passed climbing each time.
    # The very first result has no baseline (see the test above); every repeat
    # after that sets a fresh record.
    history = [_tests(1, 0)]
    for passed in range(2, 6):
        history.append(_tests(passed, 0))
        assert is_advancing(_TEST_ANCHORS, history) is True


def test_stagnant_repeat_is_not_advancing() -> None:
    # Same result twice in a row: the last step set no new record.
    history = [_tests(4, 0), _tests(4, 0)]
    assert is_advancing(_TEST_ANCHORS, history) is False


def test_regressing_test_count_is_not_advancing() -> None:
    history = [_tests(5, 0), _tests(3, 2)]
    assert is_advancing(_TEST_ANCHORS, history) is False


def test_no_applicable_anchor_is_not_advancing() -> None:
    # F1b: an email agent (or any tool with no test/exit_code evidence) must
    # never read as "advancing" just because no anchor has an opinion.
    history = [_edit(), _edit("another")]
    assert is_advancing(_TEST_ANCHORS, history) is False


def test_gap_in_evidence_is_not_advancing() -> None:
    # A record was set, then the next event carries no test evidence at all:
    # TestProgressAnchor.measure keeps returning the last known level (12),
    # so the "current == previous" comparison correctly reads as no fresh
    # record -- a gap must never be confused with a new high.
    history = [_tests(1, 0), _tests(2, 0), _edit()]
    assert is_advancing(_TEST_ANCHORS, history) is False


def test_exit_code_anchor_success_is_advancing() -> None:
    history = [_shell(1), _shell(0)]
    assert is_advancing(_TEST_ANCHORS, history) is True


def test_exit_code_anchor_repeat_failure_is_not_advancing() -> None:
    history = [_shell(1), _shell(1)]
    assert is_advancing(_TEST_ANCHORS, history) is False


def test_loop_suppression_anchors_excludes_git() -> None:
    # Trou C: git diff size is gameable (ping-pong + useless padding grows the
    # diff every turn), so it must never back the loop<-progress suppression,
    # unlike the full `coding_anchors()` set NoProgressDetector uses.
    anchors = loop_suppression_anchors()
    assert {type(a) for a in anchors} == {TestProgressAnchor, ExitCodeProgressAnchor}
    assert not any(isinstance(a, GitProgressAnchor) for a in anchors)


def test_git_only_progress_never_suppresses_via_loop_suppression_anchors() -> None:
    # Adversarial case (Trou C): a growing git diff, no test/exit_code evidence
    # at all, must never register as "advancing" for the suppression's anchors.
    history = [
        ToolEvent(tool="edit_file", result={"insertions": 5, "deletions": 0}),
        ToolEvent(tool="edit_file", result={"insertions": 12, "deletions": 0}),
    ]
    assert is_advancing(loop_suppression_anchors(), history) is False
    # Confirms the git anchor *would* have called this advancing on its own --
    # i.e. the exclusion in loop_suppression_anchors is load-bearing, not moot.
    assert is_advancing([GitProgressAnchor()], history) is True
