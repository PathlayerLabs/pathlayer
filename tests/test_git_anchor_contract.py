"""GitProgressAnchor capture-contract regression (audit F-5).

``GitProgressAnchor`` documents (see ``progress.py``) that the diff stat it
consumes must be computed against a **baseline frozen at run start**, not a
moving ``HEAD``. The risk was documented but untested: if the capture layer
feeds a diff-vs-moving-``HEAD`` instead, an agent that commits mid-run sees its
cumulative diff collapse to ~0, the level goes flat, and ``NoProgressDetector``
false-stops a genuinely progressing run.

These tests pin both sides of that contract using the *same underlying work*
(an agent adding new code every step). The only thing that changes is how the
capture layer reports the diff size:

- frozen baseline -> cumulative diff grows -> new records -> detector silent;
- moving HEAD (commit each step) -> diff size flat -> stall -> false stop.

If a future change makes the frozen-baseline case fire, that is a regression.
"""

from __future__ import annotations

from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import GitProgressAnchor
from pathlayer.events import ToolEvent

_LINES_PER_STEP = 6
_STEPS = 25


def _git_event(diff_size: int) -> ToolEvent:
    """A tool call carrying a git diff stat of ``diff_size`` total lines."""
    return ToolEvent(tool="git_diff", args={}, result={"insertions": diff_size, "deletions": 0})


def _frozen_baseline_history() -> list[ToolEvent]:
    # Correct capture: each step's diff is measured vs the run-start commit, so a
    # progressing agent's cumulative diff grows monotonically (6, 12, 18, ...).
    return [_git_event(_LINES_PER_STEP * step) for step in range(1, _STEPS + 1)]


def _moving_head_history() -> list[ToolEvent]:
    # Contract violation: the agent commits after each step, so the diff is
    # measured vs a moving HEAD and collapses back to a single step's size every
    # time -> a flat level despite real, ongoing work.
    return [_git_event(_LINES_PER_STEP) for _ in range(_STEPS)]


def test_frozen_baseline_progress_stays_silent() -> None:
    # A correctly-wired anchor (frozen baseline) must NOT stop a progressing run:
    # the growing cumulative diff sets a new record every step, so CUSUM never
    # accumulates and the no-progress confidence stays at 0.
    detector = NoProgressDetector([GitProgressAnchor()])
    signal = detector.evaluate(_frozen_baseline_history())
    assert signal.confidence == 0.0


def test_moving_head_diff_collapse_false_stops() -> None:
    # The documented blind spot, made visible: a moving-HEAD diff that collapses
    # after each commit looks flat, so the detector fires on a run that is in fact
    # making progress. This is the cost of violating the frozen-baseline contract.
    detector = NoProgressDetector([GitProgressAnchor()])
    signal = detector.evaluate(_moving_head_history())
    assert signal.confidence > 0.0


def test_contract_difference_is_only_the_capture_wiring() -> None:
    # The two histories represent the SAME real work (same lines added per step);
    # only the capture wiring differs. The detector's opposite verdicts therefore
    # isolate the contract as the single cause -- the point of the regression.
    detector = NoProgressDetector([GitProgressAnchor()])
    frozen = detector.evaluate(_frozen_baseline_history())
    moving = detector.evaluate(_moving_head_history())
    assert frozen.confidence == 0.0
    assert moving.confidence > frozen.confidence
