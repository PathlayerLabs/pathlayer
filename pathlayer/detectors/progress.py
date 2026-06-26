"""ProgressAnchor interface and the built-in coding anchors (NOP-1).

A :class:`ProgressAnchor` turns the observed event history into a *progress
level*: a scalar that rises when the agent makes real, verifiable headway and
stays flat when it is busy but stuck. ``NoProgressDetector`` aggregates one or
more anchors and never references a concrete one, so the same detector serves
coding, email, scraping, ... agents -- only the anchors change.

Auto-gating: ``measure`` returns ``None`` when the anchor sees none of its
evidence in the history (e.g. the coding anchors on an email agent). ``None``
means "no opinion", never "no progress" -- this is what prevents a false
positive on an agent the anchor was never built for.

Success suppression: ``is_satisfied`` reports when the signal is at its success
state (e.g. a green test suite). A flat level *there* is normal, not a stall, so
``NoProgressDetector`` stays silent while an applicable anchor is satisfied --
the brake that lets the no-progress signal STOP without halting an agent that
has actually succeeded. The default is ``False`` (an anchor with no notion of
success never suppresses).

The coding anchors are grounded on deterministic ground truth the events
already carry -- test ``passed`` / ``failed`` counts and process ``exit_code``
-- not text similarity. They are split so each anchor stays on a *single* scale:
folding a cumulative success count and a ``passed - failed`` level into one
anchor would compare an early exit-code prefix against a later test level (a
spurious "drop" -> false positive). ``ExitCodeProgressAnchor`` also abstains
whenever any test evidence is present, so tests take precedence and a
never-decreasing success count can never mask a real test stall under the
detector's optimistic (most-progress-wins) aggregation.

``GitProgressAnchor`` (NOP-4) adds a third coding ground truth: the size of the
git working diff, for coding agents that edit but never run tests. It defers to
tests (abstains while test evidence is present) but *not* to exit codes: when no
tests are present, git and the exit-code anchor are both applicable and the
detector takes the most-progress-wins view across them. The deferring shape was
validated on labeled traces (``tests/test_nop4_git_experiment.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pathlayer.events import ToolEvent

_PASSED = "passed"
_FAILED = "failed"
_EXIT_CODE = "exit_code"
_TEST_KEYS = (_PASSED, _FAILED)
_INSERTIONS = "insertions"
_DELETIONS = "deletions"
_NET_LINES = "net_lines"
_GIT_KEYS = (_INSERTIONS, _DELETIONS, _NET_LINES)


class ProgressAnchor(ABC):
    """Maps an event history to a continuous progress level, or ``None``."""

    name: str = "anchor"

    @abstractmethod
    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        """Return the current progress level, or ``None`` if not applicable.

        Pure and deterministic: the same history must yield the same value, with
        no clock, randomness, network or LLM. The value is a *level*, so a
        detector may call ``measure`` on prefixes of the history to recover the
        progress series over time.
        """
        raise NotImplementedError

    def is_satisfied(self, history: Sequence[ToolEvent]) -> bool:
        """Whether the progress signal is currently at its *success* state.

        "Satisfied" means "nothing broken left to fix right now" (e.g. a green
        test suite), not "the whole task is provably finished". A flat progress
        level *at* this state is normal, not a stall, so ``NoProgressDetector``
        stays silent while an applicable anchor is satisfied. Default ``False``:
        an anchor that has no notion of success never suppresses the detector.
        Pure and deterministic, like :meth:`measure`.
        """
        return False


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _evidence(event: ToolEvent, keys: tuple[str, ...]) -> Mapping[str, Any] | None:
    """Return the structured source (event result or metadata) carrying any of
    ``keys``, or ``None`` if the event carries none."""
    result = event.result
    if isinstance(result, Mapping) and any(key in result for key in keys):
        return result
    if any(key in event.metadata for key in keys):
        return event.metadata
    return None


def _test_target(event: ToolEvent) -> str:
    """Key identifying which test suite an event reports on.

    Uses the explicit ``target`` arg when present, else the tool name. This is
    what lets the level be summed *per suite* instead of comparing the
    passed-minus-failed of one suite against another (which would manufacture a
    huge spurious delta when the agent switches suites)."""
    target = event.args.get("target")
    return str(target) if target is not None else event.tool


def _latest_test_results(history: Sequence[ToolEvent]) -> dict[str, tuple[int, int]]:
    """Most recent ``(passed, failed)`` per test suite, keyed by :func:`_test_target`.

    Shared by :meth:`TestProgressAnchor.measure` (which sums ``passed - failed``)
    and :meth:`TestProgressAnchor.is_satisfied` (which checks the totals), so both
    aggregate per suite identically: a green re-run of suite A can never mask a
    still-red suite B.
    """
    latest: dict[str, tuple[int, int]] = {}
    for event in history:
        source = _evidence(event, _TEST_KEYS)
        if source is None:
            continue
        passed = _as_int(source.get(_PASSED, 0), default=0)
        failed = _as_int(source.get(_FAILED, 0), default=0)
        latest[_test_target(event)] = (passed, failed)
    return latest


class TestProgressAnchor(ProgressAnchor):
    """Ground-truth coding progress from test results.

    The level is the **sum over test suites** of ``passed - failed`` of each
    suite's most recent run (suites keyed by :func:`_test_target`). Summing per
    suite means switching suites or adding a new (initially red) suite changes
    the level by that suite's real delta, never by an artefact of comparing two
    different suites. Returns ``None`` when the history carries no test evidence,
    so it abstains on a test-less (e.g. non-coding) agent.

    ``is_satisfied`` is ``True`` once every suite's latest run is green
    (total ``failed == 0``) with at least one passing test (total ``passed > 0``,
    so an empty "0 passed / 0 failed" run is not mistaken for success).
    """

    name = "coding_tests"

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        latest = _latest_test_results(history)
        if not latest:
            return None
        return float(sum(passed - failed for passed, failed in latest.values()))

    def is_satisfied(self, history: Sequence[ToolEvent]) -> bool:
        latest = _latest_test_results(history)
        if not latest:
            return False
        # Per suite, not on totals: every suite must be individually green
        # (failed == 0) *and* carry a real result (passed > 0). Checking totals
        # would let a passing suite mask a sibling at 0/0 ("no signal"), wrongly
        # suppressing the detector while that empty suite stalls.
        return all(failed == 0 and passed > 0 for passed, failed in latest.values())


class ExitCodeProgressAnchor(ProgressAnchor):
    """Fallback coding progress from process exit codes.

    The level is the running count of successful (``exit_code == 0``) commands.
    It abstains (``None``) when the history carries no exit-code evidence *or*
    when it carries any test evidence: tests are the stronger ground truth and
    take precedence, which also keeps this never-decreasing count from masking a
    real test stall under the detector's optimistic aggregation.
    """

    name = "coding_exit_code"

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        successes = 0
        saw_exit_code = False
        for event in history:
            if _evidence(event, _TEST_KEYS) is not None:
                # Tests present anywhere -> defer to TestProgressAnchor.
                return None
            source = _evidence(event, (_EXIT_CODE,))
            if source is None:
                continue
            saw_exit_code = True
            if _as_int(source.get(_EXIT_CODE), default=-1) == 0:
                successes += 1
        return float(successes) if saw_exit_code else None


class GitProgressAnchor(ProgressAnchor):
    """No-test fallback coding progress from the git working diff (NOP-4).

    The level is the **size** of the cumulative diff versus the run's starting
    commit: ``insertions + deletions`` of the latest git evidence in the
    history. Size (not signed ``insertions - deletions``) is deliberate -- it
    credits in-place edits and deletions, which a net count flattens to zero, so
    a legit bug fix is not mistaken for a stall. A churn that rewrites the same
    zone leaves the diff size constant -> a flat level -> the CUSUM fires; real
    work that touches new ground grows the diff size -> new records -> silence.

    Like :class:`ExitCodeProgressAnchor`, it **defers to tests**: it abstains
    (``None``) whenever the history carries any test evidence, so it can never
    mask a stall the stronger test anchor would catch. This test-deferring shape
    was chosen from the NOP-4 experiment (see ``tests/test_nop4_git_experiment.py``):
    the naive always-on variant masked a genuine no-progress run, whereas the
    deferring variant strictly added detection (no-test churn) with no masking.

    It abstains (``None``) when the history carries no git evidence, so it has no
    opinion on a non-coding agent or an agent that does not report diff stats.
    ``net_lines`` is accepted as a degraded shortcut (size = its magnitude) when
    insertions/deletions are unavailable.

    Capture contract: the diff stat must be computed versus a **baseline frozen
    at run start**, not a moving ``HEAD`` -- otherwise an agent that commits
    mid-run would see its diff drop to ~0 (a spurious level collapse). This lives
    in the capture layer; the anchor only consumes the reported value.

    Known blind spot: git cannot tell *useful* from *useless* new code -- only
    ground truth (tests) can. When another anchor is applicable, the optimistic
    aggregation means git can only make the detector quieter, so there it can
    only add false negatives, not false positives. The exception is when git is
    the *sole* applicable anchor (an agent with no tests and no exit codes): a
    flat diff size then fires on its own, which is the intended catch for no-test
    churn but, on a genuinely-progressing run whose diff size happens to plateau,
    could be a false stop -- the residual cost of having any signal at all there.
    """

    name = "coding_git"

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        # Single pass: defer to the stronger ground truth (any test evidence ->
        # abstain immediately so git can never mask a test stall), otherwise keep
        # the latest git diff size.
        size: float | None = None
        for event in history:
            if _evidence(event, _TEST_KEYS) is not None:
                return None
            source = _evidence(event, _GIT_KEYS)
            if source is not None:
                size = _diff_size(source)
        return size


def _diff_size(source: Mapping[str, Any]) -> float:
    """Cumulative diff size from a git evidence mapping.

    ``insertions + deletions`` when present; else the magnitude of the degraded
    ``net_lines`` shortcut.
    """
    if _INSERTIONS in source or _DELETIONS in source:
        insertions = _as_int(source.get(_INSERTIONS, 0), default=0)
        deletions = _as_int(source.get(_DELETIONS, 0), default=0)
        return float(insertions + deletions)
    return float(abs(_as_int(source.get(_NET_LINES, 0), default=0)))


def coding_anchors() -> list[ProgressAnchor]:
    """The built-in coding anchors, ready to wire into ``NoProgressDetector``.

    Tests first (the stronger signal), exit codes as the fallback for agents that
    report process status but no test results, and the git diff (NOP-4) as the
    fallback for coding agents that edit but never run tests. The git anchor
    defers to tests, so it only speaks when no test evidence is present.
    """
    return [TestProgressAnchor(), ExitCodeProgressAnchor(), GitProgressAnchor()]


def progress_series(
    anchor: ProgressAnchor, history: Sequence[ToolEvent], window: int
) -> list[float | None]:
    """Progress level after each of the last ``window`` events.

    Entry ``i`` is ``anchor.measure`` over the prefix ending at that event, so a
    detector recovers the progress *series* over time. ``None`` entries mark
    events where the anchor has no opinion yet (no evidence so far). Pure: same
    history -> same series.
    """
    n = len(history)
    start = max(0, n - window)
    return [anchor.measure(history[: i + 1]) for i in range(start, n)]
