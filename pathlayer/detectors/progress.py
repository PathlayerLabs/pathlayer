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

PERF-2: ``measure``/``is_satisfied`` stay the pure specification -- an anchor's
observable behaviour is defined by them, forever. ``create_state``/``step``
(see :class:`AnchorState`) are an optional, *equivalent-by-construction* fold:
folding ``step`` one event at a time must reproduce, after each event, the same
``(level, satisfied)`` pair that calling ``measure``/``is_satisfied`` on the
prefix ending at that event would. The built-in coding anchors below override
both together for an O(1)-amortized-per-event fold (no more rescanning the
whole prefix on every tick); a custom anchor that implements only
``measure``/``is_satisfied`` (e.g. the NOP-4 experiment's prototype anchor)
keeps working unmodified via the default batch-rebuild fallback -- correct,
just O(n) per event for *that* anchor only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pathlayer.events import EventKind, ToolEvent

_PASSED = "passed"
_FAILED = "failed"
_EXIT_CODE = "exit_code"
_TEST_KEYS = (_PASSED, _FAILED)
_INSERTIONS = "insertions"
_DELETIONS = "deletions"
_NET_LINES = "net_lines"
_GIT_KEYS = (_INSERTIONS, _DELETIONS, _NET_LINES)


@dataclass(slots=True, frozen=True)
class AnchorSnapshot:
    """Incremental echo of ``(measure, is_satisfied)`` after one event (PERF-2).

    ``level`` mirrors ``measure(prefix)``, ``satisfied`` mirrors
    ``is_satisfied(prefix)``, for the prefix ending at the event that produced
    this snapshot via :meth:`AnchorState.advance`.
    """

    level: float | None
    satisfied: bool


class AnchorState(ABC):
    """Pure incremental state for one :class:`ProgressAnchor` (PERF-2).

    Immutable: :meth:`advance` returns a *new* state, never mutates ``self``,
    so a caller (``NoProgressDetector``, ``Pipeline``) can compute the next
    tick's state without touching the current one -- required for transactional
    commit (a later detector or the policy raising must leave every prior state
    untouched, exactly like ``Pipeline``'s existing ``HysteresisState``
    rollback). Same rules as :meth:`ProgressAnchor.measure`: no clock, no
    randomness, no network, no LLM.
    """

    @abstractmethod
    def advance(self, event: ToolEvent) -> tuple[AnchorState, AnchorSnapshot]:
        """Fold in one more event; return the next state and its snapshot."""
        raise NotImplementedError


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

    def create_state(self) -> Any:
        """Return a fresh incremental state for :meth:`step` (PERF-2).

        Default: an empty tuple, consumed by the default :meth:`step` below as
        the batch-rebuild fallback. A built-in coding anchor overriding both
        this and :meth:`step` returns an :class:`AnchorState` instead.
        """
        return ()

    def step(self, state: Any, event: ToolEvent) -> tuple[Any, AnchorSnapshot]:
        """Advance the incremental fold by one event; see the module docstring.

        Default (batch fallback): append ``event`` to the state history and call
        ``measure``/``is_satisfied`` on the result -- correct for any anchor,
        including ones that never heard of PERF-2, just O(n) per call for that
        one anchor.
        """
        history = (*state, event)
        level = self.measure(history)
        return history, AnchorSnapshot(level=level, satisfied=self.is_satisfied(history))


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


def _target_is_green(passed: int, failed: int) -> bool:
    """Per-suite success: no failures *and* a real (non-empty) run."""
    return failed == 0 and passed > 0


@dataclass(slots=True, frozen=True)
class _TestAnchorState(AnchorState):
    """Incremental state for :class:`TestProgressAnchor` (PERF-2).

    ``latest`` mirrors :func:`_latest_test_results`; ``total_level`` and
    ``red_count`` are running aggregates kept in lockstep with it so both
    ``measure`` (``total_level``) and ``is_satisfied`` (``red_count == 0`` with
    ``latest`` non-empty) are O(1) to read, updated in O(1) per event
    regardless of how many distinct suites have ever been seen.

    Stored as a ``tuple`` of ``(target, (passed, failed))`` pairs rather than
    a ``dict``: every other ``AnchorState`` in this module is frozen *and*
    hashable (all fields are immutable), and a ``dict`` field would silently
    break that invariant (``hash()`` raises ``TypeError`` on an unhashable
    field) the moment anything -- tests, memoization, a future ``set()`` of
    states -- treats these sibling states as interchangeable.
    """

    latest: tuple[tuple[str, tuple[int, int]], ...] = ()
    total_level: float = 0.0
    red_count: int = 0

    def advance(self, event: ToolEvent) -> tuple[AnchorState, AnchorSnapshot]:
        source = _evidence(event, _TEST_KEYS)
        if source is None:
            level = self.total_level if self.latest else None
            satisfied = bool(self.latest) and self.red_count == 0
            return self, AnchorSnapshot(level=level, satisfied=satisfied)
        passed = _as_int(source.get(_PASSED, 0), default=0)
        failed = _as_int(source.get(_FAILED, 0), default=0)
        target = _test_target(event)
        new_latest_dict = dict(self.latest)
        old = new_latest_dict.get(target)
        new_latest_dict[target] = (passed, failed)
        total_level = self.total_level
        red_count = self.red_count
        if old is not None:
            total_level -= old[0] - old[1]
            if not _target_is_green(old[0], old[1]):
                red_count -= 1
        total_level += passed - failed
        if not _target_is_green(passed, failed):
            red_count += 1
        new_state = _TestAnchorState(
            latest=tuple(new_latest_dict.items()),
            total_level=total_level,
            red_count=red_count,
        )
        satisfied = red_count == 0  # new_latest is non-empty: we just inserted a key.
        return new_state, AnchorSnapshot(level=total_level, satisfied=satisfied)


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

    def create_state(self) -> _TestAnchorState:
        return _TestAnchorState()

    def step(self, state: Any, event: ToolEvent) -> tuple[Any, AnchorSnapshot]:
        assert isinstance(state, _TestAnchorState)
        return state.advance(event)


@dataclass(slots=True, frozen=True)
class _ExitCodeAnchorState(AnchorState):
    """Incremental state for :class:`ExitCodeProgressAnchor` (PERF-2).

    ``saw_test`` is sticky (once ``True``, forever ``True``): the moment any
    event carries test evidence, ``measure`` defers to
    :class:`TestProgressAnchor` for the rest of the history -- exactly matching
    the batch implementation's per-call full-history scan, which returns
    ``None`` as soon as it meets *any* test evidence anywhere in the prefix.
    """

    saw_test: bool = False
    saw_exit_code: bool = False
    successes: int = 0

    def advance(self, event: ToolEvent) -> tuple[AnchorState, AnchorSnapshot]:
        if self.saw_test:
            return self, AnchorSnapshot(level=None, satisfied=False)
        if _evidence(event, _TEST_KEYS) is not None:
            deferred = _ExitCodeAnchorState(
                saw_test=True, saw_exit_code=self.saw_exit_code, successes=self.successes
            )
            return deferred, AnchorSnapshot(level=None, satisfied=False)
        source = _evidence(event, (_EXIT_CODE,))
        if source is None:
            level = float(self.successes) if self.saw_exit_code else None
            return self, AnchorSnapshot(level=level, satisfied=False)
        successes = self.successes + (1 if _as_int(source.get(_EXIT_CODE), default=-1) == 0 else 0)
        new_state = _ExitCodeAnchorState(saw_test=False, saw_exit_code=True, successes=successes)
        return new_state, AnchorSnapshot(level=float(successes), satisfied=False)


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

    def create_state(self) -> _ExitCodeAnchorState:
        return _ExitCodeAnchorState()

    def step(self, state: Any, event: ToolEvent) -> tuple[Any, AnchorSnapshot]:
        assert isinstance(state, _ExitCodeAnchorState)
        return state.advance(event)


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

    def create_state(self) -> _GitAnchorState:
        return _GitAnchorState()

    def step(self, state: Any, event: ToolEvent) -> tuple[Any, AnchorSnapshot]:
        assert isinstance(state, _GitAnchorState)
        return state.advance(event)


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


@dataclass(slots=True, frozen=True)
class _GitAnchorState(AnchorState):
    """Incremental state for :class:`GitProgressAnchor` (PERF-2).

    ``saw_test`` is sticky, mirroring :class:`_ExitCodeAnchorState`: the git
    anchor defers to tests forever once any test evidence is seen. Never
    satisfied (a non-empty diff means "code changed", not "task done"),
    matching the base :class:`ProgressAnchor.is_satisfied` default.
    """

    saw_test: bool = False
    saw_git: bool = False
    latest_size: float = 0.0

    def advance(self, event: ToolEvent) -> tuple[AnchorState, AnchorSnapshot]:
        if self.saw_test:
            return self, AnchorSnapshot(level=None, satisfied=False)
        if _evidence(event, _TEST_KEYS) is not None:
            deferred = _GitAnchorState(
                saw_test=True, saw_git=self.saw_git, latest_size=self.latest_size
            )
            return deferred, AnchorSnapshot(level=None, satisfied=False)
        source = _evidence(event, _GIT_KEYS)
        if source is None:
            level = self.latest_size if self.saw_git else None
            return self, AnchorSnapshot(level=level, satisfied=False)
        size = _diff_size(source)
        new_state = _GitAnchorState(saw_test=False, saw_git=True, latest_size=size)
        return new_state, AnchorSnapshot(level=size, satisfied=False)


def coding_anchors() -> list[ProgressAnchor]:
    """The built-in coding anchors, ready to wire into ``NoProgressDetector``.

    Tests first (the stronger signal), exit codes as the fallback for agents that
    report process status but no test results, and the git diff (NOP-4) as the
    fallback for coding agents that edit but never run tests. The git anchor
    defers to tests, so it only speaks when no test evidence is present.
    """
    return [TestProgressAnchor(), ExitCodeProgressAnchor(), GitProgressAnchor()]


def loop_suppression_anchors() -> list[ProgressAnchor]:
    """Anchors used to suppress ``loop`` on genuine progress (LOOP-2 precision).

    Deliberately **narrower** than :func:`coding_anchors`: only
    ``TestProgressAnchor`` and ``ExitCodeProgressAnchor``, never
    ``GitProgressAnchor``. Git diff size is a known-gameable signal (see
    :class:`GitProgressAnchor`'s docstring: "git cannot tell useful from useless
    new code") -- fine as a last-resort *catch* for ``NoProgressDetector``
    (optimistic aggregation means it can only add false negatives there, not
    false positives), but wrong as evidence to *suppress* a different detector.
    An agent that ping-pongs between two files while padding them with useless
    content would grow the git diff every turn, permanently suppressing
    ``loop`` on exactly the churn it exists to catch. Restricting to ground
    truth (tests, exit codes) keeps the suppression tied to the proven F1a bug
    without extending the blast radius to an anchor already flagged as weak.
    """
    return [TestProgressAnchor(), ExitCodeProgressAnchor()]


def is_advancing(anchors: Sequence[ProgressAnchor], history: Sequence[ToolEvent]) -> bool:
    """True iff the *last* event strictly improved at least one applicable anchor.

    Positive progress evidence, distinct from ``NoProgressDetector``'s silence:
    ``no_progress`` can stay at 0 confidence either because the agent is
    advancing (a new record) or simply because no anchor is applicable yet (no
    test/exit_code evidence in the history at all) -- the two are indistinguishable
    from that signal alone. ``is_advancing`` collapses only the first case to
    ``True``, so a caller that wants to know "is there *fresh* evidence of real
    progress right now" never mistakes silence-from-no-evidence for progress.

    Compares ``measure(history)`` to ``measure(history[:-1])`` directly --
    simpler and safer than reusing the CUSUM's cumulative-reset branch, which
    also resets on an anchor *losing* its baseline (a gap in evidence), not just
    on a genuine new record. A gap must never read as "advancing".

    Deliberately a strict boolean on the last step, not a continuous score: the
    caller only wants to act on evidence that is fresh *right now*. A loop that
    progressed once and now stagnates becomes suppressible again as soon as the
    next step stops setting records. Pure and deterministic: no clock, no
    randomness, no network, like every :class:`ProgressAnchor`.

    Filters to ``EventKind.TOOL_CALL`` first, matching the ``event_kinds``
    convention every action-only detector uses (see
    :class:`~pathlayer.detectors.base.Detector`): a non-action event (e.g. an
    ``LLM_CALL`` cost event) injected between two tool calls must never read
    as -- or hide -- a fresh record.
    """
    history = [event for event in history if event.kind is EventKind.TOOL_CALL]
    if not history:
        return False
    prefix = history[:-1]
    for anchor in anchors:
        current = anchor.measure(history)
        if current is None:
            continue
        previous = anchor.measure(prefix)
        if previous is not None and current > previous:
            return True
    return False


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
