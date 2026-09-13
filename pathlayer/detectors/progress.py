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
already carry -- aggregate test counts, optional stable per-test outcomes, and
process ``exit_code`` -- not text similarity. They are split so each anchor
stays on a *single* scale:
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
whole prefix on every tick). Aggregate updates are O(1) for a fixed target set;
NOP-5 rich updates are O(K) in unique test IDs because immutable debt ledgers
are copied transactionally. A custom anchor that implements only
``measure``/``is_satisfied`` (e.g. the NOP-4 experiment's prototype anchor)
keeps working unmodified via the default batch-rebuild fallback -- correct,
just O(n) per event for *that* anchor only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pathlayer.events import EventKind, ToolEvent

_PASSED = "passed"
_FAILED = "failed"
_TEST_OUTCOMES = "test_outcomes"
_SCOPE_ID = "scope_id"
_COLLECTION_COMPLETE = "collection_complete"
_SCOPE_EXHAUSTIVE = "scope_exhaustive"
_EXIT_CODE = "exit_code"
_TEST_KEYS = (_PASSED, _FAILED)
_TEST_STATUSES = frozenset({"passed", "failed", "skipped", "error", "flaky"})
_DEBT_STATUSES = frozenset({"failed", "error", "flaky"})
# Rich payloads are untrusted adapter input. Bounding both one payload and the
# cumulative per-target ledger prevents ever-new IDs from turning immutable
# transactional copies into unbounded memory growth / quadratic-time DoS.
_MAX_TRACKED_TEST_IDS_PER_TARGET = 10_000
_MAX_TRACKED_TEST_IDS_TOTAL = 10_000
_MAX_RICH_TEST_TARGETS = 256
_MAX_RICH_SCOPES_PER_TARGET = 256
_MAX_SCOPE_STATUS_ENTRIES_TOTAL = 10_000
_MAX_SCOPE_ID_LENGTH = 512
# Guarded reversibility (NOP-5 follow-up): a target's rich mode is sticky
# across a single aggregate-only event (one malformed/degraded producer tick
# must not erase a whole debt ledger), but not forever -- this many
# *consecutive* aggregate-only events for the same target revert it to plain
# legacy tracking. Modeled on ARB-3's escalate/de-escalate hysteresis
# (confirmation by repetition, not by one tick). This reopens a bounded,
# documented re-count risk: see ``_TestAnchorState``'s docstring.
_LEGACY_REVERT_STREAK = 3
_INSERTIONS = "insertions"
_DELETIONS = "deletions"
_NET_LINES = "net_lines"
_GIT_KEYS = (_INSERTIONS, _DELETIONS, _NET_LINES)


@dataclass(slots=True, frozen=True)
class AnchorSnapshot:
    """Incremental echo of ``(measure, is_satisfied)`` after one event (PERF-2).

    ``level`` mirrors ``measure(prefix)``, ``satisfied`` mirrors
    ``is_satisfied(prefix)``, for the prefix ending at the event that produced
    this snapshot via :meth:`AnchorState.advance`. ``diagnostics`` mirrors
    :meth:`ProgressAnchor.diagnose` for that same prefix: human-readable notes
    about anything degraded-but-not-fatal at *this* event (a malformed payload
    dropped, a capacity cap reached, a sticky mode reverting) -- empty in the
    common case, never load-bearing for ``level``/``satisfied``. ``unavailable``
    mirrors :meth:`ProgressAnchor.is_unavailable`.
    """

    level: float | None
    satisfied: bool
    diagnostics: tuple[str, ...] = ()
    unavailable: bool = False


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

    def diagnose(self, history: Sequence[ToolEvent]) -> tuple[str, ...]:
        """Human-readable notes on anything degraded-but-not-fatal (optional).

        Default: nothing to report, ever. A concrete anchor overrides this
        (alongside populating ``AnchorSnapshot.diagnostics`` in its incremental
        fold) to surface conditions that are otherwise silently absorbed -- a
        malformed payload dropped, a capacity cap reached, a sticky mode
        reverting -- instead of just quietly changing ``measure``'s behaviour.
        Pure and deterministic like ``measure``/``is_satisfied``: same history
        in, same diagnostics out, describing only the *last* event of
        ``history`` (equivalent-by-construction to the ``diagnostics`` field of
        the ``AnchorSnapshot`` :meth:`step` returns for that same prefix,
        verified prefix-by-prefix like every other PERF-2 fold).
        """
        return ()

    def is_unavailable(self, history: Sequence[ToolEvent]) -> bool:
        """Whether the anchor was measuring and lost its measurement at the last event.

        Not the same as ``measure(history) is None``, which also means "never
        applicable" (no evidence of this kind at all): an unavailable anchor
        had a baseline, and its producer reported that it observed nothing
        this time (ANC-4's ``progress_proof_unobserved``). ``measure`` must
        then return ``None``. :class:`NoProgressDetector` holds itself silent
        on such an event whatever its other anchors say: this anchor's
        progress may be exactly what was covering a co-anchor's flat level, so
        losing it must never, on its own, stop an agent. Default ``False``.
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
        return history, AnchorSnapshot(
            level=level,
            satisfied=self.is_satisfied(history),
            diagnostics=self.diagnose(history),
            unavailable=self.is_unavailable(history),
        )


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


def _has_test_evidence(event: ToolEvent) -> bool:
    """Whether an event carries usable aggregate or per-test ground truth.

    A malformed rich claim is inert rather than sticky: it cannot disable the
    weaker anchors for the rest of the run. Valid aggregate evidence in the
    same event still counts and is consumed by the legacy fallback.
    """
    return _evidence(event, _TEST_KEYS) is not None or _rich_test_result(event) is not None


@dataclass(slots=True, frozen=True)
class _RichTestResult:
    """Validated per-test result carried by one tool event (NOP-5)."""

    outcomes: tuple[tuple[str, str], ...]
    scope_id: str
    collection_complete: bool
    scope_exhaustive: bool


def _rich_test_result(event: ToolEvent) -> _RichTestResult | None:
    """Parse the optional NOP-5 per-test contract, or reject it atomically."""
    source = _evidence(event, (_TEST_OUTCOMES,))
    if source is None:
        return None
    raw_outcomes = source.get(_TEST_OUTCOMES)
    scope_id = source.get(_SCOPE_ID)
    collection_complete = source.get(_COLLECTION_COMPLETE)
    scope_exhaustive = source.get(_SCOPE_EXHAUSTIVE)
    if (
        not isinstance(raw_outcomes, Mapping)
        or len(raw_outcomes) > _MAX_TRACKED_TEST_IDS_PER_TARGET
        or not isinstance(scope_id, str)
        or not scope_id
        or len(scope_id) > _MAX_SCOPE_ID_LENGTH
        or not isinstance(collection_complete, bool)
        or not isinstance(scope_exhaustive, bool)
    ):
        return None
    outcomes: list[tuple[str, str]] = []
    for raw_test_id, raw_status in raw_outcomes.items():
        if (
            not isinstance(raw_test_id, str)
            or not raw_test_id
            or not isinstance(raw_status, str)
            or raw_status not in _TEST_STATUSES
        ):
            return None
        outcomes.append((raw_test_id, raw_status))
    return _RichTestResult(
        outcomes=tuple(outcomes),
        scope_id=scope_id,
        collection_complete=collection_complete,
        scope_exhaustive=scope_exhaustive,
    )


def _test_target(event: ToolEvent) -> str:
    """Key identifying which test suite an event reports on.

    Uses the explicit ``target`` arg when present, else the tool name. This is
    what lets the level be summed *per suite* instead of comparing the
    passed-minus-failed of one suite against another (which would manufacture a
    huge spurious delta when the agent switches suites)."""
    target = event.args.get("target")
    return str(target) if target is not None else event.tool


def _target_is_green(passed: int, failed: int) -> bool:
    """Per-suite success: no failures *and* a real (non-empty) run."""
    return failed == 0 and passed > 0


@dataclass(slots=True, frozen=True)
class _RichTargetState:
    """Immutable per-target debt ledger for rich test outcomes (NOP-5)."""

    legacy_offset: float
    debts: frozenset[str] = frozenset()
    credited: frozenset[str] = frozenset()
    scope_statuses: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    satisfied: bool = False
    overflowed: bool = False
    # Consecutive aggregate-only events seen *while in rich mode* (guarded
    # reversibility, NOP-5 follow-up). Any rich event resets this to 0 (a
    # fresh ``_RichTargetState`` literal below always defaults it back to 0),
    # matching ARB-3's hysteresis: only a *streak* commits the mode change.
    legacy_streak: int = 0


@dataclass(slots=True, frozen=True)
class _TestAnchorState(AnchorState):
    """Immutable mixed legacy/rich state for :class:`TestProgressAnchor`.

    Rich mode is sticky *per target*, but guarded-reversible (NOP-5
    follow-up): see ``_LEGACY_REVERT_STREAK`` and :class:`TestProgressAnchor`'s
    docstring. A target's debt ledger is keyed by stable test id, never by
    scope, so overlapping selections cannot double-credit a repair.
    Tuple/frozenset fields keep the state frozen and hashable for
    transactional PERF-2 rollback. Updating a rich event costs O(K) time and
    memory in the number of unique test ids observed, never in history length.
    """

    legacy_latest: tuple[tuple[str, tuple[int, int]], ...] = ()
    rich_targets: tuple[tuple[str, _RichTargetState], ...] = ()
    rich_target_overflowed: bool = False

    def _snapshot(self, *, diagnostics: tuple[str, ...] = ()) -> AnchorSnapshot:
        legacy = dict(self.legacy_latest)
        rich = dict(self.rich_targets)
        targets = set(legacy) | set(rich)
        if not targets:
            return AnchorSnapshot(level=None, satisfied=False, diagnostics=diagnostics)
        level = 0.0
        satisfied = not self.rich_target_overflowed
        # Stable ordering is part of detector determinism: floating-point
        # accumulation must not depend on Python's per-process hash seed.
        for target in sorted(targets):
            rich_state = rich.get(target)
            if rich_state is not None:
                level += rich_state.legacy_offset + float(len(rich_state.credited))
                satisfied = satisfied and rich_state.satisfied
                continue
            passed, failed = legacy[target]
            level += passed - failed
            satisfied = satisfied and _target_is_green(passed, failed)
        return AnchorSnapshot(level=level, satisfied=satisfied, diagnostics=diagnostics)

    def advance(self, event: ToolEvent) -> tuple[_TestAnchorState, AnchorSnapshot]:
        target = _test_target(event)
        rich_result = _rich_test_result(event)
        rich_targets = dict(self.rich_targets)
        if rich_result is not None:
            legacy_latest = dict(self.legacy_latest)
            current = rich_targets.get(target)
            if current is None:
                if len(rich_targets) >= _MAX_RICH_TEST_TARGETS:
                    overflowed_state = _TestAnchorState(
                        legacy_latest=self.legacy_latest,
                        rich_targets=self.rich_targets,
                        rich_target_overflowed=True,
                    )
                    diagnostic = (
                        f"target {target!r}: rich test-target capacity "
                        f"({_MAX_RICH_TEST_TARGETS}) reached; new target ignored"
                    )
                    return overflowed_state, overflowed_state._snapshot(diagnostics=(diagnostic,))
                previous = legacy_latest.get(target)
                offset = float(previous[0] - previous[1]) if previous is not None else 0.0
                current = _RichTargetState(legacy_offset=offset)
            debts = set(current.debts)
            credited = set(current.credited)
            previous_scope_statuses = {
                scope_id: dict(statuses) for scope_id, statuses in current.scope_statuses
            }
            scope_statuses = dict(current.scope_statuses)
            scope_is_tracked = rich_result.scope_id in scope_statuses
            scope_can_be_tracked = (
                scope_is_tracked
                or len(scope_statuses) < _MAX_RICH_SCOPES_PER_TARGET
                or (rich_result.collection_complete and rich_result.scope_exhaustive)
            )
            if not scope_can_be_tracked:
                overflowed_target = _RichTargetState(
                    legacy_offset=current.legacy_offset,
                    debts=current.debts,
                    credited=current.credited,
                    scope_statuses=current.scope_statuses,
                    satisfied=False,
                    overflowed=True,
                )
                rich_targets[target] = overflowed_target
                overflowed_state = _TestAnchorState(
                    legacy_latest=self.legacy_latest,
                    rich_targets=tuple(rich_targets.items()),
                    rich_target_overflowed=self.rich_target_overflowed,
                )
                diagnostic = (
                    f"target {target!r}: rich scope capacity "
                    f"({_MAX_RICH_SCOPES_PER_TARGET}) reached; scope "
                    f"{rich_result.scope_id!r} dropped"
                )
                return overflowed_state, overflowed_state._snapshot(diagnostics=(diagnostic,))
            comparable_debts = {
                test_id
                for statuses in previous_scope_statuses.values()
                for test_id, status in statuses.items()
                if status in _DEBT_STATUSES
            }
            raw_outcomes = dict(rich_result.outcomes)
            known_ids = (
                {test_id for statuses in previous_scope_statuses.values() for test_id in statuses}
                | debts
                | credited
            )
            total_known_ids = sum(
                len(
                    set(rich_state.debts)
                    | set(rich_state.credited)
                    | {
                        test_id
                        for _, statuses in rich_state.scope_statuses
                        for test_id, _ in statuses
                    }
                )
                for rich_state in rich_targets.values()
            )
            available = min(
                _MAX_TRACKED_TEST_IDS_PER_TARGET - len(known_ids),
                _MAX_TRACKED_TEST_IDS_TOTAL - total_known_ids,
            )
            previous_scope = dict(scope_statuses.get(rich_result.scope_id, ()))
            next_scope = {} if rich_result.collection_complete else dict(previous_scope)
            total_scope_entries = sum(
                len(statuses)
                for rich_state in rich_targets.values()
                for _, statuses in rich_state.scope_statuses
            )
            if rich_result.collection_complete and rich_result.scope_exhaustive:
                total_scope_entries -= sum(len(statuses) for _, statuses in current.scope_statuses)
            elif rich_result.collection_complete:
                total_scope_entries -= len(previous_scope)
            available_scope_entries = max(0, _MAX_SCOPE_STATUS_ENTRIES_TOTAL - total_scope_entries)
            available_new_ids = max(0, available)
            outcomes: dict[str, str] = {}
            overflowed = current.overflowed
            dropped_outcomes = 0
            for test_id, status in raw_outcomes.items():
                is_new_id = test_id not in known_ids
                consumes_scope_entry = test_id not in next_scope
                if (is_new_id and available_new_ids == 0) or (
                    consumes_scope_entry and available_scope_entries == 0
                ):
                    overflowed = True
                    dropped_outcomes += 1
                    continue
                outcomes[test_id] = status
                if is_new_id:
                    known_ids.add(test_id)
                    available_new_ids -= 1
                if consumes_scope_entry:
                    available_scope_entries -= 1
            for test_id, status in outcomes.items():
                if status in _DEBT_STATUSES:
                    debts.add(test_id)
                elif (
                    not overflowed
                    and status == _PASSED
                    and test_id in comparable_debts
                    and test_id in debts
                ):
                    credited.add(test_id)

            next_scope.update(outcomes)
            if rich_result.collection_complete and rich_result.scope_exhaustive:
                # A whole-target observation supersedes every partial view:
                # omitted IDs lose comparability in every prior scope.
                scope_statuses.clear()
            scope_statuses[rich_result.scope_id] = tuple(next_scope.items())
            satisfied = (
                not overflowed
                and rich_result.collection_complete
                and rich_result.scope_exhaustive
                and bool(outcomes)
                and all(status == _PASSED for status in outcomes.values())
                and all(outcomes.get(test_id) == _PASSED for test_id in debts)
            )
            rich_targets[target] = _RichTargetState(
                legacy_offset=current.legacy_offset,
                debts=frozenset(debts),
                credited=frozenset(credited),
                scope_statuses=tuple(scope_statuses.items()),
                satisfied=satisfied,
                overflowed=overflowed,
            )
            new_state = _TestAnchorState(
                legacy_latest=self.legacy_latest,
                rich_targets=tuple(rich_targets.items()),
                rich_target_overflowed=self.rich_target_overflowed,
            )
            diagnostics: tuple[str, ...] = ()
            if dropped_outcomes:
                # Covers both cap causes in the loop above (new test-id capacity
                # or scope-entry capacity exhausted) -- either way the operator
                # needs the same signal: some outcomes in this event were not
                # recorded.
                plural = "s" if dropped_outcomes != 1 else ""
                diagnostics = (
                    f"target {target!r}: rich outcome capacity reached; "
                    f"{dropped_outcomes} outcome{plural} dropped this event",
                )
            return new_state, new_state._snapshot(diagnostics=diagnostics)

        # A malformed rich-only claim is inert. If valid aggregate counts share
        # the event, continue below through the backward-compatible fallback.
        if _evidence(event, (_TEST_OUTCOMES,)) is not None and _evidence(event, _TEST_KEYS) is None:
            diagnostic = f"target {target!r}: malformed test_outcomes payload ignored"
            return self, self._snapshot(diagnostics=(diagnostic,))

        source = _evidence(event, _TEST_KEYS)
        if source is None:
            return self, self._snapshot()
        passed = _as_int(source.get(_PASSED, 0), default=0)
        failed = _as_int(source.get(_FAILED, 0), default=0)
        # A malformed test_outcomes payload can coexist with valid aggregate
        # counts in the same event (the rich half fails validation but the
        # legacy half is still usable). The branch above only catches a
        # malformed rich claim with *no* aggregate fallback; this one must not
        # let the dropped rich half vanish silently just because the event is
        # still usable through the legacy path below.
        malformed_diagnostics: tuple[str, ...] = (
            (
                f"target {target!r}: malformed test_outcomes payload ignored "
                "(aggregate counts still applied)",
            )
            if _evidence(event, (_TEST_OUTCOMES,)) is not None
            else ()
        )
        current_rich = rich_targets.get(target)
        if current_rich is not None:
            # Legacy evidence while the target is in rich mode: guarded
            # reversion (NOP-5 follow-up). A lone aggregate-only event never
            # flips the mode -- see ``_LEGACY_REVERT_STREAK``'s docstring --
            # only _LEGACY_REVERT_STREAK consecutive ones do, discarding the
            # rich debt ledger and falling back to the plain aggregate.
            streak = current_rich.legacy_streak + 1
            if streak < _LEGACY_REVERT_STREAK:
                rich_targets[target] = replace(current_rich, legacy_streak=streak)
                diagnostic = (
                    f"target {target!r}: aggregate test evidence ignored while "
                    f"in rich mode ({streak}/{_LEGACY_REVERT_STREAK} before revert)"
                )
                new_state = _TestAnchorState(
                    legacy_latest=self.legacy_latest,
                    rich_targets=tuple(rich_targets.items()),
                    rich_target_overflowed=self.rich_target_overflowed,
                )
                return new_state, new_state._snapshot(
                    diagnostics=(diagnostic, *malformed_diagnostics)
                )
            del rich_targets[target]
            legacy_latest = dict(self.legacy_latest)
            legacy_latest[target] = (passed, failed)
            diagnostic = (
                f"target {target!r}: reverted to aggregate test evidence after "
                f"{_LEGACY_REVERT_STREAK} consecutive aggregate-only events; "
                "rich debt ledger discarded"
            )
            new_state = _TestAnchorState(
                legacy_latest=tuple(legacy_latest.items()),
                rich_targets=tuple(rich_targets.items()),
                rich_target_overflowed=self.rich_target_overflowed,
            )
            return new_state, new_state._snapshot(diagnostics=(diagnostic, *malformed_diagnostics))
        legacy_latest = dict(self.legacy_latest)
        legacy_latest[target] = (passed, failed)
        new_state = _TestAnchorState(
            legacy_latest=tuple(legacy_latest.items()),
            rich_targets=self.rich_targets,
            rich_target_overflowed=self.rich_target_overflowed,
        )
        return new_state, new_state._snapshot(diagnostics=malformed_diagnostics)


def _test_snapshot(history: Sequence[ToolEvent]) -> AnchorSnapshot:
    state = _TestAnchorState()
    snapshot = AnchorSnapshot(level=None, satisfied=False)
    for event in history:
        state, snapshot = state.advance(event)
    return snapshot


class TestProgressAnchor(ProgressAnchor):
    """Ground-truth coding progress from aggregate or per-test results.

    Legacy targets retain their latest ``passed - failed`` contribution. A
    target switches permanently to NOP-5 rich mode when it emits a valid
    ``test_outcomes`` payload: its prior contribution becomes a fixed offset,
    then each stable test debt contributes exactly once on its first comparable
    transition from ``failed``/``error``/``flaky`` to ``passed``. Deletion,
    renaming, skipping and repeated oscillation never manufacture progress.

    Success is deliberately separate from this monotone credit: a rich target
    is satisfied only after a complete, exhaustive, all-green observation that
    explicitly contains every debt it has ever exposed.

    Rich mode is *sticky but not permanent* per target (NOP-5 follow-up): one
    aggregate-only event while in rich mode is inert (a lone malformed or
    degraded producer tick must not erase a whole debt ledger), but
    ``_LEGACY_REVERT_STREAK`` *consecutive* aggregate-only events revert the
    target to plain legacy tracking, discarding its rich ledger. A later rich
    event restarts rich mode with a fresh offset from the latest legacy value.
    This bounds, but does not eliminate, a re-count risk: a producer that
    deliberately oscillates rich and aggregate data for the same target across
    a revert could re-credit a debt it already fixed once, because reverting
    forgets which debts were already resolved. Malformed payloads, capacity
    caps and mode reversions are never silent: see :meth:`diagnose`.
    """

    name = "coding_tests"

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        return _test_snapshot(history).level

    def is_satisfied(self, history: Sequence[ToolEvent]) -> bool:
        return _test_snapshot(history).satisfied

    def diagnose(self, history: Sequence[ToolEvent]) -> tuple[str, ...]:
        return _test_snapshot(history).diagnostics

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
        if _has_test_evidence(event):
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
            if _has_test_evidence(event):
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
            if _has_test_evidence(event):
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
        if _has_test_evidence(event):
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
