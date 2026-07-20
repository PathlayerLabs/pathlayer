"""NoProgressDetector (class ii: no-progress-without-repetition).

The differentiated case (the 11-day / ~$47k loop): every step looks different
yet nothing advances, so repetition detectors stay silent. This detector
aggregates pluggable :class:`~pathlayer.detectors.progress.ProgressAnchor`s: it
recovers each anchor's progress level over the recent window and fires when the
agent stays active without setting a new progress *record*.

It references no concrete anchor -- the anchors are injected -- so the same
detector serves any agent and only the anchors are domain-specific. An anchor
that sees none of its evidence abstains (``measure`` -> ``None``); if *no*
anchor is applicable the detector is silent (confidence 0). We never stop an
agent we cannot measure -- a false stop is worse than no Pathlayer at all.

NOP-2 scores the progress series with a CUSUM (Page, 1954) reformulated over
*records* rather than raw deltas: each step that fails to beat the best level
seen so far adds ``target_rate`` to a cumulative sum ``C``; a new record resets
``C`` to 0. The detector fires once ``C`` reaches ``threshold``. Compared to a
delta CUSUM, a single regression (TDD red, a growing suite) is just "no new
record" (one ``+target_rate`` step), so it never triggers an instant stop -- the
full patience must elapse before firing.

NOP-3 makes this signal *actionable* (it can STOP). The piece that earlier kept
it advisory was telling "done" from "stuck": a flat level after success looks
like a stall. That is now resolved by anchor success suppression -- the detector
stays silent while any applicable anchor reports ``is_satisfied`` (e.g. a green
test suite), so a succeeded agent is never stopped. A runaway that keeps acting
*after* success is a cost/resource concern (``BudgetDetector`` /
``ResourceDetector``), not a no-progress one.

The "activity-without-progress ratio" sketched in the NOP-3 issue is not a
separate mechanism here: the record-based CUSUM already counts the active steps
that fail to advance, so a second ratio detector would be redundant under sane
thresholds. The issue's intent is met by the CUSUM.

The git anchor (NOP-4, ``GitProgressAnchor``) extends coverage to coding agents
that never run tests: it credits the size of the working diff and catches churn
the test/exit-code anchors cannot see. It *defers to tests* (abstains while any
test evidence is present), so it can never mask a real stall the test anchor
would catch -- this test-deferring shape was chosen from a labeled-trace
experiment (``tests/test_nop4_git_experiment.py``) where the naive always-on
variant masked a genuine no-progress run.

Known limitation: the signal is only as good as its anchor. Two residual blind
spots: (1) git cannot tell *useful* from *useless* new code -- only ground truth
(tests) can; whenever a stronger anchor is also applicable the optimistic
aggregation lets git only quiet the detector (false negatives), not fire it. The
one place git can fire alone is an agent with no tests and no exit codes, where a
flat diff size could in principle false-stop a genuinely-progressing run -- the
cost of having any signal there at all. (2) A long flat-red test suite while the
agent genuinely writes code can still fire, because a constant ``passed - failed``
is indistinguishable from a stall. That second one is a test-anchor/patience
concern (a real progressing run shows the failing count *dropping*, which already
sets records and stays silent), not something git should paper over -- using git
to silence the test anchor is exactly the masking the deferring design avoids.

PERF-2 -- incremental fold: ``evaluate`` above stays the pure specification for
every anchor actually shipped here. :meth:`create_state`/:meth:`step` below are
a *faster equivalent* implementation of that same contract, verified
prefix-by-prefix against ``evaluate`` (``tests/test_perf2_incremental.py``),
not just asserted -- the one deliberate, documented exception being a `NaN`
level from a pathological custom anchor (see ``_cusum``'s and
``_push_level``'s matching guard below): PERF-2 also *defines* that edge case
(normalized to a gap), which no built-in anchor can ever produce, rather than
leaving it as underspecified batch behaviour. The motivation for the fold
itself is complexity, not behaviour: ``evaluate`` re-derives every anchor's
level from scratch and calls :func:`progress_series` (itself ``window`` calls to
``measure``, each rescanning its prefix) on *every* tick, i.e. O(window * n) per
tick / O(window * n^2) total over a run of length n -- measured at 15-16x
slower on a 4,000-event OpenClaw reproduction (issue #76). ``step`` instead
folds each anchor's ``(level, satisfied)`` via :meth:`ProgressAnchor.step` (O(1)
amortized for the three built-in coding anchors, O(n) only for a
non-conforming custom anchor -- see ``pathlayer.detectors.progress``) and
tracks the sliding-window running-maximum with a monotone deque bounded by
``window`` (see :func:`_push_level`), turning the whole tick into O(window) --
a fixed constant -- so a full run is O(window * n), linear in ``n``.

The deque preserves *ties*: the running maximum's leftmost occurrence survives
a push of an equal value, matching ``_cusum``'s own tie-breaking exactly --
equal to ``best`` never resets ``best``, so the maximum's first index in the
window is what the exact CUSUM distance is measured from (proof: once the
window's true maximum is reached, ``best`` locks onto it -- nothing later can
exceed a maximum -- so every subsequent step is "no new record", each adding
``target_rate``; the count of such steps is exactly "positions after the
maximum's first occurrence", which is what the deque's head index gives in
O(1) after eviction). ``_stagnation_table`` mirrors ``_cusum``'s *repeated
addition* (not a single multiplication) so the two paths round identically in
floating point.

Built-in anchors never emit ``NaN``/``inf`` levels, but a custom one could:
``NaN`` is normalized to a gap (treated like ``None``) in both ``_cusum`` and
``_push_level``, matching the NaN-is-inert convention ``Policy.fuse`` already
uses (``pathlayer/policies/base.py``) -- required for the deque, not just
cosmetic, since ``NaN`` fails every float comparison and would otherwise get
stuck in the deque, permanently pinning a stale head index. ``+inf``/``-inf``
need no special case: they compare normally (a total order holds), so ties and
evictions work exactly like any other float.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.progress import ProgressAnchor, progress_series
from pathlayer.events import EventKind, ToolEvent


def _stagnation_table(window: int, target_rate: float) -> tuple[float, ...]:
    """Precompute ``target_rate`` summed by repeated addition, 0..window-1 times.

    Matches ``_cusum``'s own repeated ``cumulative += target_rate`` bit-for-bit
    (a single ``target_rate * k`` multiplication can round differently).
    """
    sums = [0.0]
    total = 0.0
    for _ in range(window - 1):
        total += target_rate
        sums.append(total)
    return tuple(sums)


def _push_level(
    entries: tuple[tuple[int, float], ...],
    count: int,
    window: int,
    level: float | None,
) -> tuple[tuple[tuple[int, float], ...], int, int]:
    """Pure step of one anchor's sliding-window running-maximum (PERF-2).

    ``entries`` is a monotone-decreasing-with-ties deque of ``(index, level)``,
    oldest first; ``count`` is the number of relevant events fed so far
    (0-indexed position of the *next* push). Returns ``(new_entries, new_count,
    stalled)`` where ``stalled`` is the number of steps since the window's
    current running maximum was first set -- multiply by ``target_rate`` (via
    :func:`_stagnation_table`) for the CUSUM contribution. A ``None`` level
    (anchor not applicable at this position) clears the deque, mirroring
    ``_cusum``'s own reset-on-``None`` -- the next non-``None`` push starts a
    fresh baseline, exactly as a fresh call to ``progress_series``/``_cusum``
    would compute for the equivalent windowed batch series.

    Immutable and O(window): never mutates ``entries``, and every operation is
    bounded by ``window`` regardless of ``count`` -- this is what turns the
    O(window * n) batch rescan per tick into O(window) per tick.
    """
    if level is None or math.isnan(level):
        # A NaN level is treated exactly like a gap (no signal), matching
        # ``_cusum`` below and the NaN-is-inert convention already used by
        # ``Policy.fuse`` (pathlayer/policies/base.py). This is load-bearing,
        # not cosmetic: the deque's pop-while-less-than invariant relies on a
        # total order, and NaN fails every comparison in both directions --
        # appending it unconditionally (as the ``level is None`` guard alone
        # would let happen) corrupts the deque so a later, larger level no
        # longer evicts it, permanently pinning a stale head index.
        return (), count + 1, 0
    index = count
    kept = list(entries)
    while kept and kept[-1][1] < level:
        kept.pop()
    kept.append((index, level))
    new_count = count + 1
    cutoff = new_count - window
    while kept and kept[0][0] < cutoff:
        kept.pop(0)
    stalled = (new_count - 1) - kept[0][0]
    return tuple(kept), new_count, stalled


@dataclass(slots=True, frozen=True)
class _AnchorRuntime:
    """Immutable per-anchor incremental state inside ``NoProgressDetector`` (PERF-2)."""

    anchor_state: object
    entries: tuple[tuple[int, float], ...]
    count: int


@dataclass(slots=True, frozen=True)
class _NoProgressState:
    """Immutable incremental state for ``NoProgressDetector.step`` (PERF-2).

    ``last_signal`` lets a non-``TOOL_CALL`` event (e.g. an LLM-cost event)
    pass through unchanged in O(1), matching ``evaluate``'s own filtering: the
    filtered history simply does not grow on such an event, so re-evaluating it
    would reproduce the exact same signal anyway.
    """

    runtimes: tuple[_AnchorRuntime, ...]
    last_signal: Signal


class NoProgressDetector(Detector):
    name = "no_progress"
    # Reason over agent actions only: LLM-cost events carry no progress evidence
    # and would each count as a stagnant step, inflating the CUSUM toward a false
    # stop now that this signal is actionable.
    event_kinds = frozenset({EventKind.TOOL_CALL})

    def __init__(
        self,
        anchors: Sequence[ProgressAnchor],
        target_rate: float = 0.2,
        threshold: float = 2.0,
        window: int = 64,
    ) -> None:
        self._anchors = list(anchors)
        if not self._anchors:
            raise ValueError("NoProgressDetector needs at least one anchor")
        if target_rate <= 0:
            raise ValueError("target_rate must be > 0")
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        # The first windowed level only sets the baseline (no increment), so a
        # window of W contributes at most W-1 stagnant steps, i.e. C <= (W-1)*k.
        # Requiring (W-1)*k >= h keeps confidence 1.0 reachable; a shorter window
        # would cap C below threshold and make the stop unreachable.
        min_window = math.ceil(threshold / target_rate) + 1
        if window < min_window:
            raise ValueError(f"window must be >= ceil(threshold/target_rate) + 1 = {min_window}")
        self._target_rate = target_rate
        self._threshold = threshold
        self._window = window
        self._stagnation_table = _stagnation_table(window, target_rate)

    def create_state(self) -> _NoProgressState:
        runtimes = tuple(
            _AnchorRuntime(anchor_state=anchor.create_state(), entries=(), count=0)
            for anchor in self._anchors
        )
        return _NoProgressState(runtimes=runtimes, last_signal=Signal(self.name, confidence=0.0))

    def step(self, state: _NoProgressState, event: ToolEvent) -> tuple[_NoProgressState, Signal]:
        if event.kind not in self.event_kinds:
            return state, state.last_signal
        next_runtimes: list[_AnchorRuntime] = []
        applicable_cusums: list[float] = []
        any_satisfied = False
        for anchor, runtime in zip(self._anchors, state.runtimes, strict=True):
            new_anchor_state, snapshot = anchor.step(runtime.anchor_state, event)
            new_entries, new_count, stalled = _push_level(
                runtime.entries, runtime.count, self._window, snapshot.level
            )
            next_runtimes.append(_AnchorRuntime(new_anchor_state, new_entries, new_count))
            if snapshot.level is not None:
                applicable_cusums.append(self._stagnation_table[stalled])
                if snapshot.satisfied:
                    any_satisfied = True
        if not applicable_cusums or any_satisfied:
            signal = Signal(self.name, confidence=0.0)
        else:
            cumulative = min(applicable_cusums)
            if cumulative <= 0.0:
                signal = Signal(self.name, confidence=0.0)
            else:
                confidence = min(1.0, cumulative / self._threshold)
                stalled_steps = round(cumulative / self._target_rate)
                plural = "s" if stalled_steps != 1 else ""
                reason = f"{stalled_steps} step{plural} without new progress (cusum)"
                signal = Signal(self.name, confidence=confidence, reason=reason)
        return _NoProgressState(runtimes=tuple(next_runtimes), last_signal=signal), signal

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        history = self._relevant(history)
        # An anchor is applicable iff it has an opinion on the full history.
        applicable = [anchor for anchor in self._anchors if anchor.measure(history) is not None]
        if not applicable:
            # No anchor can measure this agent -> stay silent.
            return Signal(self.name, confidence=0.0)
        # Success suppression: if any applicable anchor is at its success state
        # (e.g. a green test suite), a flat level is "nothing left to fix", not a
        # stall -- stay silent so a succeeded agent is never stopped. This is the
        # brake that makes the no-progress signal safe to act on (STOP).
        if any(anchor.is_satisfied(history) for anchor in applicable):
            return Signal(self.name, confidence=0.0)
        # One CUSUM per applicable anchor; take the most optimistic (smallest):
        # if any anchor still sets records the agent is advancing.
        cumulative = min(
            self._cusum(progress_series(anchor, history, self._window)) for anchor in applicable
        )
        if cumulative <= 0.0:
            return Signal(self.name, confidence=0.0)
        confidence = min(1.0, cumulative / self._threshold)
        stalled = round(cumulative / self._target_rate)
        plural = "s" if stalled != 1 else ""
        reason = f"{stalled} step{plural} without new progress (cusum)"
        return Signal(self.name, confidence=confidence, reason=reason)

    def _cusum(self, series: Sequence[float | None]) -> float:
        """CUSUM over progress *records*.

        Walks the level series; a new record (``level > best``) resets the
        cumulative sum, otherwise each step adds ``target_rate``. A ``None`` gap
        (the anchor lost its baseline) resets too: there is no record to beat.
        """
        best: float | None = None
        cumulative = 0.0
        for level in series:
            if level is None or math.isnan(level):
                # See _push_level's matching guard: NaN is inert (a gap), never
                # a record -- kept in lockstep so evaluate()/step() agree on any
                # anchor pathological enough to emit NaN.
                best = None
                cumulative = 0.0
            elif best is None or level > best:
                best = level
                cumulative = 0.0
            else:
                cumulative += self._target_rate
        return cumulative
