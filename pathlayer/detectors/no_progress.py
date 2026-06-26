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
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.progress import ProgressAnchor, progress_series
from pathlayer.events import EventKind, ToolEvent


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
            if level is None:
                best = None
                cumulative = 0.0
            elif best is None or level > best:
                best = level
                cumulative = 0.0
            else:
                cumulative += self._target_rate
        return cumulative
