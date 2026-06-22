"""NoProgressDetector (class ii: no-progress-without-repetition).

The differentiated case (the 11-day / ~$47k loop): every step looks different
yet nothing advances, so repetition detectors stay silent. This detector
aggregates pluggable :class:`~pathlayer.detectors.progress.ProgressAnchor`s: it
recovers each anchor's progress level over the recent window and fires when the
agent stays active without the level improving.

It references no concrete anchor -- the anchors are injected -- so the same
detector serves any agent and only the anchors are domain-specific. An anchor
that sees none of its evidence abstains (``measure`` -> ``None``); if *no*
anchor is applicable the detector is silent (confidence 0). We never stop an
agent we cannot measure -- a false stop is worse than no Pathlayer at all.

NOP-1 ships a deliberately minimal flat-progress score. A CUSUM change detector
over the same series (NOP-2) and an explicit activity/progress ratio (NOP-3)
build on this.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.progress import ProgressAnchor
from pathlayer.events import ToolEvent


class NoProgressDetector(Detector):
    name = "no_progress"

    def __init__(
        self,
        anchors: Sequence[ProgressAnchor],
        no_progress_steps: int = 5,
        window: int = 64,
    ) -> None:
        self._anchors = list(anchors)
        if not self._anchors:
            raise ValueError("NoProgressDetector needs at least one anchor")
        if no_progress_steps < 1:
            raise ValueError("no_progress_steps must be >= 1")
        if window < no_progress_steps:
            # A window shorter than the threshold caps stagnation below 1.0, so
            # the stop threshold could never be reached.
            raise ValueError("window must be >= no_progress_steps")
        self._no_progress_steps = no_progress_steps
        self._window = window

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        # Measure each anchor once on the full history: this both filters the
        # applicable anchors and yields the current level, reused below to avoid
        # a duplicate full-history measure inside _stagnant_steps.
        applicable: list[tuple[ProgressAnchor, float]] = []
        for anchor in self._anchors:
            level = anchor.measure(history)
            if level is not None:
                applicable.append((anchor, level))
        if not applicable:
            # No anchor can measure this agent -> stay silent.
            return Signal(self.name, confidence=0.0)
        # If any single anchor still shows recent progress, the agent is
        # advancing, so take the most optimistic (smallest) stagnation.
        stagnant = min(
            self._stagnant_steps(anchor, history, current) for anchor, current in applicable
        )
        if stagnant <= 0:
            return Signal(self.name, confidence=0.0)
        confidence = min(1.0, stagnant / self._no_progress_steps)
        plural = "s" if stagnant > 1 else ""
        reason = f"{stagnant} action{plural} without progress"
        return Signal(self.name, confidence=confidence, reason=reason)

    def _stagnant_steps(
        self, anchor: ProgressAnchor, history: Sequence[ToolEvent], current: float
    ) -> int:
        """Trailing events over which the anchor's level has not strictly
        improved, bounded by the recent window.

        Counts back from the latest event while the progress level stays flat or
        drops, stopping at the first improvement or once the anchor stops being
        applicable further back (its level becomes ``None``, i.e. there is no
        earlier baseline to compare against). ``current`` is the level at the
        full history, already computed by :meth:`evaluate`.
        """
        n = len(history)
        span = min(self._window, n)
        # Progress level at prefixes history[:n-span] .. history[:n-1]; the
        # current level (history[:n]) is appended from the caller. The window
        # bounds the scan: re-measuring each prefix is ~O(window * n) per call.
        # The full per-run cost is quadratic; smoothing/streaming that series is
        # deferred to NOP-2 (CUSUM), which reworks the progress series anyway.
        levels: list[float | None] = [
            anchor.measure(history[: n - span + offset]) for offset in range(span)
        ]
        levels.append(current)
        stagnant = 0
        for index in range(len(levels) - 1, 0, -1):
            cur = levels[index]
            previous = levels[index - 1]
            if cur is None or previous is None:
                break
            if cur > previous:
                break
            stagnant += 1
        return stagnant
