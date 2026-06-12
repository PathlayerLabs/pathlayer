"""LoopDetector (class i: repetition).

Table stakes. Canonicalizes each event into an action token (see
:func:`pathlayer.detectors.canonical.canonicalize`) and looks for an exact
repeated block at the tail of the stream: a looping agent keeps replaying the
same action(s). The repeating unit can be a single action (``read, read,
read``) or a multi-action cycle (``a, b, a, b``).

The signal is continuous. ``k`` is the minimum number of consecutive
repetitions of the block before any confidence is reported; confidence then
rises toward 1 as the loop persists. N-gram/entropy scoring and a windowed
``B(t)`` are deferred to LOOP-2/LOOP-3.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.canonical import canonicalize
from pathlayer.events import ToolEvent


class LoopDetector(Detector):
    name = "loop"

    def __init__(self, k: int = 2, window: int = 256) -> None:
        if k < 2:
            raise ValueError("k must be >= 2: a loop needs at least two repetitions")
        if window < 2 * k:
            raise ValueError("window must be >= 2 * k to fit at least k repetitions")
        self._k = k
        self._window = window

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        # Bound per-step cost: only the recent tail window is canonicalized and
        # scanned. Pipeline.process() runs every detector on every event, so an
        # unbounded scan would be ~O(N^3) over a long run; the window keeps it
        # O(window^2) per step regardless of history length.
        tokens = [canonicalize(event) for event in history[-self._window :]]
        reps, period = self._max_trailing_repetition(tokens)
        if reps < self._k:
            return Signal(self.name, confidence=0.0)
        # Saturating map: reps == k -> 0.5, then halve the gap to 1 per extra rep.
        confidence = 1.0 - 0.5 ** (reps - self._k + 1)
        reason = f"{reps}x repeat of {period}-action block"
        return Signal(self.name, confidence=confidence, reason=reason)

    @staticmethod
    def _max_trailing_repetition(tokens: list[str]) -> tuple[int, int]:
        """Return (repetitions, period) of the strongest repeated tail block."""
        n = len(tokens)
        best_reps, best_period = 1, 0
        for period in range(1, n // 2 + 1):
            block = tokens[n - period :]
            reps = 1
            idx = n - period
            while idx - period >= 0 and tokens[idx - period : idx] == block:
                reps += 1
                idx -= period
            if reps > best_reps:
                best_reps, best_period = reps, period
        return best_reps, best_period
