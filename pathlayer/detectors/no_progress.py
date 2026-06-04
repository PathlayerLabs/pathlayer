"""NoProgressDetector (class ii: no-progress-without-repetition).

The differentiated case (the 11-day / ~$47k loop): every step looks different
yet nothing advances. Repetition detectors stay silent here.

Plan: feed a domain progress signal into a CUSUM change-detector (Page, 1954)
with persistence over k windows. For coding agents the progress signal is
deterministic ground truth -- tests passing, compile/exit codes, validated
git-diff size -- not text similarity.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import ToolEvent


class NoProgressDetector(Detector):
    name = "no_progress"

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        # TODO: compute domain progress signal, run CUSUM, require persistence.
        return Signal(self.name, confidence=0.0)
