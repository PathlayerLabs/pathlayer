"""LoopDetector (class i: repetition).

Table stakes. Plan: canonicalize each event into an action token, then detect
repetition via exact-repeat / n-gram recurrence and action-stream entropy
(a looping agent's token stream compresses trivially). Keep it cheap.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import ToolEvent


class LoopDetector(Detector):
    name = "loop"

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        # TODO: action tokenization + n-gram/entropy repetition score.
        return Signal(self.name, confidence=0.0)
