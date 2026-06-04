"""RetryDetector.

Counts repeated calls to the same tool that keep returning the same error /
outcome. Backed by policies `max_same_error` and `max_tool_retries`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import ToolEvent


class RetryDetector(Detector):
    name = "retry"

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        # TODO: count same-tool same-error streaks against policy thresholds.
        return Signal(self.name, confidence=0.0)
