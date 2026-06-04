"""BudgetDetector.

Tracks cumulative cost against `max_cost_usd` and raises confidence as the
run approaches / exceeds the ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import ToolEvent


class BudgetDetector(Detector):
    name = "budget"

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        # TODO: sum cost_usd and compare against the budget policy.
        return Signal(self.name, confidence=0.0)
