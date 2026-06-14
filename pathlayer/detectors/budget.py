"""BudgetDetector (BUD-1, class iii: cost runaway).

The simplest detector: pure accumulation. It sums ``cost_usd`` across the
observed events and reports a continuous confidence equal to the fraction of
the budget consumed, saturating at 1.0 once the ceiling is reached.

Cross-run spend (the crash-resume / multi-run case) is supported via
``baseline_usd``: the cumulative cost already recorded for this run id before
the current process started. Wiring that baseline from the local SQLite store
is INF-1's job -- this detector never touches storage itself, it just adds the
seed it is given. Until then ``baseline_usd`` defaults to 0.0 (single run).

With the default arbitration (stop_threshold 0.9) a run is stopped once it has
burned ~90% of the budget -- an early brake rather than a post-mortem once the
money is already gone.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import ToolEvent


class BudgetDetector(Detector):
    name = "budget"

    def __init__(self, max_cost_usd: float = 5.0, baseline_usd: float = 0.0) -> None:
        if max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be > 0")
        if baseline_usd < 0:
            raise ValueError("baseline_usd must be >= 0")
        self._max_cost_usd = max_cost_usd
        self._baseline_usd = baseline_usd

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        spend = self._baseline_usd + sum(event.cost_usd for event in history)
        confidence = min(1.0, spend / self._max_cost_usd)
        if confidence <= 0.0:
            return Signal(self.name, confidence=0.0)
        remaining = max(0.0, self._max_cost_usd - spend)
        reason = f"${spend:.2f} / ${self._max_cost_usd:.2f} spent, ${remaining:.2f} left"
        return Signal(self.name, confidence=confidence, reason=reason)
