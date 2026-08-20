"""BudgetDetector (BUD-1/BUD-3, class iii: cost runaway).

The detector accumulates captured ``cost_usd`` values and reports a continuous
confidence equal to the fraction of the budget consumed, saturating at 1.0 once
the ceiling is reached. BUD-3 fills the gap left by adapters that cannot price a
tool/model call: a configured per-tool USD estimate is used only when a
``TOOL_CALL`` or ``LLM_CALL`` has no positive captured cost. A captured positive
cost always wins, so model pricing is never charged twice.

``baseline_usd`` is an explicit external seed for spend that is not present in
the replayed event journal. SQLite restoration replays every durable event, so
its counters must never be fed back as a baseline or the same cost would be
charged twice.

With the default arbitration (stop_threshold 0.9) a run is stopped once it has
burned ~90% of the budget -- an early brake rather than a post-mortem once the
money is already gone.

The accumulator itself is public (:class:`BudgetState`, :meth:`accumulate`) so an
end-of-run report (OBS-2) can attribute a cost figure to its source instead of
printing one total and leaving the reader to assume it was measured.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import EventKind, ToolEvent

_ESTIMABLE_KINDS = frozenset({EventKind.TOOL_CALL, EventKind.LLM_CALL})


def _validated_cost(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


@dataclass(frozen=True, slots=True)
class BudgetState:
    """Public, immutable cost fold shared by BUD and persisted reports.

    ``priced_events`` counts valid, strictly positive captured costs. Under the
    normalized event contract, zero means no captured price rather than a
    measured zero; configured estimates remain a separate component.
    ``invalid_costs`` counts events whose ``cost_usd`` was unusable (non-finite
    or negative) and was therefore ignored.

    The configured cross-run seed is *not* part of the accumulator: it is a
    property of the detector (:attr:`BudgetDetector.baseline_usd`), not something
    the events produced.
    """

    captured_usd: float = 0.0
    estimated_usd: float = 0.0
    invalid_costs: int = 0
    priced_events: int = 0

    @property
    def effective_usd(self) -> float:
        return self.captured_usd + self.estimated_usd


def budget_delta(
    event: ToolEvent,
    *,
    tool_costs_usd: Mapping[str, float] | None = None,
    default_cost_usd: float = 0.0,
) -> BudgetState:
    """Return the deterministic cost contribution of one event.

    Passing no estimates still records captured and invalid costs. This is the
    mode persistence uses when the budget detector is disabled.
    """
    raw = event.cost_usd
    valid = isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw)
    if valid and raw > 0.0:
        return BudgetState(captured_usd=float(raw), priced_events=1)

    invalid = not valid or raw < 0.0
    estimate = 0.0
    if event.kind in _ESTIMABLE_KINDS:
        estimate = (tool_costs_usd or {}).get(event.tool, default_cost_usd)
    return BudgetState(estimated_usd=estimate, invalid_costs=int(invalid))


def advance_budget_state(state: BudgetState, delta: BudgetState) -> BudgetState:
    """Add one event contribution to an accumulated cost state."""
    return BudgetState(
        captured_usd=state.captured_usd + delta.captured_usd,
        priced_events=state.priced_events + delta.priced_events,
        estimated_usd=state.estimated_usd + delta.estimated_usd,
        invalid_costs=state.invalid_costs + delta.invalid_costs,
    )


class BudgetDetector(Detector):
    name = "budget"

    def __init__(
        self,
        max_cost_usd: float = 5.0,
        baseline_usd: float = 0.0,
        *,
        tool_costs_usd: Mapping[str, float] | None = None,
        default_cost_usd: float = 0.0,
    ) -> None:
        max_cost = _validated_cost(max_cost_usd, "max_cost_usd")
        if max_cost == 0.0:
            raise ValueError("max_cost_usd must be > 0")
        self._max_cost_usd = max_cost
        self._baseline_usd = _validated_cost(baseline_usd, "baseline_usd")
        self._default_cost_usd = _validated_cost(default_cost_usd, "default_cost_usd")

        # The private ``costs`` dict is a fresh copy, so a later mutation of the
        # caller's source mapping cannot reach the detector. The MappingProxyType
        # only makes that private copy read-only (guarding the detector's own
        # purity); the values are floats, so there is nothing mutable to alias.
        costs: dict[str, float] = {}
        for tool, value in (tool_costs_usd or {}).items():
            if not isinstance(tool, str):
                raise ValueError("tool_costs_usd keys must be strings")
            costs[tool] = _validated_cost(value, f"tool_costs_usd[{tool!r}]")
        self._tool_costs_usd: Mapping[str, float] = MappingProxyType(costs)

    @property
    def max_cost_usd(self) -> float:
        """The configured ceiling this detector measures spend against."""
        return self._max_cost_usd

    @property
    def baseline_usd(self) -> float:
        """Cross-run spend seeded into every accumulation (0.0 for a single run).

        An explicit external seed for spend that is *not* present in the event
        history this detector folds. Public alongside :class:`BudgetState`
        (OBS-2) so a report can show the third component of the effective spend
        instead of implying the run started from zero. A store that restores a
        run by replaying its journal must never feed its own counters back here:
        the replay already recounts them.
        """
        return self._baseline_usd

    def accumulate(self, history: Sequence[ToolEvent]) -> BudgetState:
        """Fold ``history`` into a :class:`BudgetState` without arbitrating.

        The same pure fold :meth:`evaluate` runs, but returning the accumulator
        instead of the :class:`Signal`, so a session report can state how much
        of the spend was captured and how much was a BUD-3 estimate. Pure and
        deterministic: no clock, no I/O, no shared state -- calling it again on
        the same events yields the same state.
        """
        state = self.create_state()
        for event in history:
            state = self._advance(state, event)
        return state

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        return self._signal(self.accumulate(history))

    def create_state(self) -> BudgetState:
        return BudgetState()

    def step(self, state: BudgetState, event: ToolEvent) -> tuple[BudgetState, Signal]:
        next_state = self._advance(state, event)
        return next_state, self._signal(next_state)

    def _advance(self, state: BudgetState, event: ToolEvent) -> BudgetState:
        return advance_budget_state(
            state,
            budget_delta(
                event,
                tool_costs_usd=self._tool_costs_usd,
                default_cost_usd=self._default_cost_usd,
            ),
        )

    def _signal(self, state: BudgetState) -> Signal:
        spend = self._baseline_usd + state.captured_usd + state.estimated_usd
        confidence = min(1.0, spend / self._max_cost_usd)
        diagnostics = (
            (
                f"{state.invalid_costs} invalid event cost(s) ignored; "
                "configured estimates applied where eligible",
            )
            if state.invalid_costs
            else ()
        )
        if confidence <= 0.0:
            return Signal(self.name, confidence=0.0, diagnostics=diagnostics)
        remaining = max(0.0, self._max_cost_usd - spend)
        reason = (
            f"${spend:.2f} effective / ${self._max_cost_usd:.2f} budget, "
            f"${remaining:.2f} left "
            f"(${state.captured_usd:.2f} captured + ${state.estimated_usd:.2f} estimated"
        )
        if self._baseline_usd:
            reason += f" + ${self._baseline_usd:.2f} baseline"
        reason += ")"
        return Signal(self.name, confidence=confidence, reason=reason, diagnostics=diagnostics)
