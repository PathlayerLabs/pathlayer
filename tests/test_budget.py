"""BUD-1: tests for the BudgetDetector."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathlayer.detectors.budget import BudgetDetector
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.validation import Trace, inject_cost_spike, load_trace

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def _events(*costs: float) -> list[ToolEvent]:
    return [ToolEvent(tool="call", args={"i": i}, cost_usd=c) for i, c in enumerate(costs)]


def test_empty_history_is_silent() -> None:
    assert BudgetDetector(max_cost_usd=5.0).evaluate([]).confidence == 0.0


def test_under_budget_is_proportional() -> None:
    signal = BudgetDetector(max_cost_usd=5.0).evaluate(_events(1.0, 1.0))
    assert signal.confidence == pytest.approx(0.4)


def test_reaches_one_at_ceiling() -> None:
    assert BudgetDetector(max_cost_usd=5.0).evaluate(_events(2.0, 3.0)).confidence == 1.0


def test_saturates_over_budget() -> None:
    signal = BudgetDetector(max_cost_usd=5.0).evaluate(_events(4.0, 4.0))
    assert signal.confidence == 1.0
    assert "$0.00 left" in signal.reason


def test_reason_reports_spend_and_remaining() -> None:
    signal = BudgetDetector(max_cost_usd=10.0).evaluate(_events(3.0))
    assert "$3.00" in signal.reason
    assert "$7.00 left" in signal.reason


def test_baseline_seeds_cross_run_cost() -> None:
    detector = BudgetDetector(max_cost_usd=5.0, baseline_usd=4.0)
    assert detector.evaluate(_events(1.0)).confidence == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [{"max_cost_usd": 0.0}, {"max_cost_usd": -1.0}, {"baseline_usd": -1.0}],
)
def test_invalid_config_raises(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BudgetDetector(**kwargs)


def test_pipeline_stops_near_the_ceiling() -> None:
    pipeline = Pipeline([BudgetDetector(max_cost_usd=5.0)], Policy())
    pipeline.run(_events(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    assert pipeline.stopped
    # stop_threshold 0.9 -> stops once spend reaches 4.5, i.e. the 5th $1 event.
    assert pipeline.history[-1].args["i"] == 4


# --------------------------------------------------------------------------- #
# harness-usable fixtures (both directions)
# --------------------------------------------------------------------------- #
def test_zero_cost_golden_trace_is_silent() -> None:
    # Negative direction on a real labeled trace: a healthy run with no cost
    # must never trip the budget guard.
    trace = load_trace(_TRACES_DIR / "healthy" / "varied_long_run.json")
    assert BudgetDetector(max_cost_usd=5.0).evaluate(trace.events).confidence == 0.0


def test_fires_on_cost_spike_mutation() -> None:
    # Positive direction on a VAL-4-generated fault: a single $1 event spiked
    # x10 blows the $5 ceiling. The mutated trace carries a budget expectation,
    # so this is not a mislabeled "healthy" trace.
    base = Trace(label="healthy", events=[ToolEvent(tool="call", cost_usd=1.0)])
    mutated = inject_cost_spike(base, event_idx=0, multiplier=10.0)
    signal = BudgetDetector(max_cost_usd=5.0).evaluate(mutated.events)
    assert signal.confidence == 1.0
    assert Policy().decide([signal]) is Action.STOP
