"""BUD-1: tests for the BudgetDetector."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathlayer.detectors.budget import (
    BudgetDetector,
    BudgetState,
    advance_budget_state,
    budget_delta,
)
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.validation import Trace, inject_cost_spike, load_trace

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _events(*costs: float) -> list[ToolEvent]:
    return [ToolEvent(tool="call", args={"i": i}, cost_usd=c) for i, c in enumerate(costs)]


def test_empty_history_is_silent() -> None:
    assert BudgetDetector(max_cost_usd=5.0).evaluate([]).confidence == 0.0


def test_budget_fold_counts_only_positive_captured_costs_as_priced() -> None:
    state = BudgetState()
    for event in _events(1.25, 0.0, 2.75):
        state = advance_budget_state(state, budget_delta(event))

    assert state == BudgetState(captured_usd=4.0, priced_events=2)


def test_budget_state_preserves_historical_positional_construction() -> None:
    assert BudgetState(1.0, 2.0, 3) == BudgetState(
        captured_usd=1.0,
        estimated_usd=2.0,
        invalid_costs=3,
        priced_events=0,
    )


def test_invalid_and_estimated_costs_are_not_priced_events() -> None:
    state = advance_budget_state(
        BudgetState(),
        budget_delta(
            ToolEvent(tool="call", cost_usd=float("nan")),
            default_cost_usd=0.5,
        ),
    )

    assert state == BudgetState(estimated_usd=0.5, invalid_costs=1)


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


def test_expensive_unpriced_tool_reaches_budget_faster_than_cheap_tool() -> None:
    detector = BudgetDetector(
        max_cost_usd=2.0,
        tool_costs_usd={"read_file": 0.1, "train_model": 1.0},
    )
    cheap = [ToolEvent(tool="read_file") for _ in range(2)]
    expensive = [ToolEvent(tool="train_model") for _ in range(2)]

    assert detector.evaluate(cheap).confidence == pytest.approx(0.1)
    assert detector.evaluate(expensive).confidence == 1.0


def test_captured_positive_cost_wins_over_estimate() -> None:
    detector = BudgetDetector(max_cost_usd=10.0, tool_costs_usd={"train_model": 8.0})
    signal = detector.evaluate([ToolEvent(tool="train_model", cost_usd=1.0)])

    assert signal.confidence == pytest.approx(0.1)
    assert "$1.00 captured + $0.00 estimated" in signal.reason


def test_unknown_tool_uses_default_estimate() -> None:
    detector = BudgetDetector(max_cost_usd=5.0, default_cost_usd=0.5)
    signal = detector.evaluate([ToolEvent(tool="unknown"), ToolEvent(tool="unknown")])
    assert signal.confidence == pytest.approx(0.2)


def test_zero_estimate_can_ignore_an_unpriced_tool() -> None:
    detector = BudgetDetector(
        max_cost_usd=5.0, default_cost_usd=1.0, tool_costs_usd={"free_tool": 0.0}
    )
    assert detector.evaluate([ToolEvent(tool="free_tool")]).confidence == 0.0


def test_llm_call_uses_captured_cost_then_configured_fallback() -> None:
    detector = BudgetDetector(max_cost_usd=5.0, tool_costs_usd={"gpt-4o": 2.0})
    captured = ToolEvent(tool="gpt-4o", kind=EventKind.LLM_CALL, cost_usd=0.5)
    unpriced = ToolEvent(tool="gpt-4o", kind=EventKind.LLM_CALL)

    signal = detector.evaluate([captured, unpriced])
    assert signal.confidence == pytest.approx(0.5)
    assert "$0.50 captured + $2.00 estimated" in signal.reason


def test_progress_summary_never_uses_default_estimate() -> None:
    detector = BudgetDetector(max_cost_usd=5.0, default_cost_usd=1.0)
    event = ToolEvent(tool="pathlayer_progress_summary", kind=EventKind.PROGRESS_SUMMARY)
    assert detector.evaluate([event]).confidence == 0.0


@pytest.mark.parametrize("bad_cost", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_runtime_cost_uses_estimate_and_reports_diagnostic(bad_cost: float) -> None:
    detector = BudgetDetector(max_cost_usd=5.0, tool_costs_usd={"call": 0.5})
    signal = detector.evaluate([ToolEvent(tool="call", cost_usd=bad_cost)])

    assert signal.confidence == pytest.approx(0.1)
    assert signal.diagnostics == (
        "1 invalid event cost(s) ignored; configured estimates applied where eligible",
    )


def test_invalid_progress_summary_cost_is_ignored_without_estimate() -> None:
    detector = BudgetDetector(max_cost_usd=5.0, default_cost_usd=1.0)
    event = ToolEvent(
        tool="pathlayer_progress_summary",
        kind=EventKind.PROGRESS_SUMMARY,
        cost_usd=float("nan"),
    )
    signal = detector.evaluate([event])

    assert signal.confidence == 0.0
    assert signal.diagnostics == (
        "1 invalid event cost(s) ignored; configured estimates applied where eligible",
    )


def test_accumulated_captured_cost_overflowing_to_inf_still_saturates() -> None:
    # Each individual cost is finite and valid, but their sum overflows the float
    # range to +inf. The confidence must still saturate at 1.0 (not become NaN or
    # crash the reason formatting).
    huge = 1e308
    signal = BudgetDetector(max_cost_usd=5.0).evaluate(_events(huge, huge))

    assert signal.confidence == 1.0
    assert "$0.00 left" in signal.reason


def test_continuous_and_resumed_runs_have_equivalent_effective_spend() -> None:
    events = [ToolEvent(tool="read_file") for _ in range(4)]
    continuous = BudgetDetector(max_cost_usd=5.0, tool_costs_usd={"read_file": 0.5}).evaluate(
        events
    )
    resumed = BudgetDetector(
        max_cost_usd=5.0,
        baseline_usd=1.0,
        tool_costs_usd={"read_file": 0.5},
    ).evaluate(events[2:])

    assert resumed.confidence == continuous.confidence == pytest.approx(0.4)
    assert "$1.00 baseline" in resumed.reason


def test_source_mapping_mutation_does_not_change_detector() -> None:
    costs = {"call": 0.5}
    detector = BudgetDetector(max_cost_usd=5.0, tool_costs_usd=costs)
    costs["call"] = 5.0

    assert detector.evaluate([ToolEvent(tool="call")]).confidence == pytest.approx(0.1)


def test_incremental_step_matches_batch_for_every_prefix() -> None:
    detector = BudgetDetector(
        max_cost_usd=5.0,
        baseline_usd=0.25,
        default_cost_usd=0.2,
        tool_costs_usd={"train_model": 2.0},
    )
    events = [
        ToolEvent(tool="read_file"),
        ToolEvent(tool="train_model"),
        ToolEvent(tool="gpt-4o", kind=EventKind.LLM_CALL, cost_usd=0.5),
        ToolEvent(tool="broken", cost_usd=float("nan")),
        ToolEvent(tool="pathlayer_progress_summary", kind=EventKind.PROGRESS_SUMMARY),
    ]
    state = detector.create_state()
    for index, event in enumerate(events):
        state, incremental = detector.step(state, event)
        assert incremental == detector.evaluate(events[: index + 1])


def test_shared_detector_keeps_pipeline_states_isolated() -> None:
    detector = BudgetDetector(
        max_cost_usd=2.0,
        tool_costs_usd={"cheap": 0.1, "expensive": 1.0},
    )
    expensive_pipeline = Pipeline([detector], Policy())
    cheap_pipeline = Pipeline([detector], Policy())

    expensive_pipeline.run([ToolEvent("expensive"), ToolEvent("expensive")])
    cheap_pipeline.run([ToolEvent("cheap"), ToolEvent("cheap")])

    assert expensive_pipeline.stopped
    assert not cheap_pipeline.stopped


def test_estimated_cost_boundary_fixture_covers_both_configurations() -> None:
    trace = load_trace(_FIXTURES_DIR / "bud3_unpriced_tool_calls.json")
    cheap_config = trace.metadata["cheap_config"]
    expensive_config = trace.metadata["expensive_config"]
    expectations = trace.metadata["expectations_by_config"]

    cheap = BudgetDetector(**cheap_config).evaluate(trace.events)
    expensive = BudgetDetector(**expensive_config).evaluate(trace.events)

    assert (cheap.confidence == 1.0) is expectations["cheap_config"]["budget_should_reach_ceiling"]
    assert (expensive.confidence == 1.0) is expectations["expensive_config"][
        "budget_should_reach_ceiling"
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cost_usd": 0.0},
        {"max_cost_usd": -1.0},
        {"max_cost_usd": float("nan")},
        {"baseline_usd": -1.0},
        {"baseline_usd": float("inf")},
        {"default_cost_usd": -1.0},
        {"default_cost_usd": float("nan")},
    ],
)
def test_invalid_config_raises(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BudgetDetector(**kwargs)


@pytest.mark.parametrize("bad_cost", [-1.0, float("nan"), float("inf"), True, "1.0"])
def test_invalid_tool_estimate_raises(bad_cost: object) -> None:
    with pytest.raises(ValueError):
        BudgetDetector(tool_costs_usd={"call": bad_cost})  # type: ignore[dict-item]


def test_non_string_tool_name_in_estimate_mapping_raises() -> None:
    with pytest.raises(ValueError, match="keys must be strings"):
        BudgetDetector(tool_costs_usd={1: 0.5})  # type: ignore[dict-item]


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


@pytest.mark.parametrize("amount", [0.0, 1.0])
def test_adapter_estimate_retains_provenance_without_fallback(amount: float) -> None:
    detector = BudgetDetector(max_cost_usd=2.0, default_cost_usd=100.0)
    event = ToolEvent(
        tool="model",
        kind=EventKind.LLM_CALL,
        cost_usd=amount,
        metadata={"cost_measurement": "estimated_total_tokens"},
    )
    state, signal = detector.step(detector.create_state(), event)
    assert state == BudgetState(estimated_usd=amount)
    assert detector.accumulate([event]) == state
    assert detector.evaluate([event]) == signal
    assert signal.confidence == amount / 2.0
    assert budget_delta(event) == state  # also used when the detector is disabled
    if amount:
        assert "$0.00 captured + $1.00 estimated" in signal.reason
