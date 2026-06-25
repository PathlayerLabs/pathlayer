"""Event-kind filtering: action detectors ignore non-action events.

A dedicated ``EventKind.LLM_CALL`` event (issue #77) carries token cost so
``BudgetDetector`` can account for model spend. Those events live in the same
history as tool calls, so the action detectors (loop/retry/no_progress) must
filter them out via ``Detector.event_kinds`` -- otherwise a reasoning agent that
issues several LLM calls would look like a loop, or inflate the no-progress
CUSUM toward a false (and now actionable) stop.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.events import EventKind, ToolEvent


def _llm(cost: float = 0.0) -> ToolEvent:
    return ToolEvent(tool="gpt-4o", cost_usd=cost, kind=EventKind.LLM_CALL)


def _tests(passed: int, failed: int) -> ToolEvent:
    return ToolEvent(tool="run_tests", result={"passed": passed, "failed": failed})


def _edit(i: int) -> ToolEvent:
    return ToolEvent(tool="edit_file", args={"patch": f"change {i}"})


def test_relevant_is_noop_when_event_kinds_is_none() -> None:
    class AllKinds(Detector):
        def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
            return Signal(self.name, confidence=0.0)

    detector = AllKinds()
    history = [_tests(1, 0), _llm()]
    # None means "all events": the same sequence is returned unchanged.
    assert detector._relevant(history) is history


def test_relevant_filters_to_declared_kinds() -> None:
    class ToolOnly(Detector):
        event_kinds = frozenset({EventKind.TOOL_CALL})

        def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
            return Signal(self.name, confidence=0.0)

    history = [_tests(1, 0), _llm(), _edit(0)]
    filtered = ToolOnly()._relevant(history)
    assert all(event.kind is EventKind.TOOL_CALL for event in filtered)
    assert len(filtered) == 2


def test_loop_ignores_repeated_llm_calls() -> None:
    # A run of identical LLM-cost events is not a tool loop.
    history = [_llm() for _ in range(8)]
    assert LoopDetector().evaluate(history).confidence == 0.0


def test_loop_still_fires_on_tool_loop_with_llm_calls_interleaved() -> None:
    # Interleaving LLM events between identical tool calls must not hide the loop.
    bare = [ToolEvent(tool="search", args={"q": "same"}) for _ in range(6)]
    interleaved: list[ToolEvent] = []
    for event in bare:
        interleaved.append(_llm())
        interleaved.append(event)
    assert (
        LoopDetector().evaluate(interleaved).confidence == LoopDetector().evaluate(bare).confidence
    )
    assert LoopDetector().evaluate(interleaved).confidence > 0.0


def test_no_progress_unaffected_by_interleaved_llm_calls() -> None:
    # Flat ground-truth tests with LLM calls between edits: the CUSUM must match
    # the LLM-free run exactly (LLM steps are not stagnant progress steps).
    bare: list[ToolEvent] = [_tests(11, 3)]
    for i in range(6):
        bare.append(_edit(i))
        bare.append(_tests(11, 3))
    interleaved: list[ToolEvent] = []
    for event in bare:
        interleaved.append(event)
        interleaved.append(_llm())

    detector = NoProgressDetector(coding_anchors())
    assert detector.evaluate(bare).confidence == 1.0
    assert detector.evaluate(interleaved).confidence == detector.evaluate(bare).confidence


def test_budget_sums_llm_call_costs() -> None:
    # BudgetDetector (event_kinds=None) DOES see LLM events -- that is the point.
    history = [_edit(i) for i in range(3)] + [_llm(2.0), _llm(2.0), _llm(2.0)]
    assert BudgetDetector(max_cost_usd=5.0).evaluate(history).confidence == 1.0
