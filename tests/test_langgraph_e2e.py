"""End-to-end stress test of the LangGraph/LangChain adapter (ADP-0).

Unlike ``test_adapter_langgraph.py`` (which checks the adapter's capture
contract in isolation), this module proves that events *actually captured* from
real LangChain ``@tool`` runs -- and from a real LangGraph ``StateGraph`` --
drive the **existing** detectors through the **existing** ``Pipeline`` + default
``Policy`` to the expected outcome:

* (a) LOOP        -- identical tool calls -> ``LoopDetector`` fires -> STOP.
* (b) RETRY       -- same call, same error repeated -> ``RetryDetector`` fires
                     -> STOP.
* (c) NO-PROGRESS -- many *different* calls, flat ground-truth progress signal
                     -> ``NoProgressDetector`` fires (the differentiated "$47k"
                     case) while ``LoopDetector`` stays silent. Since NOP-3 this
                     signal is **actionable**: the default ``Policy`` STOPs on it.
                     Success suppression keeps a *green* suite from being stopped,
                     but here the suite stays red, so the run is halted.
* (d) BUDGET      -- captured ``LLM_CALL`` cost (issue #77) drives
                     ``BudgetDetector`` to STOP once spend crosses the ceiling,
                     while loop/no_progress stay silent (varied tools, no test
                     evidence), proving the STOP is on budget alone.

Each scenario carries a numeric assertion on the detector confidence *and* a
negative (healthy) counterpart, so a false positive would fail the suite too.
Skipped entirely unless the optional ``langchain_core`` dependency is present.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool

try:
    # Optional: only the ACT-1 StateGraph test below needs it, and that test
    # importorskips "langgraph" before using _WarnState. Guarded so the
    # langchain_core-only tests in this module still collect without langgraph.
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover - langgraph is an optional extra
    add_messages = None  # type: ignore[assignment]

from pathlayer import Pipeline
from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.detectors.base import Signal
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    GitProgressAnchor,
    TestProgressAnchor,
    coding_anchors,
)
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.policies import Action, Policy
from pathlayer.policies.base import PolicyConfig


class _GraphState(TypedDict):
    """Module-level so LangGraph can resolve the node's string annotations
    (``from __future__ import annotations`` turns them into strings that
    ``get_type_hints`` looks up in module globals, not in a function's locals).
    """

    steps: int


# --------------------------------------------------------------------------- #
# Real LangChain tools. Pure and deterministic so capture is reproducible.
# --------------------------------------------------------------------------- #


@tool
def search(query: str) -> str:
    """Return a constant result regardless of the query (a stuck search)."""
    return "static-result"


@tool
def flaky(x: int) -> int:
    """Always raise the same error (a call that will never succeed)."""
    raise RuntimeError("db timeout")


@tool
def run_tests(suite: str) -> dict[str, int]:
    """Report a *flat* test result: ground-truth progress never moves."""
    return {"passed": 11, "failed": 3}


@tool
def run_tests_improving(failed: int) -> dict[str, int]:
    """Report a strictly improving test result (real progress)."""
    return {"passed": 11, "failed": failed}


@tool
def run_tests_rich(failed_ids: list[str], run: int) -> dict[str, object]:
    """Report stable per-test outcomes using the optional NOP-5 contract."""
    test_ids = ("a", "b", "c")
    outcomes = {test_id: ("failed" if test_id in failed_ids else "passed") for test_id in test_ids}
    return {
        "passed": len(test_ids) - len(failed_ids),
        "failed": len(failed_ids),
        "test_outcomes": outcomes,
        "scope_id": "all",
        "collection_complete": True,
        "scope_exhaustive": True,
    }


@tool
def edit_file(patch: str) -> str:
    """A genuinely different action each turn (breaks repetition detectors)."""
    return "ok"


# --------------------------------------------------------------------------- #
# Capture helpers: run real tools through the adapter, collect the events.
# --------------------------------------------------------------------------- #


def _capture(run: Callable[[LangGraphAdapter], None]) -> list[ToolEvent]:
    """Drive ``run`` against a fresh adapter and return the captured events."""
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append)
    adapter.attach()
    run(adapter)
    adapter.detach()
    return events


def _all_detectors() -> list:
    """The full detector set, so each scenario also proves the others stay quiet."""
    return [
        LoopDetector(),
        RetryDetector(),
        NoProgressDetector(coding_anchors()),
        BudgetDetector(max_cost_usd=5.0),
    ]


def _stops(events: list[ToolEvent], policy: Policy | None = None) -> bool:
    """Replay captured events through Pipeline + a Policy; did it STOP?

    Defaults to the production ``Policy`` (no_progress actionable since NOP-3).
    Pass a custom policy to probe a different arbitration.
    """
    pipeline = Pipeline(_all_detectors(), policy or Policy())
    pipeline.run(events)
    return pipeline.stopped


def _actionable_policy() -> Policy:
    """A Policy with no advisory exclusions, so no_progress can STOP on its own."""
    return Policy(PolicyConfig(advisory_signals=frozenset()))


# --------------------------------------------------------------------------- #
# (a) LOOP
# --------------------------------------------------------------------------- #


def test_loop_detected_from_captured_tool_calls() -> None:
    def run(adapter: LangGraphAdapter) -> None:
        for _ in range(6):
            search.invoke({"query": "same"}, config=adapter.as_config())

    events = _capture(run)
    assert len(events) == 6
    # All six canonicalize to one token, so the loop is unmistakable.
    assert LoopDetector().evaluate(events).confidence == pytest.approx(0.96875)
    assert _stops(events)


def test_loop_negative_distinct_calls_stay_silent() -> None:
    def run(adapter: LangGraphAdapter) -> None:
        for i in range(6):
            search.invoke({"query": f"q-{i}"}, config=adapter.as_config())

    events = _capture(run)
    assert LoopDetector().evaluate(events).confidence == 0.0
    assert not _stops(events)


# --------------------------------------------------------------------------- #
# (b) RETRY
# --------------------------------------------------------------------------- #


def test_retry_detected_from_captured_errors() -> None:
    def run(adapter: LangGraphAdapter) -> None:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="db timeout"):
                flaky.invoke({"x": 1}, config=adapter.as_config())

    events = _capture(run)
    assert len(events) == 3
    # The adapter records the error in metadata["error"]; RetryDetector reads it.
    assert all(e.metadata.get("error") == "db timeout" for e in events)
    assert RetryDetector().evaluate(events).confidence == 1.0
    assert _stops(events)


def test_retry_negative_changed_args_break_the_streak() -> None:
    # Same tool, same error, but a genuinely different argument each attempt is
    # not a stuck retry: canonicalization gives a different token, so the streak
    # never builds.
    def run(adapter: LangGraphAdapter) -> None:
        for i in range(3):
            with pytest.raises(RuntimeError, match="db timeout"):
                flaky.invoke({"x": i}, config=adapter.as_config())

    events = _capture(run)
    assert RetryDetector().evaluate(events).confidence == 0.0


# --------------------------------------------------------------------------- #
# (c) NO-PROGRESS (the differentiated case)
# --------------------------------------------------------------------------- #


def test_no_progress_detected_with_flat_ground_truth() -> None:
    # Every turn is a *different* edit (no repetition), but the test result never
    # moves: high activity, zero progress. 13 events, 12 of which set no new
    # record -> CUSUM = 12 * target_rate (0.2) = 2.4 >= threshold (2.0) -> 1.0.
    def run(adapter: LangGraphAdapter) -> None:
        run_tests.invoke({"suite": "all"}, config=adapter.as_config())
        for i in range(6):
            edit_file.invoke({"patch": f"change {i}"}, config=adapter.as_config())
            run_tests.invoke({"suite": "all"}, config=adapter.as_config())

    events = _capture(run)
    detector = NoProgressDetector(coding_anchors())
    # (a) The differentiator works at the *detector* level: the CUSUM saturates.
    assert detector.evaluate(events).confidence == 1.0
    # The repetition detectors are not fooled into firing on varied activity.
    assert LoopDetector().evaluate(events).confidence == 0.0
    assert RetryDetector().evaluate(events).confidence == 0.0
    # (b) Since NOP-3 no_progress is actionable: the default Policy STOPs on the
    # saturated signal. The flat suite stays red, so success suppression does not
    # apply (it would silence a *green* suite, not this one).
    assert _stops(events)
    # (c) Still stops under an explicitly non-advisory policy (regression anchor).
    assert _stops(events, _actionable_policy())


def test_no_progress_negative_improving_tests_stay_silent() -> None:
    # Same activity shape, but the test result strictly improves each turn ->
    # real progress -> the detector must not fire, even under a policy that would
    # let no_progress stop. The CUSUM resets on every new record, so it stays 0.
    def run(adapter: LangGraphAdapter) -> None:
        for failed in (3, 2, 1, 0):
            edit_file.invoke({"patch": f"fix {failed}"}, config=adapter.as_config())
            run_tests_improving.invoke({"failed": failed}, config=adapter.as_config())

    events = _capture(run)
    detector = NoProgressDetector(coding_anchors())
    assert detector.evaluate(events).confidence == 0.0
    # Not even an actionable (non-advisory) policy stops on genuine progress.
    assert not _stops(events, _actionable_policy())


def test_rich_test_outcomes_survive_real_adapter_and_drive_progress() -> None:
    def run(adapter: LangGraphAdapter) -> None:
        for index, failed_ids in enumerate((["a", "b", "c"], ["b", "c"], ["c"])):
            for edit in range(3):
                edit_file.invoke(
                    {"patch": f"rich repair {index}-{edit}"}, config=adapter.as_config()
                )
            run_tests_rich.invoke(
                {"failed_ids": failed_ids, "run": index}, config=adapter.as_config()
            )

    events = _capture(run)
    captured_results = [event.result for event in events if event.tool == "run_tests_rich"]
    assert all(
        isinstance(result, dict) and "test_outcomes" in result for result in captured_results
    )
    anchor = TestProgressAnchor()
    rich_prefixes = [
        events[: index + 1] for index, event in enumerate(events) if event.tool == "run_tests_rich"
    ]
    assert [anchor.measure(prefix) for prefix in rich_prefixes] == [0.0, 1.0, 2.0]
    assert NoProgressDetector(coding_anchors()).evaluate(events).confidence == 0.0
    assert not _stops(events, _actionable_policy())


def test_rich_test_stall_from_real_adapter_stops_on_no_progress() -> None:
    def run(adapter: LangGraphAdapter) -> None:
        run_tests_rich.invoke({"failed_ids": ["a"], "run": 0}, config=adapter.as_config())
        for index in range(11):
            edit_file.invoke({"patch": f"rich stagnant {index}"}, config=adapter.as_config())

    events = _capture(run)
    detector = NoProgressDetector(coding_anchors())
    assert detector.evaluate(events).confidence >= 0.9
    assert LoopDetector().evaluate(events).confidence == 0.0
    assert RetryDetector().evaluate(events).confidence == 0.0
    assert _stops(events, _actionable_policy())


# --------------------------------------------------------------------------- #
# (d) BUDGET -- model spend is now captured (issue #77): the adapter emits
#     priced LLM_CALL events, so BudgetDetector stops on real model cost.
# --------------------------------------------------------------------------- #


def _drive_llm_cost(adapter: LangGraphAdapter, output_tokens: int, model: str = "gpt-4o") -> None:
    """Feed the adapter a deterministic LLM call worth a known cost (offline)."""
    run_id = uuid4()
    adapter._handler.on_chat_model_start({"kwargs": {"model": model}}, [[]], run_id=run_id)
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 0,
            "output_tokens": output_tokens,
            "total_tokens": output_tokens,
        },
    )
    response = LLMResult(
        generations=[[ChatGeneration(message=message)]], llm_output={"model_name": model}
    )
    adapter._handler.on_llm_end(response, run_id=run_id)


def test_budget_stops_on_captured_llm_cost() -> None:
    # Issue #77: model spend is now captured. Each LLM call costs $1
    # (100k gpt-4o output tokens at $10/1M), interleaved with *varied* edits that
    # carry no test/exit evidence -> loop and no_progress stay silent, so the only
    # detector that can STOP is budget. The default ceiling ($5, stop at 90%) is
    # crossed mid-run.
    def run(adapter: LangGraphAdapter) -> None:
        for i in range(6):
            edit_file.invoke({"patch": f"change {i}"}, config=adapter.as_config())
            _drive_llm_cost(adapter, output_tokens=100_000)

    events = _capture(run)
    llm_events = [e for e in events if e.kind is EventKind.LLM_CALL]
    assert len(llm_events) == 6
    assert all(e.cost_usd == pytest.approx(1.0) for e in llm_events)
    # Budget saturates; the action detectors are not fooled by the LLM events.
    assert BudgetDetector(max_cost_usd=5.0).evaluate(events).confidence == 1.0
    assert LoopDetector().evaluate(events).confidence == 0.0
    assert NoProgressDetector(coding_anchors()).evaluate(events).confidence == 0.0
    # End to end through the default policy: the run stops, on budget alone.
    assert _stops(events)


def test_budget_silent_under_the_ceiling() -> None:
    # Same shape, total spend $2 < $5 ceiling -> no STOP (no false positive).
    def run(adapter: LangGraphAdapter) -> None:
        for i in range(2):
            edit_file.invoke({"patch": f"change {i}"}, config=adapter.as_config())
            _drive_llm_cost(adapter, output_tokens=100_000)

    events = _capture(run)
    assert BudgetDetector(max_cost_usd=5.0).evaluate(events).confidence == pytest.approx(0.4)
    assert not _stops(events)


# --------------------------------------------------------------------------- #
# Graph-level capture: a real LangGraph StateGraph, not just tool.invoke.
# --------------------------------------------------------------------------- #


def test_loop_detected_through_a_real_stategraph() -> None:
    # Proves capture works when the tool is called from *inside* a compiled
    # StateGraph node: LangGraph threads the adapter's callbacks through the run,
    # so a looping node is caught end to end.
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph

    def node(state: _GraphState) -> _GraphState:
        search.invoke({"query": "same"})
        return {"steps": state["steps"] + 1}

    def should_continue(state: _GraphState) -> str:
        return END if state["steps"] >= 6 else "act"

    builder = StateGraph(_GraphState)
    builder.add_node("act", node)
    builder.add_edge(START, "act")
    builder.add_conditional_edges("act", should_continue, {END: END, "act": "act"})
    graph = builder.compile()

    def run(adapter: LangGraphAdapter) -> None:
        graph.invoke({"steps": 0}, config=adapter.as_config())

    events = _capture(run)
    assert [e.tool for e in events] == ["search"] * 6
    assert LoopDetector().evaluate(events).confidence == pytest.approx(0.96875)
    assert _stops(events)


class _WarnState(TypedDict):
    """Module-level (see _GraphState) messages state with the add_messages reducer."""

    messages: Annotated[list[BaseMessage], add_messages]


def test_act1_warn_injected_into_a_real_stategraph() -> None:
    # ACT-1 end to end: a buffered WARN advisory is delivered into a live
    # StateGraph as a SystemMessage by the actuator's injection node, *appended*
    # (add_messages reducer) before the model node -- without stopping the run.
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph

    from pathlayer.policies.actuation import ActuationContext, EventSnapshot
    from pathlayer.policies.langgraph_actuator import LangGraphActuator

    actuator = LangGraphActuator()
    actuator.attach()
    # Simulate a WARN transition arriving from the pipeline (arbitrated already).
    # The signal's reason carries a secret (as a RetryDetector reason would): the
    # injected advisory must name the signal but never leak the reason (SEC-3).
    secret_reason = "db timeout: password=hunter2"
    actuator.apply_with_context(
        ActuationContext(
            action=Action.WARN,
            fused=0.6,
            reason=secret_reason,
            signals=(Signal(name="no_progress", confidence=0.6, reason=secret_reason),),
            event=EventSnapshot(
                tool="edit_file", kind=EventKind.TOOL_CALL, ts=1.0, index=4, cost_usd=0.0
            ),
            post_mortem=None,
        )
    )

    def agent(state: _WarnState) -> dict[str, list[BaseMessage]]:
        return {"messages": [AIMessage(content="carrying on")]}

    builder = StateGraph(_WarnState)
    builder.add_node("pathlayer_warn", actuator.injection_node())
    builder.add_node("agent", agent)
    builder.add_edge(START, "pathlayer_warn")
    builder.add_edge("pathlayer_warn", "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [HumanMessage(content="do the task")]})

    system = [m for m in result["messages"] if isinstance(m, SystemMessage)]
    assert len(system) == 1
    # The advisory names the signal but never leaks the secret-bearing reason.
    assert "no_progress" in system[0].content
    assert secret_reason not in system[0].content
    # The run completed normally: the agent's own message is still there and the
    # drain was never requested (WARN never halts).
    assert any(isinstance(m, AIMessage) for m in result["messages"])
    assert actuator.control.drain_requested is False


def test_act1_real_detector_warns_then_injects_into_stategraph() -> None:
    # ACT-1 fully end to end: a *real* detector (LoopDetector) drives a *real*
    # Policy across the WARN threshold via the Pipeline + wire() + a real
    # LangGraphActuator, and the resulting advisory is injected into a live
    # StateGraph -- no scripted policy or hand-built context anywhere.
    #
    # Calibration (see LoopDetector.evaluate): identical calls give confidence
    # 0.5, 0.75, 0.875, 0.9375... With tier_thresholds warn=0.3 and
    # stop_threshold=0.9, three identical calls peak at 0.75 -> WARN fires and
    # the run is never stopped (0.75 < 0.9).
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph

    from pathlayer.policies.langgraph_actuator import LangGraphActuator
    from pathlayer.policies.wiring import wire

    policy = Policy(PolicyConfig(stop_threshold=0.9, tier_thresholds={Action.WARN: 0.3}))
    pipeline = Pipeline([LoopDetector()], policy)
    actuator = LangGraphActuator()
    actuator.attach()
    adapter = LangGraphAdapter(wire(pipeline, actuator))
    adapter.attach()

    # Three identical tool calls, captured for real through the adapter.
    for _ in range(3):
        search.invoke({"query": "same"}, config=adapter.as_config())
    adapter.detach()

    # A real WARN was arbitrated and enforced; STOP never fired.
    assert Action.WARN in pipeline.actions
    assert Action.STOP not in pipeline.actions
    assert actuator.control.drain_requested is False

    def agent(state: _WarnState) -> dict[str, list[BaseMessage]]:
        return {"messages": [AIMessage(content="carrying on")]}

    builder = StateGraph(_WarnState)
    builder.add_node("pathlayer_warn", actuator.injection_node())
    builder.add_node("agent", agent)
    builder.add_edge(START, "pathlayer_warn")
    builder.add_edge("pathlayer_warn", "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [HumanMessage(content="do the task")]})

    system = [m for m in result["messages"] if isinstance(m, SystemMessage)]
    assert len(system) == 1
    assert "loop" in system[0].content
    assert actuator.control.drain_requested is False


# --------------------------------------------------------------------------- #
# PERF-2: incremental step() vs batch evaluate()/measure(), on *real* captured
# events (not just hand-written fixtures or the synthetic corpus in
# test_perf2_incremental.py). Closes the gap flagged by the PERF-2 review
# committee: the fold and the anchors are equivalence-tested prefix by prefix
# against traces produced by actually running LangChain tools through the
# adapter, exercising the real (tool, args, result) shapes it emits.
# --------------------------------------------------------------------------- #


def _assert_anchor_step_matches_batch_on_real_events(
    anchor_factory: Callable[[], object], events: list[ToolEvent]
) -> None:
    anchor = anchor_factory()
    state = anchor.create_state()  # type: ignore[attr-defined]
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, snapshot = anchor.step(state, event)  # type: ignore[attr-defined]
        expected_level = anchor.measure(history)  # type: ignore[attr-defined]
        assert snapshot.level == expected_level, (len(history), snapshot.level, expected_level)
        expected_satisfied = (
            anchor.is_satisfied(history) if expected_level is not None else False  # type: ignore[attr-defined]
        )
        assert snapshot.satisfied == expected_satisfied


def _assert_detector_step_matches_batch_on_real_events(
    detector: NoProgressDetector, events: list[ToolEvent]
) -> None:
    state = detector.create_state()
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, step_signal = detector.step(state, event)
        batch_signal = detector.evaluate(tuple(history))
        assert step_signal.confidence == pytest.approx(batch_signal.confidence, abs=1e-9), (
            len(history),
            step_signal,
            batch_signal,
        )
        assert step_signal.reason == batch_signal.reason


def _real_event_traces() -> list[tuple[str, list[ToolEvent]]]:
    """Every captured trace already used by this module's scenario tests."""
    traces: list[tuple[str, list[ToolEvent]]] = []

    def loop_run(adapter: LangGraphAdapter) -> None:
        for _ in range(6):
            search.invoke({"query": "same"}, config=adapter.as_config())

    traces.append(("loop", _capture(loop_run)))

    def retry_run(adapter: LangGraphAdapter) -> None:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="db timeout"):
                flaky.invoke({"x": 1}, config=adapter.as_config())

    traces.append(("retry", _capture(retry_run)))

    def no_progress_run(adapter: LangGraphAdapter) -> None:
        run_tests.invoke({"suite": "all"}, config=adapter.as_config())
        for i in range(6):
            edit_file.invoke({"patch": f"change {i}"}, config=adapter.as_config())
            run_tests.invoke({"suite": "all"}, config=adapter.as_config())

    traces.append(("no_progress_flat", _capture(no_progress_run)))

    def improving_run(adapter: LangGraphAdapter) -> None:
        for failed in (3, 2, 1, 0):
            edit_file.invoke({"patch": f"fix {failed}"}, config=adapter.as_config())
            run_tests_improving.invoke({"failed": failed}, config=adapter.as_config())

    traces.append(("no_progress_improving", _capture(improving_run)))

    def rich_progress_run(adapter: LangGraphAdapter) -> None:
        for index, failed_ids in enumerate((["a", "b"], ["b"], [])):
            edit_file.invoke({"patch": f"rich fix {index}"}, config=adapter.as_config())
            run_tests_rich.invoke(
                {"failed_ids": failed_ids, "run": index}, config=adapter.as_config()
            )

    traces.append(("no_progress_rich", _capture(rich_progress_run)))

    def budget_run(adapter: LangGraphAdapter) -> None:
        for i in range(6):
            edit_file.invoke({"patch": f"change {i}"}, config=adapter.as_config())
            _drive_llm_cost(adapter, output_tokens=100_000)

    traces.append(("budget", _capture(budget_run)))

    return traces


_REAL_TRACES = _real_event_traces()
_REAL_TRACE_IDS = [name for name, _ in _REAL_TRACES]


@pytest.mark.parametrize("name,events", _REAL_TRACES, ids=_REAL_TRACE_IDS)
@pytest.mark.parametrize(
    "anchor_factory", [TestProgressAnchor, ExitCodeProgressAnchor, GitProgressAnchor]
)
def test_anchor_step_matches_batch_on_real_captured_events(
    name: str, events: list[ToolEvent], anchor_factory: Callable[[], object]
) -> None:
    _assert_anchor_step_matches_batch_on_real_events(anchor_factory, events)


@pytest.mark.parametrize("name,events", _REAL_TRACES, ids=_REAL_TRACE_IDS)
def test_no_progress_detector_step_matches_batch_on_real_captured_events(
    name: str, events: list[ToolEvent]
) -> None:
    detector = NoProgressDetector(coding_anchors())
    _assert_detector_step_matches_batch_on_real_events(detector, events)
