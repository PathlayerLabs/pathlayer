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
                     case) while ``LoopDetector`` stays silent. In V1 (NOP-2)
                     this signal is **advisory**: the default ``Policy`` logs it
                     but never STOPs on it alone (``DEFAULT_ADVISORY_SIGNALS``),
                     so we prove the detector saturates *and* that only the
                     policy inhibits the stop (a non-advisory policy does stop).
* (d) BUDGET      -- documents a real limitation: the ADP-0 adapter never
                     populates ``cost_usd``, so ``BudgetDetector`` cannot fire
                     from captured events alone.

Each scenario carries a numeric assertion on the detector confidence *and* a
negative (healthy) counterpart, so a false positive would fail the suite too.
Skipped entirely unless the optional ``langchain_core`` dependency is present.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import tool

from pathlayer import Pipeline
from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent
from pathlayer.policies import Policy
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

    Defaults to the production ``Policy`` (no_progress advisory). Pass a custom
    policy to probe a different arbitration (e.g. no advisory exclusion).
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
    # (b) no_progress is advisory in V1 (NOP-2): the default Policy logs it but
    # never STOPs on it alone (it cannot yet tell "task done" from "stuck").
    assert not _stops(events)
    # (c) Strip the advisory exclusion and the *same* signal stops the run --
    # proving only the policy inhibits the stop, not a weak detector.
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


# --------------------------------------------------------------------------- #
# (d) BUDGET -- documented limitation of the ADP-0 adapter
# --------------------------------------------------------------------------- #


def test_budget_cannot_fire_because_adapter_drops_cost() -> None:
    # FINDING: the LangGraph adapter only hooks tool callbacks and never sets
    # ToolEvent.cost_usd (no on_llm_end token-usage capture in ADP-0). So no
    # matter how many calls a run makes, captured cost stays 0 and BudgetDetector
    # is structurally unable to fire from adapter events alone. This is a real
    # gap to close before Pathlayer can guard spend on a LangGraph agent.
    def run(adapter: LangGraphAdapter) -> None:
        for _ in range(20):
            search.invoke({"query": "anything"}, config=adapter.as_config())

    events = _capture(run)
    assert all(e.cost_usd == 0.0 for e in events)
    assert BudgetDetector(max_cost_usd=5.0).evaluate(events).confidence == 0.0


def test_budget_detector_itself_works_when_cost_is_present() -> None:
    # Control: the detector is correct; the gap is purely in capture. With cost
    # injected, the same detector saturates -- proving the missing piece is the
    # adapter, not the BudgetDetector.
    priced = [ToolEvent(tool="llm_call", cost_usd=1.0) for _ in range(5)]
    assert BudgetDetector(max_cost_usd=5.0).evaluate(priced).confidence == 1.0


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
