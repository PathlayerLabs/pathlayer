"""The flagship no-progress "slow burn", caught inside a live agent.

The differentiated case, the pattern behind a publicly reported $47k incident
(see docs/no-progress-demo.md): an agent looks busy and changes
its actions every single turn -- a genuinely different edit, then the test
suite, over and over -- but the ground-truth progress signal (tests passing)
never moves. Repetition detectors stay silent because nothing repeats; only
``NoProgressDetector`` (CUSUM on the flat test result) catches the stall.

This runs a *real* compiled LangGraph ``StateGraph`` (no LLM, no network: the
model node is scripted and deterministic) through the real runtime path::

    LangGraphAdapter (capture) -> wire(pipeline, actuator) -> build_pipeline()
        -> NoProgressDetector -> Policy (graded ladder) -> LangGraphActuator

The graded intervention is shown end to end:

    warn      -> a non-interrupting advisory is injected into the agent context
    summarize -> a strict gate forces the agent to emit a structured progress
                 summary before any other tool runs (the agent complies here)
    stop      -> progress still never resumes, so the run is drained and halted

``loop`` and ``retry`` are wired **advisory** (observe-only): the demo prints
their (flat) confidence to prove they stay silent, but only ``no_progress``
drives the ladder. This is the calibrated separation for the no-progress story
(see tests/test_demo2.py).

A healthy counterpart -- same shape of activity, but the failing-test count
strictly decreases to zero -- is run through the identical graph and is never
warned, summarized or stopped: a false positive (intervening on an agent that
is really progressing) is worse than no Pathlayer at all.

Deterministic, offline, no LLM, no network. Requires the langgraph extra
(``pip install "pathlayer[langgraph]"``). Run::

    python examples/demo_no_progress.py
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import PROGRESS_SUMMARY_TOOL, EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action
from pathlayer.policies.langgraph_actuator import (
    MODEL_ROUTE,
    TOOLS_ROUTE,
    LangGraphActuator,
)

# Ten patch+test pairs (20 tool events). The no_progress CUSUM rises ~0.1 per
# flat test result and saturates near event 12; with escalate_ticks=2 the ladder
# needs 14 events (7 pairs) to actually escalate to STOP -- a 12-event trace only
# reaches SUMMARIZE (both boundaries pinned in tests/test_demo2.py). Ten pairs
# gives a clean warn -> summarize -> stop margin.
CHURN_PAIRS = 10
# Safety net: bound the scripted model so a (hypothetical) failure to STOP can
# never hang the demo. STOP fires well before this.
MODEL_TURN_CAP = 60
# Without Pathlayer this slow burn would keep going; we bound the "naive" count
# only so the before/after story terminates, and label it "still going".
NAIVE_CAP = 200


class DemoState(TypedDict):
    """Message state with the standard append reducer (module-level so LangGraph
    can resolve the string annotation under ``from __future__ annotations``)."""

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def edit_file(patch: str) -> str:
    """Apply a genuinely different edit each turn (defeats repetition detectors)."""
    return "ok"


@tool
def run_tests(target: str) -> dict[str, int]:
    """Report a FLAT test result: ground-truth progress never moves."""
    return {"passed": 11, "failed": 3}


@tool
def run_tests_improving(target: str, failed: int) -> dict[str, int]:
    """Report a strictly improving test result (genuine progress)."""
    return {"passed": 14 - failed, "failed": failed}


def demo_config() -> PathlayerConfig:
    """The calibrated demo policy: no_progress drives the ladder; loop/retry
    are advisory (observe-only). Thresholds verified against the healthy corpus
    in tests/test_demo2.py (zero STOP, zero SUMMARIZE on healthy traces).
    """
    config = PathlayerConfig()
    config.policy.advisory_signals = frozenset({LoopDetector.name, RetryDetector.name})
    config.policy.tier_thresholds = {"warn": 0.3, "summarize": 0.5}
    config.policy.stop_threshold = 0.9
    config.policy.escalate_ticks = 2
    return config


def _pathlayer_wants_summary(messages: list[BaseMessage]) -> bool:
    """True if Pathlayer's most recent instruction asks for a summary the agent
    has not answered yet -- observed purely from the message stream (an injected
    ``SystemMessage`` or a blocking ``ToolMessage``), the way a real agent would.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage) and any(
            call.get("name") == PROGRESS_SUMMARY_TOOL for call in message.tool_calls
        ):
            return False  # already summarized since the last request
        content = getattr(message, "content", "")
        if (
            isinstance(message, (SystemMessage, ToolMessage))
            and isinstance(content, str)
            and PROGRESS_SUMMARY_TOOL in content
        ):
            return True
    return False


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _stuck_model(turn: dict[str, int]):  # type: ignore[no-untyped-def]
    """A scripted, cooperative agent that never actually makes progress.

    Each turn it makes a *different* edit then runs the suite. When Pathlayer
    arms the summary gate and asks for a checkpoint, it complies once (emitting a
    structured summary), then goes right back to the same unproductive churn --
    so no_progress stays saturated and the ladder escalates to STOP.
    """

    def model(state: DemoState) -> dict[str, list[BaseMessage]]:
        messages = state["messages"]
        if _pathlayer_wants_summary(messages):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            _tool_call(
                                PROGRESS_SUMMARY_TOOL,
                                {
                                    "completed": "tried six different parse.py patches",
                                    "blocker": "3 tests still fail; none of my edits move them",
                                    "next_action": "try another edit to the delimiter handling",
                                },
                                f"summary-{turn['n']}",
                            )
                        ],
                    )
                ]
            }
        n = turn["n"]
        turn["n"] = n + 1
        if n >= MODEL_TURN_CAP:
            return {"messages": [AIMessage(content="giving up")]}
        if n % 2 == 0:
            call = _tool_call("edit_file", {"patch": f"attempt {n // 2}"}, f"edit-{n}")
        else:
            call = _tool_call("run_tests", {"target": "tests/test_parse.py"}, f"test-{n}")
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    return model


def _healthy_model(turn: dict[str, int]):  # type: ignore[no-untyped-def]
    """Same activity shape, but every turn genuinely fixes one failing test, so
    the suite improves monotonically to zero and the agent finishes on its own.
    """
    schedule = [3, 2, 1, 0]

    def model(state: DemoState) -> dict[str, list[BaseMessage]]:
        n = turn["n"]
        pair = n // 2
        if pair >= len(schedule):
            return {"messages": [AIMessage(content="all tests pass")]}
        turn["n"] = n + 1
        if n % 2 == 0:
            call = _tool_call("edit_file", {"patch": f"real fix {pair}"}, f"edit-{n}")
        else:
            call = _tool_call(
                "run_tests_improving",
                {"target": "tests/test_parse.py", "failed": schedule[pair]},
                f"test-{n}",
            )
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    return model


def _build_graph(model, actuator: LangGraphActuator, sink, tools: list):  # type: ignore[no-untyped-def]
    """Wire the standard Pathlayer graph: inject warnings, gate summaries, run
    tools -- START -> warn -> model -> gate -> (tools -> warn | model).

    Uses the idiomatic ``langgraph.prebuilt.ToolNode``: the LangGraph adapter unwraps
    the ``ToolMessage`` a ``ToolNode`` produces, so ``TestProgressAnchor`` still
    sees the structured ``{passed, failed}`` dict and no_progress fires.
    """

    def route_after_model(state: DemoState) -> str:
        last = state["messages"][-1]
        return "gate" if isinstance(last, AIMessage) and last.tool_calls else END

    builder = StateGraph(DemoState)
    builder.add_node("warn", actuator.injection_node())
    builder.add_node("model", model)
    builder.add_node("gate", actuator.summary_gate_node(sink))
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "warn")
    builder.add_edge("warn", "model")
    builder.add_conditional_edges("model", route_after_model, {"gate": "gate", END: END})
    builder.add_conditional_edges(
        "gate", actuator.summary_route(), {TOOLS_ROUTE: "tools", MODEL_ROUTE: "model"}
    )
    builder.add_edge("tools", "warn")
    return builder.compile()


def _run_live(model, tools: list) -> tuple[Pipeline, list[ToolEvent], bool]:  # type: ignore[no-untyped-def]
    """Run the scripted agent through the real Pathlayer runtime path.

    Returns the pipeline (decisions/history), every captured event, and whether
    the run was drained (STOP enforced).
    """
    pipeline = build_pipeline(demo_config())
    actuator = LangGraphActuator()
    actuator.attach()
    wire_sink = _wire(pipeline, actuator)

    captured: list[ToolEvent] = []

    def sink(event: ToolEvent) -> None:
        captured.append(event)
        wire_sink(event)

    adapter = LangGraphAdapter(sink)
    adapter.attach()
    # The summary tool is per-actuator (executing it lifts the gate), so it is
    # added to the tool set here rather than by the caller.
    graph = _build_graph(model, actuator, sink, [*tools, actuator.summary_tool()])

    drained = False
    config = {**adapter.as_config(), "recursion_limit": 4 * MODEL_TURN_CAP}
    try:
        graph.invoke(
            {"messages": [HumanMessage(content="fix the failing parser tests")]},
            control=actuator.control,
            config=config,
        )
    except GraphDrained:
        drained = True
    finally:
        adapter.detach()
    return pipeline, captured, drained


def _wire(pipeline: Pipeline, actuator: LangGraphActuator):  # type: ignore[no-untyped-def]
    # Imported lazily so the module docstring's "requires langgraph" note is the
    # only import-time coupling; wire() itself is framework-agnostic.
    from pathlayer.policies.wiring import wire

    return wire(pipeline, actuator)


def _action_events(events: list[ToolEvent]) -> list[ToolEvent]:
    """Only the events detectors actually score (plain TOOL_CALLs): excludes the
    captured PROGRESS_SUMMARY (and any synthetic gate events)."""
    return [event for event in events if event.kind is EventKind.TOOL_CALL]


def _max_confidence(detector, events: list[ToolEvent]) -> float:  # type: ignore[no-untyped-def]
    """Highest confidence the detector ever reached over the run (for reporting
    that loop/retry stay flat while no_progress climbs)."""
    highest = 0.0
    for i in range(1, len(events) + 1):
        highest = max(highest, detector.evaluate(events[:i]).confidence)
    return highest


def _transitions(pipeline: Pipeline) -> list[str]:
    """The ladder transitions actually emitted, de-duplicated in order."""
    seen: list[str] = []
    for action in pipeline.actions:
        if action is Action.NONE:
            continue
        if not seen or seen[-1] != action.name:
            seen.append(action.name)
    return seen


def _first_action_index(pipeline: Pipeline) -> int | None:
    for index, action in enumerate(pipeline.actions):
        if action is not Action.NONE:
            return index
    return None


def scenario_stuck() -> None:
    print("--- SLOW BURN: busy agent, zero progress (the differentiated case) ---")
    pipeline, events, drained = _run_live(_stuck_model({"n": 0}), [edit_file, run_tests])
    action_events = _action_events(events)

    loop_max = _max_confidence(LoopDetector(), action_events)
    retry_max = _max_confidence(RetryDetector(), action_events)
    nop_max = _max_confidence(NoProgressDetector(coding_anchors()), action_events)
    summarized = any(event.kind is EventKind.PROGRESS_SUMMARY for event in events)
    first = _first_action_index(pipeline)
    processed = len(pipeline.history)

    print(f"Without Pathlayer: {NAIVE_CAP}+ actions and still going (demo cap {NAIVE_CAP}).")
    print(f"With Pathlayer:    {'STOP (run drained)' if drained else 'NO STOP (UNEXPECTED)'}.")
    print(f"   Repetition detectors : loop={loop_max:.2f}  retry={retry_max:.2f} (silent)")
    print(f"   No progress detected : no_progress={nop_max:.2f} (rises and saturates)")
    if first is not None:
        # The index is zero-based; people count actions from 1.
        print(f"   First intervention   : action {first + 1}")
    print(f"   Graded ladder        : {' -> '.join(_transitions(pipeline)) or 'none'}")
    summary_state = "yes (pathlayer_progress_summary)" if summarized else "no"
    print(f"   Forced summary       : {summary_state}")
    pm = pipeline.post_mortem
    if pm is not None:
        print(f"   STOP reason          : {pm.reason}")
    avoided = NAIVE_CAP - processed
    print(f"   Work avoided         : ~{avoided} of {NAIVE_CAP} actions ({processed} processed).\n")


def scenario_healthy() -> None:
    print("--- HEALTHY: same activity, but the tests improve (counterexample) ---")
    pipeline, events, drained = _run_live(
        _healthy_model({"n": 0}), [edit_file, run_tests_improving]
    )
    actions = [action for action in pipeline.actions if action is not Action.NONE]
    nop_max = _max_confidence(NoProgressDetector(coding_anchors()), _action_events(events))

    if drained or actions:
        print(f"With Pathlayer:    intervention (FALSE POSITIVE): {_transitions(pipeline)}.\n")
        return
    print(
        f"With Pathlayer:    {len(pipeline.history)} actions, no_progress={nop_max:.2f}, "
        "no intervention. The agent finishes on its own.\n"
    )


def main() -> None:
    print("=" * 64)
    print("PATHLAYER DEMO - a no-progress slow burn detected and stopped")
    print("=" * 64 + "\n")
    scenario_stuck()
    scenario_healthy()


if __name__ == "__main__":
    main()
