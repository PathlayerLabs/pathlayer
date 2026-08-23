"""Smoke test for the LangGraph/LangChain capture adapter (ADP-0).

Skipped entirely unless the optional ``langchain_core`` dependency is present.
We drive capture through real LangChain tools (``tool.invoke`` /
``tool.ainvoke`` with the adapter's config) -- the reliable mechanism that
guarantees the callbacks fire, unlike a custom node calling a plain function.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import tool

from pathlayer import Pipeline
from pathlayer.adapters.langgraph import LangGraphAdapter, _coerce_result
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.resource import ResourceDetector
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import PROGRESS_SUMMARY_TOOL, EventKind, ToolEvent
from pathlayer.policies import Action, Actuator, Policy, wire
from pathlayer.postmortem import PostMortem
from pathlayer.validation.traces import Trace, dump_trace, load_trace


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def boom(x: int) -> int:
    """Always fails."""
    raise ValueError("nope")


@tool
def echo(text: str) -> str:
    """Echo back the text."""
    return text


@tool(PROGRESS_SUMMARY_TOOL)
def pathlayer_progress_summary(completed: str, blocker: str, next_action: str) -> str:
    """Record a structured progress summary (the ACT-2 tool, by its stable name)."""
    return "progress summary recorded"


def _capture() -> tuple[list[ToolEvent], LangGraphAdapter]:
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append)
    adapter.attach()
    return events, adapter


def test_invoke_captures_normalized_event() -> None:
    events, adapter = _capture()

    result = add.invoke({"a": 2, "b": 3}, config=adapter.as_config())

    assert result == 5  # passive observer: real output unchanged
    assert len(events) == 1
    event = events[0]
    assert event.tool == "add"
    assert event.kind is EventKind.TOOL_CALL
    assert event.args == {"a": 2, "b": 3}
    assert event.result == 5
    assert event.run_id is not None
    assert "parent_run_id" in event.metadata


def test_scalar_input_falls_back_to_input_key() -> None:
    # When the tool is called with a non-dict payload, LangChain passes
    # inputs=None; args must fall back to {"input": input_str} (acceptance
    # criterion) rather than dropping the argument.
    events, adapter = _capture()

    result = echo.invoke("hello", config=adapter.as_config())

    assert result == "hello"
    assert len(events) == 1
    assert events[0].args == {"input": "hello"}


def test_ts_is_deterministic_and_ordered() -> None:
    events, adapter = _capture()

    add.invoke({"a": 1, "b": 1}, config=adapter.as_config())
    add.invoke({"a": 2, "b": 2}, config=adapter.as_config())

    assert [e.ts for e in events] == [0, 1]
    # attach() resets the counter so a fresh run is reproducible.
    events.clear()
    adapter.attach()
    add.invoke({"a": 3, "b": 3}, config=adapter.as_config())
    assert events[0].ts == 0


def test_on_tool_error_is_captured_without_breaking_observer() -> None:
    events, adapter = _capture()

    with pytest.raises(ValueError, match="nope"):
        boom.invoke({"x": 1}, config=adapter.as_config())

    assert len(events) == 1
    event = events[0]
    assert event.tool == "boom"
    assert event.result is None
    assert event.metadata["error"] == "nope"
    assert "parent_run_id" in event.metadata


def test_async_invoke_is_captured() -> None:
    # A sync handler must still capture during ainvoke (LangChain runs sync
    # callbacks in a thread pool), so no AsyncCallbackHandler is needed in V1.
    events, adapter = _capture()

    result = asyncio.run(add.ainvoke({"a": 4, "b": 5}, config=adapter.as_config()))

    assert result == 9
    assert len(events) == 1
    assert events[0].tool == "add"
    assert events[0].args == {"a": 4, "b": 5}
    assert events[0].result == 9


def test_detach_drops_pending_state() -> None:
    _, adapter = _capture()
    add.invoke({"a": 1, "b": 2}, config=adapter.as_config())
    adapter.detach()
    assert adapter._pending == {}


def test_pipeline_integration() -> None:
    pipeline = Pipeline([LoopDetector()])
    adapter = LangGraphAdapter(pipeline.sink)
    adapter.attach()

    add.invoke({"a": 1, "b": 2}, config=adapter.as_config())
    add.invoke({"a": 3, "b": 4}, config=adapter.as_config())

    assert len(pipeline.history) == 2
    assert [e.tool for e in pipeline.history] == ["add", "add"]


def test_summary_tool_is_captured_as_progress_summary_kind() -> None:
    # ACT-2: the Pathlayer summary tool is classified as PROGRESS_SUMMARY at
    # capture (by its stable name), carrying the three fields in args, at zero
    # cost. Pure capture -- no tier decision descends into the adapter.
    events, adapter = _capture()
    args = {
        "completed": "added a guard",
        "blocker": "parser fails on empty input",
        "next_action": "write a regression test",
    }
    pathlayer_progress_summary.invoke(args, config=adapter.as_config())

    assert len(events) == 1
    event = events[0]
    assert event.kind is EventKind.PROGRESS_SUMMARY
    assert event.tool == PROGRESS_SUMMARY_TOOL
    assert event.args == args
    assert event.cost_usd == 0.0


def test_detectors_ignore_captured_progress_summary_events() -> None:
    # A forced summary must not add a fake stagnation step nor fool a repetition
    # detector: the action detectors filter to TOOL_CALL, so repeated identical
    # summaries stay invisible to them, and budget stays at zero cost.
    events, adapter = _capture()
    for _ in range(6):
        pathlayer_progress_summary.invoke(
            {"completed": "c", "blocker": "b", "next_action": "n"},
            config=adapter.as_config(),
        )

    assert len(events) == 6
    assert all(e.kind is EventKind.PROGRESS_SUMMARY for e in events)
    assert LoopDetector().evaluate(events).confidence == 0.0
    assert RetryDetector().evaluate(events).confidence == 0.0
    assert NoProgressDetector(coding_anchors()).evaluate(events).confidence == 0.0
    assert BudgetDetector(max_cost_usd=5.0).evaluate(events).confidence == 0.0


def test_progress_summary_survives_trace_round_trip(tmp_path: Path) -> None:
    # VAL-1 compat: the new event kind serializes and reloads unchanged.
    events, adapter = _capture()
    pathlayer_progress_summary.invoke(
        {"completed": "c", "blocker": "b", "next_action": "n"}, config=adapter.as_config()
    )
    trace = Trace(label="healthy", events=events)
    path = tmp_path / "summary.json"
    dump_trace(trace, path)
    reloaded = load_trace(path)
    assert reloaded.events == events
    assert reloaded.events[0].kind is EventKind.PROGRESS_SUMMARY


def test_val1_round_trip(tmp_path: Path) -> None:
    events, adapter = _capture()
    add.invoke({"a": 1, "b": 2}, config=adapter.as_config())
    with pytest.raises(ValueError, match="nope"):
        boom.invoke({"x": 9}, config=adapter.as_config())

    trace = Trace(label="healthy", events=events)
    path = tmp_path / "captured.json"
    dump_trace(trace, path)

    reloaded = load_trace(path)
    assert reloaded.events == events


def test_end_or_error_without_matching_start_is_ignored() -> None:
    # Defensive: an end/error callback whose start we never saw must not emit
    # (and must not raise) -- e.g. callbacks wired mid-run.
    events, adapter = _capture()
    adapter._record_end("unknown-run", 42)
    adapter._record_error("unknown-run", ValueError("x"))
    assert events == []


def test_coerce_result_stringifies_non_json_objects() -> None:
    class Weird:
        def __str__(self) -> str:
            return "weird"

    assert _coerce_result(5) == 5
    assert _coerce_result(None) is None
    assert _coerce_result({"k": "v"}) == {"k": "v"}
    assert _coerce_result([1, 2]) == [1, 2]
    assert _coerce_result(Weird()) == "weird"


def test_coerce_result_unwraps_tool_message() -> None:
    from langchain_core.messages import ToolMessage

    # A ToolNode wraps a dict/list return as a JSON *string* in `content`; the
    # adapter must parse it back so TestProgressAnchor (which needs a Mapping)
    # is not blinded.
    assert _coerce_result(ToolMessage(content='{"passed": 11, "failed": 3}', tool_call_id="1")) == {
        "passed": 11,
        "failed": 3,
    }
    assert _coerce_result(ToolMessage(content="[1, 2]", tool_call_id="1")) == [1, 2]
    # A structured artifact (content_and_artifact tools) is the real object.
    assert _coerce_result(
        ToolMessage(content="ignored", artifact={"failed": 0}, tool_call_id="1")
    ) == {"failed": 0}
    # Plain text stays a string; a scalar-looking string is NOT coerced to a number.
    assert _coerce_result(ToolMessage(content="ok", tool_call_id="1")) == "ok"
    assert _coerce_result(ToolMessage(content="11", tool_call_id="1")) == "11"
    # Malformed JSON object falls back to the raw string (no crash).
    assert _coerce_result(ToolMessage(content="{not json}", tool_call_id="1")) == "{not json}"
    # Pathologically nested JSON makes json.loads raise RecursionError (not a
    # ValueError); it must be caught too, falling back to the raw string.
    nested = "[" * 20000 + "]" * 20000
    assert _coerce_result(ToolMessage(content=nested, tool_call_id="1")) == nested


def test_toolnode_style_invoke_captures_structured_result() -> None:
    # Invoking a tool with a full ToolCall (exactly what ToolNode does) hands the
    # callback a ToolMessage; the adapter must still capture the structured dict
    # so a real ToolNode-based agent drives progress anchors.
    @tool
    def run_tests(target: str) -> dict[str, int]:
        """Flat red suite."""
        return {"passed": 11, "failed": 3}

    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append)
    adapter.attach()
    run_tests.invoke(
        {"name": "run_tests", "args": {"target": "t"}, "id": "c1", "type": "tool_call"},
        config=adapter.as_config(),
    )
    adapter.detach()
    assert len(events) == 1
    assert events[0].result == {"passed": 11, "failed": 3}


# --------------------------------------------------------------------------- #
# Concurrency: thread-safety is the adapter's headline feature. A single sync
# handler serves async runs (LangChain calls sync callbacks from a thread pool),
# so concurrent callbacks must never lose, duplicate, or interleave events, and
# ``ts`` must stay unique and contiguous. Synchronization is deterministic
# (gather / barrier), never timing-based, so these tests cannot flake.
# --------------------------------------------------------------------------- #


def test_concurrent_ainvoke_loses_no_events_and_ts_stays_contiguous() -> None:
    events: list[ToolEvent] = []
    threads_seen: set[int] = set()
    bookkeeping = threading.Lock()

    def sink(event: ToolEvent) -> None:
        # Guards only the test's own structures; the adapter already serializes
        # emit() under its lock. Recording the thread proves callbacks really
        # arrive off the executor pool, not the main thread.
        with bookkeeping:
            threads_seen.add(threading.get_ident())
            events.append(event)

    adapter = LangGraphAdapter(sink)
    adapter.attach()

    n = 64

    async def drive() -> None:
        await asyncio.gather(
            *(add.ainvoke({"a": i, "b": i}, config=adapter.as_config()) for i in range(n))
        )

    asyncio.run(drive())
    adapter.detach()

    assert len(events) == n  # nothing dropped
    assert len({e.run_id for e in events}) == n  # nothing duplicated
    # ts is unique and contiguous from 0 regardless of completion interleaving.
    assert sorted(e.ts for e in events) == list(range(n))
    assert threads_seen  # callbacks fired (typically off the executor pool)


def test_concurrent_record_start_emit_preserves_integrity_under_contention() -> None:
    events: list[ToolEvent] = []
    bookkeeping = threading.Lock()

    def sink(event: ToolEvent) -> None:
        with bookkeeping:
            events.append(event)

    adapter = LangGraphAdapter(sink)
    adapter.attach()

    n = 64
    # All workers block on the barrier, then hammer the adapter at once -- this
    # maximizes lock contention without any sleep, so the test is deterministic.
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        run_id = f"run-{i}"
        barrier.wait()
        adapter._record_start(run_id, None, "tool", {"i": i})
        adapter._emit_paired(run_id, result=i, error=None)

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(worker, range(n)))

    assert len(events) == n  # every paired call emitted exactly once
    assert len({e.run_id for e in events}) == n  # no run_id lost or collided
    assert sorted(e.ts for e in events) == list(range(n))  # ts unique + contiguous
    assert adapter._pending == {}  # all starts paired and popped


def test_pending_map_is_bounded_when_starts_are_never_paired() -> None:
    # A start with no matching end/error (cancelled coroutine, dropped callback)
    # would otherwise leak forever on a long-running attached agent. The FIFO cap
    # evicts the oldest so the map stays bounded.
    from pathlayer.adapters.langgraph import _MAX_PENDING

    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append)
    adapter.attach()

    for i in range(_MAX_PENDING + 50):
        adapter._record_start(f"run-{i}", None, "tool", {"i": i})

    assert len(adapter._pending) == _MAX_PENDING
    # The 50 oldest were evicted; the newest survive in insertion order.
    assert "run-0" not in adapter._pending
    assert f"run-{_MAX_PENDING + 49}" in adapter._pending
    assert events == []  # eviction never emits; it only frees memory


# --------------------------------------------------------------------------- #
# LLM cost capture (issue #77): on_llm_start/end -> a dedicated LLM_CALL event
# carrying token cost in cost_usd, so BudgetDetector can guard model spend.
# Driven by feeding the handler hand-built LLMResults -- deterministic, offline,
# no API key -- which exercises both the handler and the adapter token/price path.
# --------------------------------------------------------------------------- #

from uuid import uuid4  # noqa: E402

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, Generation, LLMResult  # noqa: E402


class _TimerHandle:
    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _ManualScheduler:
    def __init__(self) -> None:
        self.handle: _TimerHandle | None = None
        self.delay: float | None = None
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def call_later(self, delay_seconds: float, callback: Callable[[], None]) -> _TimerHandle:
        self.delay = delay_seconds
        self.handle = _TimerHandle(callback)
        return self.handle

    def fire(self) -> None:
        assert self.handle is not None and not self.handle.cancelled
        assert self.delay is not None
        self.now += self.delay
        self.handle.callback()


class _StopActuator(Actuator):
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        self.actions.append(action)


def _chat_result(input_tokens: int, output_tokens: int, model: str | None = "gpt-4o") -> LLMResult:
    """A chat-model LLMResult carrying usage_metadata on its AIMessage."""
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    llm_output = {"model_name": model} if model is not None else {}
    return LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=llm_output)


def _drive_llm(adapter: LangGraphAdapter, response: LLMResult, *, serialized: dict | None) -> None:
    run_id = uuid4()
    adapter._handler.on_llm_start(serialized, ["prompt"], run_id=run_id)
    adapter._handler.on_llm_end(response, run_id=run_id)


def test_root_llm_start_arms_watchdog_before_in_flight_call_finishes() -> None:
    pipeline = Pipeline([ResourceDetector(max_duration_seconds=10)], Policy())
    actuator = _StopActuator()
    scheduler = _ManualScheduler()
    sink = wire(
        pipeline,
        actuator,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    adapter = LangGraphAdapter(sink)
    adapter.attach()

    run_id = uuid4()
    adapter._handler.on_llm_start({"kwargs": {"model": "gpt-4o"}}, ["prompt"], run_id=run_id)
    assert [event.kind for event in pipeline.history] == [EventKind.RUN_START]

    scheduler.fire()
    assert [event.kind for event in pipeline.history] == [
        EventKind.RUN_START,
        EventKind.RUN_TIMEOUT,
    ]
    assert EventKind.LLM_CALL not in [event.kind for event in pipeline.history]
    assert actuator.actions == [Action.STOP]


@pytest.mark.parametrize("fails", [False, True])
def test_completed_root_llm_cancels_watchdog(fails: bool) -> None:
    # The watchdog is cancelled by detach() (called right after the root call
    # completes, matching the README's try/finally pattern) -- not by the root
    # callback itself, since one attach() may legitimately cover several
    # independent root-level calls (see
    # test_two_independent_root_calls_share_one_watchdog below).
    pipeline = Pipeline([ResourceDetector(max_duration_seconds=10)], Policy())
    scheduler = _ManualScheduler()
    sink = wire(
        pipeline,
        _StopActuator(),
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    adapter = LangGraphAdapter(sink)
    adapter.attach()
    run_id = uuid4()
    adapter._handler.on_llm_start({"kwargs": {"model": "gpt-4o"}}, ["prompt"], run_id=run_id)

    if fails:
        adapter._handler.on_llm_error(RuntimeError("provider failed"), run_id=run_id)
    else:
        adapter._handler.on_llm_end(_chat_result(1, 1, "gpt-4o"), run_id=run_id)
    adapter.detach()

    assert scheduler.handle is not None and scheduler.handle.cancelled
    scheduler.now = 20.0
    scheduler.handle.callback()
    assert EventKind.RUN_TIMEOUT not in [event.kind for event in pipeline.history]


def test_two_independent_root_calls_share_one_watchdog() -> None:
    # A root-level callback ending (e.g. the first of two standalone
    # add.invoke() calls under one attach(), as in test_pipeline_integration)
    # must not silently kill the watchdog for the rest of the logical run --
    # only detach() may end it.
    pipeline = Pipeline([ResourceDetector(max_duration_seconds=10)], Policy())
    scheduler = _ManualScheduler()
    sink = wire(
        pipeline,
        _StopActuator(),
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    adapter = LangGraphAdapter(sink)
    adapter.attach()

    add.invoke({"a": 1, "b": 2}, config=adapter.as_config())
    assert scheduler.handle is not None and not scheduler.handle.cancelled

    add.invoke({"a": 3, "b": 4}, config=adapter.as_config())
    assert scheduler.handle is not None and not scheduler.handle.cancelled
    assert [event.kind for event in pipeline.history] == [
        EventKind.RUN_START,
        EventKind.TOOL_CALL,
        EventKind.TOOL_CALL,
    ]

    adapter.detach()
    assert scheduler.handle is not None and scheduler.handle.cancelled


def test_llm_call_is_captured_with_cost() -> None:
    events, adapter = _capture()
    _drive_llm(
        adapter, _chat_result(1000, 1000, "gpt-4o"), serialized={"kwargs": {"model": "gpt-4o"}}
    )

    assert len(events) == 1
    event = events[0]
    assert event.kind is EventKind.LLM_CALL
    assert event.tool == "gpt-4o"
    # gpt-4o: (1000 * 2.5 + 1000 * 10) / 1e6 = 0.0125
    assert event.cost_usd == pytest.approx(0.0125)
    assert event.metadata["input_tokens"] == 1000
    assert event.metadata["output_tokens"] == 1000
    assert "parent_run_id" in event.metadata


def test_llm_cost_falls_back_to_llm_output_token_usage() -> None:
    # A non-chat Generation has no usage_metadata; tokens come from llm_output.
    events, adapter = _capture()
    response = LLMResult(
        generations=[[Generation(text="x")]],
        llm_output={
            "model_name": "gpt-4o",
            "token_usage": {"prompt_tokens": 2000, "completion_tokens": 0},
        },
    )
    _drive_llm(adapter, response, serialized=None)

    assert len(events) == 1
    assert events[0].cost_usd == pytest.approx(0.005)  # 2000 * 2.5 / 1e6


def test_llm_call_with_no_tokens_costs_zero_but_still_emits() -> None:
    events, adapter = _capture()
    response = LLMResult(generations=[[Generation(text="x")]], llm_output={"model_name": "gpt-4o"})
    _drive_llm(adapter, response, serialized=None)

    assert len(events) == 1
    assert events[0].cost_usd == 0.0


def test_unknown_model_costs_zero() -> None:
    events, adapter = _capture()
    _drive_llm(adapter, _chat_result(1000, 1000, "myorg-secret"), serialized=None)

    assert events[0].cost_usd == 0.0
    assert events[0].tool == "myorg-secret"


def test_missing_model_uses_placeholder_tool_and_zero_cost() -> None:
    events, adapter = _capture()
    _drive_llm(adapter, _chat_result(1000, 1000, model=None), serialized=None)

    assert events[0].tool == "<llm>"
    assert events[0].cost_usd == 0.0


def test_model_from_start_used_when_response_omits_it() -> None:
    # llm_output has no model_name -> fall back to the model seen at on_llm_start.
    events, adapter = _capture()
    response = _chat_result(1000, 1000, model=None)  # llm_output = {}
    _drive_llm(adapter, response, serialized={"kwargs": {"model": "claude-3-5-sonnet"}})

    assert events[0].tool == "claude-3-5-sonnet"
    # claude-3-5-sonnet: (1000 * 3 + 1000 * 15) / 1e6 = 0.018
    assert events[0].cost_usd == pytest.approx(0.018)


def test_on_chat_model_start_registers_pending() -> None:
    events, adapter = _capture()
    run_id = uuid4()
    adapter._handler.on_chat_model_start({"kwargs": {"model": "gpt-4o"}}, [[]], run_id=run_id)
    adapter._handler.on_llm_end(_chat_result(0, 1000, "gpt-4o"), run_id=run_id)

    assert len(events) == 1
    assert events[0].cost_usd == pytest.approx(0.01)  # 1000 * 10 / 1e6


def test_on_llm_error_drops_pending_without_emitting() -> None:
    events, adapter = _capture()
    run_id = uuid4()
    adapter._handler.on_llm_start({"kwargs": {"model": "gpt-4o"}}, ["p"], run_id=run_id)
    adapter._handler.on_llm_error(RuntimeError("boom"), run_id=run_id)

    assert events == []
    assert adapter._pending_llm == {}


def test_llm_end_without_matching_start_is_ignored() -> None:
    events, adapter = _capture()
    adapter._record_llm_end("unknown-run", _chat_result(10, 10, "gpt-4o"))
    assert events == []


def test_llm_and_tool_events_share_one_ts_sequence() -> None:
    events, adapter = _capture()
    add.invoke({"a": 1, "b": 1}, config=adapter.as_config())
    _drive_llm(adapter, _chat_result(10, 10, "gpt-4o"), serialized=None)
    assert [e.ts for e in events] == [0, 1]
    assert [e.kind for e in events] == [EventKind.TOOL_CALL, EventKind.LLM_CALL]


def test_detach_clears_pending_llm() -> None:
    _, adapter = _capture()
    adapter._handler.on_llm_start({"kwargs": {"model": "gpt-4o"}}, ["p"], run_id=uuid4())
    adapter.detach()
    assert adapter._pending_llm == {}


def test_pending_llm_map_is_bounded() -> None:
    from pathlayer.adapters.langgraph import _MAX_PENDING

    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append)
    adapter.attach()
    for i in range(_MAX_PENDING + 50):
        adapter._record_llm_start(f"run-{i}", None, "gpt-4o")

    assert len(adapter._pending_llm) == _MAX_PENDING
    assert "run-0" not in adapter._pending_llm
    assert events == []
