"""Smoke test for the LangGraph/LangChain capture adapter (ADP-0).

Skipped entirely unless the optional ``langchain_core`` dependency is present.
We drive capture through real LangChain tools (``tool.invoke`` /
``tool.ainvoke`` with the adapter's config) -- the reliable mechanism that
guarantees the callbacks fire, unlike a custom node calling a plain function.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import tool

from pathlayer import Pipeline
from pathlayer.adapters.langgraph import LangGraphAdapter, _coerce_result
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import EventKind, ToolEvent
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
