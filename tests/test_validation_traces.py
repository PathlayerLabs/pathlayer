"""VAL-1: tests for the on-disk trace format, loader and writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathlayer.events import EventKind, ToolEvent
from pathlayer.validation import (
    VALID_LABELS,
    Trace,
    dump_trace,
    event_from_dict,
    event_to_dict,
    load_trace,
    trace_from_dict,
    trace_to_dict,
)

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def test_trace_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="invalid trace label"):
        Trace(label="exploded")


@pytest.mark.parametrize("label", VALID_LABELS)
def test_trace_accepts_known_labels(label: str) -> None:
    assert Trace(label=label).label == label


def test_event_to_dict_omits_defaults() -> None:
    data = event_to_dict(ToolEvent(tool="list_dir", args={"path": "src"}))
    assert data == {"tool": "list_dir", "args": {"path": "src"}}


def test_event_to_dict_includes_non_default_fields() -> None:
    event = ToolEvent(
        tool="spawn",
        args={"role": "worker"},
        result={"ok": True},
        cost_usd=0.5,
        ts=12.0,
        kind=EventKind.SUBAGENT_SPAWN,
        run_id="run-1",
        metadata={"source": "test"},
    )
    data = event_to_dict(event)
    assert data == {
        "tool": "spawn",
        "args": {"role": "worker"},
        "result": {"ok": True},
        "cost_usd": 0.5,
        "ts": 12.0,
        "kind": "subagent_spawn",
        "run_id": "run-1",
        "metadata": {"source": "test"},
    }


def test_event_round_trip_minimal() -> None:
    event = ToolEvent(tool="read_file", args={"path": "a.py"})
    assert event_from_dict(event_to_dict(event)) == event


def test_event_round_trip_full() -> None:
    event = ToolEvent(
        tool="restart",
        args={"reason": "crash"},
        result="ok",
        cost_usd=1.25,
        ts=99.5,
        kind=EventKind.RUN_RESTART,
        run_id="run-7",
        metadata={"attempt": 3},
    )
    assert event_from_dict(event_to_dict(event)) == event


def test_event_from_dict_defaults_missing_fields() -> None:
    event = event_from_dict({"tool": "noop"})
    assert event == ToolEvent(tool="noop")
    assert event.kind is EventKind.TOOL_CALL


def test_trace_round_trip_in_memory() -> None:
    trace = Trace(
        label="loop",
        description="two-action loop",
        metadata={"seed": 1},
        events=[
            ToolEvent(tool="list_dir", args={"path": "src"}),
            ToolEvent(tool="read_file", args={"path": "src/main.py"}),
        ],
    )
    assert trace_from_dict(trace_to_dict(trace)) == trace


def test_trace_to_dict_omits_empty_optionals() -> None:
    data = trace_to_dict(Trace(label="healthy"))
    assert data == {"label": "healthy", "events": []}


def test_dump_then_load_round_trip(tmp_path: Path) -> None:
    trace = Trace(
        label="no_progress",
        description="slow burn",
        events=[
            ToolEvent(tool="search", args={"query": "x"}, cost_usd=0.1, ts=1.0),
            ToolEvent(tool="note", args={"text": "hmm"}, ts=2.0),
        ],
    )
    path = tmp_path / "trace.json"
    dump_trace(trace, path)
    assert load_trace(path) == trace


def test_dump_writes_valid_json_with_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    dump_trace(Trace(label="healthy", events=[ToolEvent(tool="ping")]), path)
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw)["label"] == "healthy"


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("loop_trivial.json", "loop"),
        ("no_progress_trivial.json", "no_progress"),
        ("healthy_trivial.json", "healthy"),
    ],
)
def test_committed_examples_load(name: str, label: str) -> None:
    trace = load_trace(_TRACES_DIR / name)
    assert trace.label == label
    assert trace.events
    assert all(isinstance(event, ToolEvent) for event in trace.events)


def test_committed_examples_round_trip(tmp_path: Path) -> None:
    for name in ("loop_trivial.json", "no_progress_trivial.json", "healthy_trivial.json"):
        trace = load_trace(_TRACES_DIR / name)
        out = tmp_path / name
        dump_trace(trace, out)
        assert load_trace(out) == trace
