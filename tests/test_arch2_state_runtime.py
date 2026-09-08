"""ARCH-2: live StateEvidence capture, transport, replay and isolation."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import tool

from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.state_evidence import StateEvidence, validate_state_evidence
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Policy
from pathlayer.state_evidence import (
    FilesystemStateSensor,
    PytestStateSensor,
    StateEvidenceProducer,
    StateOutcome,
    state_evidence_from_event,
)
from pathlayer.store import decode_event, encode_event
from pathlayer.validation.traces import Trace, dump_trace, load_trace


def _tool(name: str, effect: Any) -> Any:
    @tool(name)
    def action() -> str:
        """Perform the fixture action."""
        effect()
        return "ok"

    return action


def test_live_filesystem_change_and_no_effect_are_distinguished(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before", encoding="utf-8")
    events: list[ToolEvent] = []
    producer = StateEvidenceProducer(
        [FilesystemStateSensor(target, resource_id="workspace/target.txt")]
    )
    adapter = LangGraphAdapter(events.append, state_evidence=producer)
    adapter.attach()

    _tool("change_file", lambda: target.write_text("after", encoding="utf-8")).invoke(
        {}, config=adapter.as_config()
    )
    _tool("leave_file", lambda: None).invoke({}, config=adapter.as_config())

    changed = state_evidence_from_event(events[0])
    unchanged = state_evidence_from_event(events[1])
    assert changed is not None and changed.outcome is StateOutcome.CHANGED
    assert unchanged is not None and unchanged.outcome is StateOutcome.UNCHANGED
    assert all(record.tool_event_ref == events[0].run_id for record in changed.records)


def test_live_pytest_transition_uses_same_adapter_path(tmp_path: Path) -> None:
    test_file = tmp_path / "test_target.py"
    test_file.write_text("def test_value():\n    assert 1 == 2\n", encoding="utf-8")
    events: list[ToolEvent] = []
    producer = StateEvidenceProducer(
        [
            PytestStateSensor(
                ("test_target.py::test_value",),
                cwd=tmp_path,
                resource_id="target-suite",
            )
        ]
    )
    adapter = LangGraphAdapter(events.append, state_evidence=producer)
    adapter.attach()

    _tool(
        "fix_test",
        lambda: test_file.write_text("def test_value():\n    assert 1 == 1\n", encoding="utf-8"),
    ).invoke({}, config=adapter.as_config())

    envelope = state_evidence_from_event(events[0])
    assert envelope is not None and envelope.outcome is StateOutcome.CHANGED
    assert envelope.records[1].delta == {
        "newly_passing": ["test_target.py::test_value"],
        "newly_failing": [],
        "still_failing": [],
    }


class _RaisingSensor:
    resource_type = "filesystem"
    resource_id = "secret"
    collector_version = "raising-sensor@1"

    def capture(
        self, *, tool_event_ref: str, phase: str, before: StateEvidence | None = None
    ) -> StateEvidence:
        raise PermissionError("sensitive path")


def test_collection_failure_is_explicit_and_does_not_break_tool() -> None:
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(
        events.append, state_evidence=StateEvidenceProducer([_RaisingSensor()])
    )
    adapter.attach()

    assert _tool("safe_tool", lambda: None).invoke({}, config=adapter.as_config()) == "ok"

    envelope = state_evidence_from_event(events[0])
    assert envelope is not None
    assert envelope.outcome is StateOutcome.COLLECTION_FAILED
    assert [record.status for record in envelope.records] == [
        "collection_failed",
        "collection_failed",
    ]
    assert "sensitive" not in repr(events[0].metadata)


def test_zero_config_event_is_strictly_unchanged() -> None:
    baseline: list[ToolEvent] = []
    adapter = LangGraphAdapter(baseline.append)
    adapter.attach()
    _tool("noop", lambda: None).invoke({}, config=adapter.as_config())

    assert state_evidence_from_event(baseline[0]) is None
    assert set(baseline[0].metadata) == {"parent_run_id"}


def test_evidence_survives_trace_and_durable_event_replay(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("a", encoding="utf-8")
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(
        events.append,
        state_evidence=StateEvidenceProducer([FilesystemStateSensor(target, resource_id="target")]),
    )
    adapter.attach()
    _tool("write", lambda: target.write_text("b", encoding="utf-8")).invoke(
        {}, config=adapter.as_config()
    )

    path = tmp_path / "trace.json"
    dump_trace(Trace(label="healthy", events=events), path)
    trace_event = load_trace(path).events[0]
    stored_event = decode_event(encode_event(events[0]))
    assert state_evidence_from_event(trace_event) == state_evidence_from_event(events[0])
    assert state_evidence_from_event(stored_event) == state_evidence_from_event(events[0])


class _StateConsumer(Detector):
    name = "state_consumer"

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        envelope = state_evidence_from_event(history[-1])
        confidence = float(envelope is not None and envelope.outcome is StateOutcome.CHANGED)
        return Signal(self.name, confidence, "embedded state")


def test_pipeline_consumes_embedded_result_without_collection_io(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("a", encoding="utf-8")
    captured: list[ToolEvent] = []
    adapter = LangGraphAdapter(
        captured.append,
        state_evidence=StateEvidenceProducer([FilesystemStateSensor(target, resource_id="target")]),
    )
    adapter.attach()
    _tool("write", lambda: target.write_text("b", encoding="utf-8")).invoke(
        {}, config=adapter.as_config()
    )
    target.unlink()  # proves the detector reads the event, not the filesystem

    pipeline = Pipeline([_StateConsumer()], Policy())
    pipeline.process(captured[0])
    assert pipeline.last_decision is not None
    assert pipeline.last_decision.signals[0].confidence == 1.0


def test_concurrent_calls_keep_evidence_correlated(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("stable", encoding="utf-8")
    events: list[ToolEvent] = []
    event_lock = threading.Lock()

    def sink(event: ToolEvent) -> None:
        with event_lock:
            events.append(event)

    adapter = LangGraphAdapter(
        sink,
        state_evidence=StateEvidenceProducer([FilesystemStateSensor(target, resource_id="target")]),
    )
    adapter.attach()
    noop = _tool("concurrent_noop", lambda: None)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: noop.invoke({}, config=adapter.as_config()), range(32)))

    assert len(events) == 32
    for event in events:
        envelope = state_evidence_from_event(event)
        assert envelope is not None
        assert envelope.outcome in {StateOutcome.UNCHANGED, StateOutcome.UNAVAILABLE}
        assert {record.tool_event_ref for record in envelope.records} == {event.run_id}


def test_overlapping_tool_boundaries_abstain_instead_of_mixing_effects(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before", encoding="utf-8")
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(
        events.append,
        state_evidence=StateEvidenceProducer([FilesystemStateSensor(target, resource_id="target")]),
    )
    adapter.attach()

    adapter._record_start("run-a", None, "write_a", {})
    target.write_text("a", encoding="utf-8")
    adapter._record_start("run-b", None, "write_b", {})
    target.write_text("b", encoding="utf-8")
    adapter._record_end("run-a", "ok")
    adapter._record_end("run-b", "ok")

    assert len(events) == 2
    assert all(
        state_evidence_from_event(event).outcome is StateOutcome.UNAVAILABLE  # type: ignore[union-attr]
        for event in events
    )


def test_malformed_or_forged_envelope_is_not_interpreted() -> None:
    raw = {
        "schema_version": 1,
        "tool_event_ref": "other",
        "phase": "before",
        "resource_type": "filesystem",
        "resource_id": "target",
        "status": "collection_failed",
        "raw_fingerprint": None,
        "canonical_fingerprint": None,
        "delta": None,
        "collector_version": "runtime-boundary@1",
    }
    assert validate_state_evidence(raw) is not None
    event = ToolEvent(
        "tool",
        run_id="run",
        metadata={
            "pathlayer_state_evidence": {
                "schema_version": 1,
                "outcome": "collection_failed",
                "records": [raw, {**raw, "phase": "after"}],
            }
        },
    )
    assert state_evidence_from_event(event) is None


def test_sensor_count_is_bounded() -> None:
    with pytest.raises(ValueError, match="at most 64"):
        StateEvidenceProducer([_RaisingSensor()] * 65)
