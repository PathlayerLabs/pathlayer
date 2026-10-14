"""ANC-4: LangGraph runs the ANC-2 progress-proof producer at every tool boundary.

Before ANC-4 only the OpenClaw recorder ran the probe live, so a LangGraph
coding agent could enable ``detectors.progress_proof`` and never receive a
single proof. The adapter now probes after each ``TOOL_CALL`` -- under the same
lock that orders emission and ``ts`` -- and never lets a probe problem break
the observed run.
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph

from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.coding_contracts import (
    CodingContract,
    FileChanges,
    FileExists,
    write_coding_contract,
)
from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.proof import (
    _PROGRESS_PROOF_KEY,
    _PROOF_SPEC_KEY,
    ProofProgressAnchor,
    ProofSpec,
)
from pathlayer.detectors.proof_producer import (
    ProgressProofProducer,
    build_producer,
    producer_from_config,
)
from pathlayer.events import PROGRESS_SUMMARY_TOOL, EventKind, ToolEvent
from pathlayer.pipeline import Pipeline, PipelineDecision
from pathlayer.policies import Action, Policy
from pathlayer.policies.langgraph_actuator import LangGraphActuator
from pathlayer.policies.wiring import wire
from pathlayer.state_evidence import (
    FilesystemStateSensor,
    StateEvidenceProducer,
    StateOutcome,
    state_evidence_from_event,
)
from pathlayer.store import decode_event, encode_event
from pathlayer.validation.traces import Trace, dump_trace, load_trace


def _producer(
    tmp_path: Path, body: str, required: tuple[str, ...] = ("done",)
) -> ProgressProofProducer:
    manifest = tmp_path / "proof_spec.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task-1",
                "proof_id": "proof-1",
                "scope_id": "scope-1",
                "kind": "checkpoint_set",
                "required": list(required),
            }
        ),
        encoding="utf-8",
    )
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return build_producer(str(manifest), (sys.executable, str(script)), 10.0)


def _exists_probe(tmp_path: Path, target: Path) -> ProgressProofProducer:
    """Reports ``done`` iff ``target`` exists when the probe runs."""
    return _producer(
        tmp_path,
        f"""
        import json, os
        print(json.dumps({{"checkpoints": {{"done": os.path.exists({str(target)!r})}}}}))
        """,
    )


def _counting_probe(tmp_path: Path, counter: Path) -> ProgressProofProducer:
    """Counts its invocations; ``a`` from the 2nd call, ``b`` from the 4th, ``c`` never."""
    return _producer(
        tmp_path,
        f"""
        import json
        path = {str(counter)!r}
        try:
            count = int(open(path).read())
        except FileNotFoundError:
            count = 0
        count += 1
        open(path, "w").write(str(count))
        print(json.dumps({{"checkpoints": {{"a": count >= 2, "b": count >= 4, "c": False}}}}))
        """,
        required=("a", "b", "c"),
    )


def _tool(name: str, effect: Any = None) -> Any:
    @tool(name)
    def action() -> str:
        """Perform the fixture action."""
        if effect is not None:
            effect()
        return "ok"

    return action


def _proof(event: ToolEvent) -> dict[str, Any] | None:
    return event.metadata.get(_PROGRESS_PROOF_KEY)


# --------------------------------------------------------------------------- #
# Capture contract
# --------------------------------------------------------------------------- #


def test_zero_config_event_is_strictly_unchanged() -> None:
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append)
    adapter.attach()

    _tool("noop").invoke({}, config=adapter.as_config())

    assert set(events[0].metadata) == {"parent_run_id"}


def test_each_tool_call_carries_a_proof_observed_after_its_effect(tmp_path: Path) -> None:
    target = tmp_path / "deliverable.txt"
    producer = _exists_probe(tmp_path, target)
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append, progress_proof=producer)
    adapter.attach()

    _tool("read_docs").invoke({}, config=adapter.as_config())
    _tool("write_deliverable", lambda: target.write_text("x", encoding="utf-8")).invoke(
        {}, config=adapter.as_config()
    )

    assert [event.metadata[_PROOF_SPEC_KEY]["required"] for event in events] == [["done"], ["done"]]
    proofs = [_proof(event) for event in events]
    assert [proof["checkpoints"] for proof in proofs] == [{"done": False}, {"done": True}]  # type: ignore[index]
    assert [proof["sequence"] for proof in proofs] == [1, 2]  # type: ignore[index]
    assert all(proof["spec_hash"] == producer.spec_hash for proof in proofs)  # type: ignore[index]


def test_summaries_and_llm_calls_are_never_probed(tmp_path: Path) -> None:
    counter = tmp_path / "calls"
    producer = _counting_probe(tmp_path, counter)
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append, progress_proof=producer)
    adapter.attach()

    _tool(PROGRESS_SUMMARY_TOOL).invoke({}, config=adapter.as_config())
    run_id = uuid4()
    adapter._handler.on_chat_model_start({"kwargs": {"model": "gpt-4o"}}, [[]], run_id=run_id)
    adapter._handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]), run_id=run_id
    )
    _tool("edit").invoke({}, config=adapter.as_config())

    kinds = [event.kind for event in events]
    assert kinds == [EventKind.PROGRESS_SUMMARY, EventKind.LLM_CALL, EventKind.TOOL_CALL]
    assert [_proof(event) is not None for event in events] == [False, False, True]
    assert counter.read_text() == "1"


def test_a_failed_tool_call_is_still_probed(tmp_path: Path) -> None:
    producer = _exists_probe(tmp_path, tmp_path / "never")
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append, progress_proof=producer)
    adapter.attach()

    def explode() -> None:
        raise ValueError("tool failed")

    with pytest.raises(ValueError):
        _tool("fragile", explode).invoke({}, config=adapter.as_config())

    assert events[0].metadata["error"] == "tool failed"
    assert _proof(events[0]) is not None


def test_probe_failure_emits_the_event_with_an_unobserved_marker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    producer = _producer(tmp_path, "import sys; sys.exit(1)")
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append, progress_proof=producer)
    adapter.attach()

    with caplog.at_level(logging.WARNING):
        assert _tool("edit").invoke({}, config=adapter.as_config()) == "ok"

    assert len(events) == 1
    assert set(events[0].metadata) == {
        "parent_run_id",
        _PROOF_SPEC_KEY,
        "progress_proof_unobserved",
    }
    assert events[0].metadata["progress_proof_unobserved"] == {
        "spec_hash": producer.spec_hash,
        "unobserved": ["done"],
        "reason": "exited with status 1",
    }
    assert events[0].result == "ok"
    assert any("no proof this tick" in record.getMessage() for record in caplog.records)


def test_probe_outage_mid_run_never_stops_a_productive_graph(tmp_path: Path) -> None:
    """The ANC-4 gap on the real adapter path: the probe observes two ticks,
    then can no longer run for twelve tool calls (more than the ~10 steps of
    patience). Without the marker those ticks froze the last level and the
    default policy would STOP a productive agent."""
    counter = tmp_path / "count"
    producer = _producer(
        tmp_path,
        f"""
        import json, sys
        path = {str(counter)!r}
        try:
            count = int(open(path).read())
        except FileNotFoundError:
            count = 0
        count += 1
        open(path, "w").write(str(count))
        if count > 2:
            sys.exit(4)
        print(json.dumps({{"checkpoints": {{"a": count >= 2, "b": False}}}}))
        """,
        required=("a", "b"),
    )
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append, progress_proof=producer)
    adapter.attach()
    for _ in range(14):
        _tool("edit").invoke({}, config=adapter.as_config())

    assert sum("progress_proof_unobserved" in event.metadata for event in events) == 12
    decisions = _decisions(events)
    assert all(
        decision is None or decision.arbitrated_action is not Action.STOP for decision in decisions
    )
    assert (
        max(
            signal.confidence
            for decision in decisions
            if decision is not None
            for signal in decision.signals
        )
        == 0.0
    )
    # Round trip: the marker survives the trace codec and the store codec.
    trace_path = tmp_path / "trace.json"
    dump_trace(Trace(label="unknown", events=events), trace_path)
    assert _decisions(load_trace(trace_path).events) == decisions
    assert _decisions([decode_event(encode_event(event)) for event in events]) == decisions


def test_a_raising_producer_never_breaks_the_observed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    producer = _exists_probe(tmp_path, tmp_path / "never")

    def boom(self: ProgressProofProducer, event: ToolEvent) -> None:
        raise RuntimeError("producer bug")

    monkeypatch.setattr(ProgressProofProducer, "attach", boom)
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(events.append, progress_proof=producer)
    adapter.attach()

    with caplog.at_level(logging.WARNING, logger="pathlayer.adapters.langgraph"):
        assert _tool("edit").invoke({}, config=adapter.as_config()) == "ok"
        assert _tool("edit_again").invoke({}, config=adapter.as_config()) == "ok"

    assert [event.tool for event in events] == ["edit", "edit_again"]
    assert all("progress_proof_unobserved" in event.metadata for event in events)
    assert all("progress_proof" not in event.metadata for event in events)
    assert any("progress_proof" in record.getMessage() for record in caplog.records)


def test_concurrent_calls_get_strictly_increasing_sequences_in_emission_order(
    tmp_path: Path,
) -> None:
    producer = _counting_probe(tmp_path, tmp_path / "calls")
    events: list[ToolEvent] = []
    lock = threading.Lock()

    def sink(event: ToolEvent) -> None:
        with lock:
            events.append(event)

    adapter = LangGraphAdapter(sink, progress_proof=producer)
    adapter.attach()
    noop = _tool("concurrent_noop")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: noop.invoke({}, config=adapter.as_config()), range(32)))

    assert len(events) == 32
    assert [event.ts for event in events] == [float(index) for index in range(32)]
    sequences = [_proof(event)["sequence"] for event in events]  # type: ignore[index]
    assert sequences == sorted(sequences) and len(set(sequences)) == 32
    anchor = ProofProgressAnchor()
    assert anchor.diagnose(events) == ()
    assert anchor.measure(events) == 2.0


def test_state_evidence_and_progress_proof_coexist(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before", encoding="utf-8")
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter(
        events.append,
        state_evidence=StateEvidenceProducer([FilesystemStateSensor(target, resource_id="t")]),
        progress_proof=_exists_probe(tmp_path, target),
    )
    adapter.attach()

    _tool("write", lambda: target.write_text("after", encoding="utf-8")).invoke(
        {}, config=adapter.as_config()
    )

    envelope = state_evidence_from_event(events[0])
    assert envelope is not None and envelope.outcome is StateOutcome.CHANGED
    assert _proof(events[0])["checkpoints"] == {"done": True}  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Live and replay decide identically on identical proofs
# --------------------------------------------------------------------------- #


def _decisions(events: list[ToolEvent]) -> list[PipelineDecision | None]:
    pipeline = Pipeline([NoProgressDetector([ProofProgressAnchor()])], Policy())
    decisions = []
    for event in events:
        pipeline.process(event)
        decisions.append(pipeline.last_decision)
    return decisions


def test_proofs_survive_trace_and_store_and_replay_identically(tmp_path: Path) -> None:
    producer = _counting_probe(tmp_path, tmp_path / "calls")
    live = Pipeline([NoProgressDetector([ProofProgressAnchor()])], Policy())
    events: list[ToolEvent] = []
    live_decisions: list[PipelineDecision | None] = []

    def observe(event: ToolEvent) -> None:
        events.append(event)
        live.process(event)
        live_decisions.append(live.last_decision)

    adapter = LangGraphAdapter(observe, progress_proof=producer)
    adapter.attach()
    for index in range(14):
        _tool(f"step_{index}").invoke({}, config=adapter.as_config())
    (tmp_path / "probe.py").unlink()  # replay must never need the probe again

    path = tmp_path / "trace.json"
    dump_trace(Trace(label="unknown", events=events), path)
    from_trace = load_trace(path).events
    from_store = [decode_event(encode_event(event)) for event in events]

    assert [_proof(event) for event in from_trace] == [_proof(event) for event in events]
    assert [_proof(event) for event in from_store] == [_proof(event) for event in events]
    assert _decisions(from_trace) == live_decisions
    assert _decisions(from_store) == live_decisions
    assert live.stopped  # the level stops rising after the 4th call, then stagnates


# --------------------------------------------------------------------------- #
# A real compiled graph, the user-facing configuration path, real enforcement
# --------------------------------------------------------------------------- #


class _RunState(TypedDict):
    steps: int


def _run_graph(
    workspace: Path, bundle: Path, mode: str, steps: int
) -> tuple[list[ToolEvent], list[PipelineDecision | None], bool, Pipeline]:
    installed = write_coding_contract(
        CodingContract(
            task="write-steps",
            objective="Write the twelve step files.",
            workspace=workspace,
            progress=(
                FileChanges("scratch.txt"),
                *(FileExists(f"step_{index:02d}.txt") for index in range(1, 13)),
            ),
            timeout_seconds=10,
        ),
        output_dir=bundle,
    )
    config = PathlayerConfig.from_mapping(
        {"detectors": {"progress_proof": installed.progress_proof_config()}}
    )
    pipeline = build_pipeline(config)
    actuator = LangGraphActuator()
    sink = wire(pipeline, actuator)
    events: list[ToolEvent] = []
    decisions: list[PipelineDecision | None] = []

    def observe(event: ToolEvent) -> None:
        events.append(event)
        sink(event)
        decisions.append(pipeline.last_decision)

    adapter = LangGraphAdapter(observe, progress_proof=producer_from_config(config))
    adapter.attach()

    @tool
    def work(step: int) -> str:
        """Perform the next step of the task."""
        if mode == "productive":
            (workspace / f"step_{step:02d}.txt").write_text("done", encoding="utf-8")
        else:  # busy but stuck: a different edit every time, no deliverable
            (workspace / "scratch.txt").write_text(f"attempt {step}", encoding="utf-8")
        return "ok"

    def node(state: _RunState) -> _RunState:
        work.invoke({"step": state["steps"] + 1})
        return {"steps": state["steps"] + 1}

    graph = StateGraph(_RunState)
    graph.add_node("act", node)
    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", lambda state: END if state["steps"] >= steps else "act")
    drained = False
    try:
        graph.compile().invoke(
            {"steps": 0},
            config={**adapter.as_config(), "recursion_limit": 4 * steps},
            control=actuator.control,
        )
    except GraphDrained:
        drained = True
    assert drained == pipeline.stopped == actuator.control.drain_requested
    return events, decisions, drained, pipeline


def test_real_graph_stagnation_is_stopped_on_no_progress(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    events, decisions, drained, _ = _run_graph(workspace, tmp_path / "bundle", "stagnant", 16)

    assert drained
    assert len(events) == 11  # first proof sets the baseline, ten flat steps reach STOP
    last = decisions[-1]
    assert last is not None and last.arbitrated_action is Action.STOP
    signals = {signal.name: signal.confidence for signal in last.signals}
    assert signals["no_progress"] == pytest.approx(1.0)
    assert signals["loop"] == 0.0
    assert all(_proof(event) is not None for event in events)


def test_real_graph_productive_repetition_is_left_alone(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    events, decisions, drained, pipeline = _run_graph(
        workspace, tmp_path / "bundle", "productive", 12
    )

    assert not drained
    assert len(events) == 12
    assert all(
        decision is not None and decision.arbitrated_action is Action.NONE for decision in decisions
    )
    assert (
        max(
            signal.confidence
            for decision in decisions
            if decision is not None
            for signal in decision.signals
            if signal.name == "no_progress"
        )
        == 0.0
    )
    assert not pipeline.stopped


def test_manifest_spec_is_the_one_attached_to_events(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    events, _, _, _ = _run_graph(workspace, tmp_path / "bundle", "productive", 1)

    spec = json.loads((tmp_path / "bundle" / "proof_spec.json").read_text(encoding="utf-8"))
    assert events[0].metadata[_PROOF_SPEC_KEY] == spec
    assert ProofSpec(**{**spec, "required": tuple(spec["required"])}).task_id == "write-steps"
