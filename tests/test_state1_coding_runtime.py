"""STATE-1 on a real LangGraph agent: paired scenarios, replay, ablation.

No LLM and no network. A scripted agent edits a real git workspace through
LangChain tools, Pathlayer observes it through the configured preset only, and
production thresholds are untouched throughout.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langchain_core.tools import tool
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph

from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.coding_state import state_evidence_producer_from_config
from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.proof import _PROGRESS_PROOF_KEY
from pathlayer.detectors.proof_producer import producer_from_config
from pathlayer.events import ToolEvent
from pathlayer.pipeline import PipelineDecision
from pathlayer.policies import Action
from pathlayer.policies.langgraph_actuator import LangGraphActuator
from pathlayer.policies.wiring import wire
from pathlayer.state_evidence import (
    STATE_EVIDENCE_METADATA_KEY,
    StateOutcome,
    state_evidence_from_event,
)
from pathlayer.store import decode_event, encode_event
from pathlayer.validation.traces import Trace, dump_trace, load_trace
from tests.conftest import git_workspace

Call = tuple[str, dict[str, Any]]


class RunState(TypedDict):
    step: int


def configuration(
    workspace: Path,
    *,
    loops: bool = False,
    progress_proof: dict[str, Any] | None = None,
    **overrides: Any,
) -> PathlayerConfig:
    """The short configuration a coding integration is expected to write."""
    return PathlayerConfig.from_mapping(
        {
            "detectors": {
                "state_evidence": {
                    "enabled": True,
                    "preset": "coding",
                    "workspace": str(workspace),
                    "tools": {"append_note": {"effect": "write", "paths": ["file_path"]}},
                    **overrides,
                },
                "loop": {"enabled": loops},
                **({"progress_proof": progress_proof} if progress_proof else {}),
            }
        }
    )


def run_agent(
    workspace: Path, script: list[Call], config: PathlayerConfig
) -> tuple[list[ToolEvent], list[PipelineDecision | None], bool]:
    """Drive a scripted agent through a compiled graph until it ends or is stopped."""
    pipeline = build_pipeline(config)
    actuator = LangGraphActuator()
    sink = wire(pipeline, actuator)
    events: list[ToolEvent] = []
    decisions: list[PipelineDecision | None] = []

    def observe(event: ToolEvent) -> None:
        events.append(event)
        sink(event)
        decisions.append(pipeline.last_decision)

    if config.detectors.progress_proof.enabled:
        # ANC-4 and STATE-1 on the same boundary, as an integration would wire them.
        adapter = LangGraphAdapter(
            observe,
            state_evidence=state_evidence_producer_from_config(config),
            progress_proof=producer_from_config(config),
        )
    else:
        adapter = LangGraphAdapter.from_config(observe, config)
    adapter.attach()

    @tool("write_file")
    def write_file(file_path: str, text: str) -> str:
        """Write text to a workspace file."""
        target = workspace / file_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError:
            return "could not write"  # a real tool reports, it does not crash the run
        return "written"

    @tool("append_note")
    def append_note(file_path: str) -> str:
        """Append one more line to a workspace file."""
        target = workspace / file_path
        try:
            existing = target.read_text(encoding="utf-8") if target.is_file() else ""
            target.write_text(f"{existing}note {len(existing.splitlines())}\n", encoding="utf-8")
        except OSError:
            return "could not append"
        return "appended"

    @tool("read_file")
    def read_file(file_path: str) -> str:
        """Read a workspace file."""
        target = workspace / file_path
        return target.read_text(encoding="utf-8") if target.is_file() else ""

    tools = {"write_file": write_file, "append_note": append_note, "read_file": read_file}

    def act(state: RunState) -> RunState:
        name, args = script[state["step"]]
        tools[name].invoke(args, config=adapter.as_config())
        return {"step": state["step"] + 1}

    graph = StateGraph(RunState)
    graph.add_node("act", act)
    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", lambda s: END if s["step"] >= len(script) else "act")

    drained = False
    try:
        graph.compile().invoke(
            {"step": 0},
            config={**adapter.as_config(), "recursion_limit": len(script) + 8},
            control=actuator.control,
        )
    except GraphDrained:
        drained = True
    adapter.detach()
    assert drained == pipeline.stopped == actuator.control.drain_requested
    return events, decisions, drained


def stop_index(decisions: list[PipelineDecision | None]) -> int | None:
    for index, decision in enumerate(decisions, 1):
        if decision is not None and decision.arbitrated_action is Action.STOP:
            return index
    return None


def signal_of(decision: PipelineDecision | None, name: str) -> float:
    if decision is None:
        return 0.0
    return max((s.confidence for s in decision.signals if s.name == name), default=0.0)


def decisions_for(
    config: PathlayerConfig, events: list[ToolEvent]
) -> list[PipelineDecision | None]:
    pipeline = build_pipeline(config)
    replayed: list[PipelineDecision | None] = []
    for event in events:
        pipeline.process(event)
        replayed.append(pipeline.last_decision)
    return replayed


def without_state_evidence(event: ToolEvent) -> ToolEvent:
    return replace(
        event,
        metadata={k: v for k, v in event.metadata.items() if k != STATE_EVIDENCE_METADATA_KEY},
    )


# --------------------------------------------------------------------------- #
# Paired scenarios
# --------------------------------------------------------------------------- #


def test_repeating_a_call_that_keeps_changing_the_file_is_left_alone(tmp_path: Path) -> None:
    """Productive repetition: identical calls, a file that really advances."""
    workspace = git_workspace(tmp_path / "ws")
    config = configuration(workspace, loops=True)
    script: list[Call] = [("append_note", {"file_path": "notes.md"})] * 14

    events, decisions, drained = run_agent(workspace, script, config)

    assert not drained
    assert len(events) == 14
    assert all(signal_of(d, "no_progress") == 0.0 for d in decisions)
    assert (workspace / "notes.md").read_text(encoding="utf-8").count("note") == 14
    # Without the evidence, the identical calls are indistinguishable from a loop.
    baseline = decisions_for(config, [without_state_evidence(event) for event in events])
    assert stop_index(baseline) is not None


def test_repeating_a_call_that_changes_nothing_is_stopped(tmp_path: Path) -> None:
    """Stagnant repetition: the same write, over and over, with no effect."""
    workspace = git_workspace(tmp_path / "ws")
    config = configuration(workspace)
    script: list[Call] = [("write_file", {"file_path": "src/auth.py", "text": "same\n"})] * 14

    events, decisions, drained = run_agent(workspace, script, config)

    assert drained
    stopped_at = stop_index(decisions)
    assert stopped_at is not None and stopped_at <= 12
    stopping = decisions[stopped_at - 1]
    assert stopping is not None
    assert signal_of(stopping, "no_progress") == pytest.approx(1.0)
    # The first write is the one that changed the file; the rest are unchanged.
    outcomes = [state_evidence_from_event(event).outcome for event in events]
    assert outcomes[0] is StateOutcome.CHANGED
    assert set(outcomes[1:]) == {StateOutcome.UNCHANGED}


def test_alternating_between_two_old_states_is_not_progress(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    config = configuration(workspace)
    script: list[Call] = [
        ("write_file", {"file_path": "src/auth.py", "text": f"variant {index % 2}\n"})
        for index in range(16)
    ]

    _, decisions, drained = run_agent(workspace, script, config)

    assert drained
    assert stop_index(decisions) is not None


def test_a_long_exploration_phase_is_never_stopped(tmp_path: Path) -> None:
    """The false stop this preset must not cause, end to end.

    Reading twenty files after one edit is ordinary work. Nothing the preset
    records may turn it into a stall.
    """
    workspace = git_workspace(tmp_path / "ws")
    config = configuration(workspace, loops=True)
    script: list[Call] = [("write_file", {"file_path": "src/auth.py", "text": "edited\n"})]
    script += [("read_file", {"file_path": f"src/module_{index}.py"}) for index in range(20)]

    events, decisions, drained = run_agent(workspace, script, config)

    assert not drained
    assert all(signal_of(d, "no_progress") == 0.0 for d in decisions)
    # Reads claim no state at all: they carry no envelope, exactly like zero-config.
    assert [STATE_EVIDENCE_METADATA_KEY in event.metadata for event in events[1:]] == [False] * 20


def test_state_that_could_not_be_observed_never_reads_as_no_progress(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    # A directory cannot be observed as a regular file: an explicit failure,
    # repeated for a whole run.
    (workspace / "target").mkdir()
    config = configuration(workspace, git=False)
    script: list[Call] = [("append_note", {"file_path": "target"})] * 14

    events, decisions, _ = run_agent(workspace, script, config)

    assert all(
        state_evidence_from_event(event).outcome is StateOutcome.COLLECTION_FAILED
        for event in events
    )
    assert all(signal_of(d, "no_progress") == 0.0 for d in decisions)


def test_overlapping_tool_calls_attribute_nothing_to_either(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    config = configuration(workspace)
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter.from_config(events.append, config)
    adapter.attach()

    @tool("read_file")
    def inner(file_path: str) -> str:
        """Change the workspace from inside another tool's boundary."""
        (workspace / "src" / "auth.py").write_text("changed by the inner call\n", encoding="utf-8")
        return "ok"

    @tool("write_file")
    def outer(file_path: str, text: str) -> str:
        """Keep this boundary open while another tool runs."""
        inner.invoke({"file_path": "src/auth.py"}, config=adapter.as_config())
        return "ok"

    outer.invoke({"file_path": "src/auth.py", "text": "x"}, config=adapter.as_config())

    # The inner call claims nothing, but it can still change the outer call's
    # file: the outer observation is withdrawn rather than crediting the change.
    envelope = state_evidence_from_event(events[-1])
    assert envelope is not None
    assert envelope.outcome is StateOutcome.UNAVAILABLE


# --------------------------------------------------------------------------- #
# Replay and durability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scenario", ["productive", "stagnant"])
def test_live_and_replay_decide_identically_without_the_workspace(
    tmp_path: Path, scenario: str
) -> None:
    workspace = git_workspace(tmp_path / "ws")
    config = configuration(workspace, loops=True)
    script: list[Call] = (
        [("append_note", {"file_path": "notes.md"})] * 12
        if scenario == "productive"
        else [("write_file", {"file_path": "src/auth.py", "text": "same\n"})] * 12
    )
    events, decisions, _ = run_agent(workspace, script, config)

    trace = tmp_path / "recorded.json"
    dump_trace(
        Trace(label="healthy" if scenario == "productive" else "no_progress", events=events), trace
    )
    restored = load_trace(trace).events
    round_tripped = [decode_event(encode_event(event)) for event in events]

    # Destroy the observed workspace: replay may depend only on the recording.
    for path in sorted(workspace.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    workspace.rmdir()

    for replayed_events in (restored, round_tripped):
        assert decisions_for(config, replayed_events) == decisions


def test_an_observed_effect_does_not_silence_a_stalled_contract(tmp_path: Path) -> None:
    """ANC-4 and STATE-1 on the same run: novelty must not outvote the objective.

    A busy agent that reaches a brand-new state on every call keeps the effect
    anchor perfectly quiet. If that quiet also covered the ANC-4 contract, the
    preset would have made a stalled run *harder* to catch than before it.
    """
    workspace = git_workspace(tmp_path / "ws")
    script: list[Call] = [
        ("write_file", {"file_path": "src/auth.py", "text": f"attempt {index}\n"})
        for index in range(14)
    ]

    # Effect evidence alone: every attempt is a new state, so nothing fires.
    _, _, drained = run_agent(workspace, script, configuration(workspace))
    assert not drained

    # The same run, with a contract whose checkpoint is never reached.
    manifest = tmp_path / "proof_spec.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "fix-login",
                "proof_id": "proof-1",
                "scope_id": "scope-1",
                "kind": "checkpoint_set",
                "required": ["login_test_passes"],
            }
        ),
        encoding="utf-8",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        'import json\nprint(json.dumps({"checkpoints": {"login_test_passes": False}}))\n',
        encoding="utf-8",
    )
    with_contract = configuration(
        git_workspace(tmp_path / "ws2"),
        progress_proof={
            "enabled": True,
            "manifest_path": str(manifest),
            "probe_command": [sys.executable, str(probe)],
            "timeout_seconds": 30.0,
        },
    )
    events, decisions, drained = run_agent(tmp_path / "ws2", script, with_contract)

    assert all(_PROGRESS_PROOF_KEY in event.metadata for event in events)
    assert all(state_evidence_from_event(event) is not None for event in events)
    assert drained and stop_index(decisions) is not None


def test_a_configured_build_command_is_observed_around_a_write(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    marker = workspace / "src" / "ready.txt"
    config = configuration(
        workspace,
        build={
            "id": "validate",
            "command": [
                sys.executable,
                "-c",
                f"import os,sys; sys.exit(0 if os.path.exists({str(marker)!r}) else 1)",
            ],
        },
    )
    events, _, _ = run_agent(
        workspace, [("write_file", {"file_path": "src/ready.txt", "text": "ok"})], config
    )

    envelope = state_evidence_from_event(events[0])
    assert envelope is not None
    command_records = [r for r in envelope.records if r.resource_type == "command"]
    assert [record.canonical_fingerprint for record in command_records] == ["failure", "success"]
    assert command_records[1].delta == {"changed": True, "before": "failure", "after": "success"}


def test_zero_config_capture_is_strictly_unchanged(tmp_path: Path) -> None:
    events: list[ToolEvent] = []
    adapter = LangGraphAdapter.from_config(events.append, PathlayerConfig())
    adapter.attach()

    @tool("write_file")
    def write_file(file_path: str, text: str) -> str:
        """Write a file with no preset configured."""
        return "written"

    write_file.invoke({"file_path": "a.py", "text": "x"}, config=adapter.as_config())

    assert set(events[0].metadata) == {"parent_run_id"}
