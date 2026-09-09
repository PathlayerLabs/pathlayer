"""VAL-5: paired real runtime executions, fixed production thresholds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.val5.run import pipeline, run_case
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import is_advancing
from pathlayer.detectors.state_progress import StateEvidenceProgressAnchor
from pathlayer.events import EventKind, ToolEvent
from pathlayer.policies.actions import Action
from pathlayer.state_evidence import (
    STATE_EVIDENCE_METADATA_KEY,
    StateOutcome,
    state_evidence_from_event,
)
from pathlayer.store import decode_event, encode_event
from pathlayer.validation import Trace, dump_trace, load_trace


@pytest.mark.parametrize(
    "mode", ["productive", "stagnant", "unavailable", "collection_failed", "cycle"]
)
@pytest.mark.parametrize("loops", [False, True])
def test_paired_real_graph_and_replay(tmp_path: Path, mode: str, loops: bool) -> None:
    events, decisions, drained = run_case(tmp_path / "live", mode, loops=loops)
    assert events
    assert {e.tool for e in events} == {"advance_output"}
    expected_stop = mode in {"stagnant", "cycle"} or (
        mode in {"unavailable", "collection_failed"} and loops
    )
    assert drained is expected_stop
    if mode == "productive":
        assert len(events) == 16
        assert all(d.arbitrated_action is Action.NONE for d in decisions)
        assert all(
            s.confidence == 0 for d in decisions for s in d.signals if s.name == "no_progress"
        )
    if mode in {"unavailable", "collection_failed"}:
        expected = StateOutcome(mode)
        assert all(state_evidence_from_event(e).outcome is expected for e in events)
        assert all(
            s.confidence == 0 for d in decisions for s in d.signals if s.name == "no_progress"
        )
    path = tmp_path / "recorded.json"
    dump_trace(
        Trace(label="healthy" if mode == "productive" else "no_progress", events=events), path
    )
    # Destroy live resources: replay must depend solely on the recorded evidence.
    (tmp_path / "live" / "output.txt").unlink()
    for restored in [load_trace(path).events, [decode_event(encode_event(e)) for e in events]]:
        replay = pipeline(loops=loops)
        replay_decisions = []
        for event in restored:
            replay.process(event)
            replay_decisions.append(replay.last_decision)
        assert replay_decisions == decisions
    # Ablation of the exact same run, no recollection or threshold changes.
    baseline = pipeline(loops=loops)
    for event in events:
        baseline.process(
            replace(
                event,
                metadata={
                    k: v for k, v in event.metadata.items() if k != STATE_EVIDENCE_METADATA_KEY
                },
            )
        )
    if mode == "productive" and loops:
        assert baseline.stopped  # false stop prevented by fresh state evidence
    if mode == "stagnant" and not loops:
        assert not baseline.stopped  # previously invisible stall now detected


def test_gaps_cycles_capacity_and_prefix_equivalence(tmp_path: Path) -> None:
    events, _, _ = run_case(tmp_path / "live", "productive")
    gap = replace(events[3], metadata={})
    malformed = replace(events[3], metadata={STATE_EVIDENCE_METADATA_KEY: {"bad": True}})
    non_tool = replace(events[3], kind=EventKind.LLM_CALL)
    history = [*events[:3], gap, malformed, non_tool, *events[3:]]
    for cap in [2, 4096]:
        anchor = StateEvidenceProgressAnchor("output", max_states=cap)
        state = anchor.create_state()
        detector = NoProgressDetector([anchor])
        ds = detector.create_state()
        for i, event in enumerate(history):
            state, snap = anchor.step(state, event)
            prefix = history[: i + 1]
            assert snap.level == anchor.measure(prefix)
            assert snap.diagnostics == anchor.diagnose(prefix)
            ds, signal = detector.step(ds, event)
            assert signal == detector.evaluate(prefix)
        if cap == 2:
            assert state.exhausted and state.snapshot.level is None
    anchor = StateEvidenceProgressAnchor("output")
    assert anchor.measure([gap]) is None
    assert not is_advancing([anchor], [events[0], gap])
    assert StateEvidenceProgressAnchor("other").measure(events) is None
    assert not anchor.is_satisfied(events)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resource_id": ""},
        {"resource_id": "x", "max_states": 1},
        {"resource_id": "x", "max_states": True},
    ],
)
def test_bad_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        StateEvidenceProgressAnchor(**kwargs)


def test_real_overlapping_boundaries_abstain(tmp_path: Path) -> None:
    from langchain_core.tools import tool

    from pathlayer.adapters.langgraph import LangGraphAdapter
    from pathlayer.state_evidence import FilesystemStateSensor, StateEvidenceProducer

    output = tmp_path / "output.txt"
    output.write_text("0")
    events = []
    adapter = LangGraphAdapter(
        events.append,
        state_evidence=StateEvidenceProducer([FilesystemStateSensor(output, resource_id="output")]),
    )
    adapter.attach()

    @tool
    def inner() -> str:
        """Change the shared artifact inside an overlapping tool boundary."""
        output.write_text("1")
        return "ok"

    @tool
    def outer() -> str:
        """Keep the outer tool boundary open while the inner tool executes."""
        inner.invoke({}, config=adapter.as_config())
        return "ok"

    outer.invoke({}, config=adapter.as_config())
    assert len(events) == 2
    assert all(state_evidence_from_event(e).outcome is StateOutcome.UNAVAILABLE for e in events)
    detector = NoProgressDetector([StateEvidenceProgressAnchor("output")])
    assert detector.evaluate(events).confidence == 0


def test_multiple_sensors_and_bad_state_abstain(tmp_path: Path) -> None:
    from pathlayer.state_evidence import FilesystemStateSensor, StateEvidenceProducer

    path = tmp_path / "file"
    path.write_text("a")
    producer = StateEvidenceProducer(
        [
            FilesystemStateSensor(path, resource_id="output"),
            FilesystemStateSensor(path, resource_id="other"),
        ]
    )
    before = producer.capture_before("event")
    path.write_text("b")
    event = producer.attach_after(ToolEvent(tool="write", run_id="event"), before)
    anchor = StateEvidenceProgressAnchor("output")
    assert anchor.measure([event]) is None
    with pytest.raises(TypeError):
        anchor.step(None, event)


def test_fresh_state_does_not_suppress_budget(tmp_path: Path) -> None:
    from pathlayer.detectors.budget import BudgetDetector
    from pathlayer.pipeline import Pipeline
    from pathlayer.policies.base import Policy, PolicyConfig, SuppressionRule

    events, _, _ = run_case(tmp_path / "live", "productive")
    pipe = Pipeline(
        [BudgetDetector(max_cost_usd=1.0)],
        Policy(PolicyConfig(suppressions={"loop": SuppressionRule("progress")})),
        progress_anchors=[StateEvidenceProgressAnchor("output")],
    )
    for event in events[:2]:
        pipe.process(replace(event, cost_usd=0.5))
    assert pipe.stopped
    assert pipe.last_decision.evidence["progress"] == 1.0


@pytest.mark.parametrize(
    "fingerprint", [[], {}, ["a"], {"x": 1}, 3, True, None, "invalid", "sha256:bad"]
)
@pytest.mark.parametrize("phase", [0, 1])
def test_structurally_valid_but_bad_filesystem_fingerprint_is_a_gap(fingerprint, phase) -> None:
    import copy

    fixture = (
        Path(__file__).resolve().parents[1] / "benchmarks/val5/results/productive-combined.json"
    )
    event = load_trace(fixture).events[0]
    metadata = copy.deepcopy(event.metadata)
    raw = metadata[STATE_EVIDENCE_METADATA_KEY]
    raw["records"][phase]["canonical_fingerprint"] = fingerprint
    event = replace(event, metadata=metadata)
    anchor = StateEvidenceProgressAnchor("output")
    assert anchor.measure([event]) is None
    assert NoProgressDetector([anchor]).evaluate([event]).confidence == 0


def test_regenerated_results_match_archives(tmp_path: Path, monkeypatch) -> None:
    import sys

    from benchmarks.val5.run import main

    reference = Path(__file__).resolve().parents[1] / "benchmarks/val5/results"
    monkeypatch.setattr(
        sys, "argv", ["val5", "--output", str(tmp_path), "--check-results", str(reference)]
    )
    main()


@pytest.mark.parametrize(
    "mutation",
    ["reference", "parent", "fingerprint", "outcome", "time", "verdict", "missing", "extra"],
)
def test_result_comparison_detects_changes(tmp_path: Path, mutation: str) -> None:
    import json
    import shutil

    from benchmarks.val5.compare import compare_results

    reference = Path(__file__).resolve().parents[1] / "benchmarks/val5/results"
    shutil.copytree(reference, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "productive-combined.json"
    payload = json.loads(path.read_text())
    event = payload["events"][0]
    evidence = event["metadata"][STATE_EVIDENCE_METADATA_KEY]
    if mutation == "reference":
        evidence["records"][1]["tool_event_ref"] = "wrong-event"
    elif mutation == "parent":
        event["metadata"]["parent_run_id"] = event["run_id"]
    elif mutation == "fingerprint":
        evidence["records"][1]["canonical_fingerprint"] = "changed"
    elif mutation == "outcome":
        evidence["outcome"] = "unavailable"
    elif mutation == "time":
        event["ts"] = 999
    elif mutation == "verdict":
        (tmp_path / "summary.json").write_text('{"verdict": "wrong"}')
    elif mutation == "missing":
        (tmp_path / "cycle-combined.json").unlink()
    elif mutation == "extra":
        (tmp_path / "unexpected.json").write_text("{}")
    path.write_text(json.dumps(payload))
    assert compare_results(reference, tmp_path)


@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize(
    "field,value",
    [("resource_id", "other"), ("resource_type", "tests"), ("collector_version", "other@1")],
)
def test_pair_scope_mismatch_abstains(phase, field, value) -> None:
    import copy

    fixture = (
        Path(__file__).resolve().parents[1] / "benchmarks/val5/results/productive-combined.json"
    )
    event = load_trace(fixture).events[0]
    metadata = copy.deepcopy(event.metadata)
    metadata[STATE_EVIDENCE_METADATA_KEY]["records"][phase][field] = value
    assert (
        StateEvidenceProgressAnchor("output").measure([replace(event, metadata=metadata)]) is None
    )


def test_missing_no_progress_signal_returns_no_go(tmp_path: Path, monkeypatch) -> None:
    import json
    import sys

    from benchmarks.val5 import run
    from pathlayer.pipeline import Pipeline

    # Remove the detector from the actual live and replay pipeline. The harness
    # should report NO-GO, not crash in max() or fabricate a successful check.
    monkeypatch.setattr(run, "pipeline", lambda **kwargs: Pipeline([]))
    monkeypatch.setattr(sys, "argv", ["val5", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="1"):
        run.main()
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["verdict"] == "NO-GO"
    assert all(row["no_progress_max"] == 0 for row in report["results"])
