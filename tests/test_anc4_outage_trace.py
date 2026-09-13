"""ANC-4: labeled evidence that a probe outage mid-run is a gap, not a STOP.

``traces/anc4/probe_outage_mid_run.json`` was recorded live from the
``unobservable_mid_run`` scenario of ``examples/demo_coding_contract.py``: a
productive agent whose test module cannot be collected for ten consecutive
tool calls. It carries the generated probe's real proofs and
``progress_proof_unobserved`` markers, so it replays without a workspace. It
lives outside ``traces/healthy/`` because the ANC-3 benchmark seals that
folder's exact contents; the golden-set guards still discover it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.proof import _PROGRESS_PROOF_UNOBSERVED_KEY, ProofProgressAnchor
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.validation.traces import load_trace

_TRACE = Path(__file__).resolve().parents[1] / "traces" / "anc4" / "probe_outage_mid_run.json"


def _first_stop(events: list[ToolEvent]) -> int | None:
    pipeline = Pipeline([NoProgressDetector([ProofProgressAnchor()])], Policy())
    for index, event in enumerate(events, 1):
        pipeline.process(event)
        decision = pipeline.last_decision
        if decision is not None and decision.arbitrated_action is Action.STOP:
            return index
    return None


def test_outage_trace_is_healthy_with_the_proof_anchor() -> None:
    trace = load_trace(_TRACE)
    assert trace.label == "healthy"
    markers = [e for e in trace.events if _PROGRESS_PROOF_UNOBSERVED_KEY in e.metadata]
    assert len(trace.events) == 22
    assert len(markers) == 10
    assert all(len(e.metadata[_PROGRESS_PROOF_UNOBSERVED_KEY]["unobserved"]) == 4 for e in markers)
    assert _first_stop(trace.events) is None


def test_same_trace_without_markers_reproduces_the_freeze() -> None:
    """The counterexample the fix exists for: strip the markers (what a
    pre-ANC-4 producer left on those ticks) and the frozen level ends in a STOP
    while the agent is mid-refactor."""
    trace = load_trace(_TRACE)
    stripped = [
        replace(
            event,
            metadata={
                key: value
                for key, value in event.metadata.items()
                if key != _PROGRESS_PROOF_UNOBSERVED_KEY
            },
        )
        for event in trace.events
    ]
    assert _first_stop(stripped) == 17


def test_outage_trace_is_silent_under_the_default_pipeline() -> None:
    pipeline = build_pipeline(PathlayerConfig())
    for event in load_trace(_TRACE).events:
        pipeline.process(event)
    assert not pipeline.stopped


def test_outage_trace_is_silent_under_the_production_pipeline_with_the_contract(
    tmp_path: Path,
) -> None:
    """The anchor set ``build_pipeline`` wires with a contract: the coding
    anchors next to the proof anchor, and the probe gaps holding no_progress."""
    trace = load_trace(_TRACE)
    manifest = tmp_path / "proof_spec.json"
    manifest.write_text(json.dumps(trace.events[0].metadata["proof_spec"]), encoding="utf-8")
    section = {"enabled": True, "manifest_path": str(manifest), "probe_command": ["probe"]}
    pipeline = build_pipeline(
        PathlayerConfig.from_mapping({"detectors": {"progress_proof": section}})
    )
    confidences = []
    for event in trace.events:
        pipeline.process(event)
        decision = pipeline.last_decision
        assert decision is not None
        confidences.extend(s.confidence for s in decision.signals if s.name == "no_progress")
    assert not pipeline.stopped
    assert len(confidences) == 22
    assert max(confidences) < 0.9
