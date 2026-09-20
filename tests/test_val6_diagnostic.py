"""Diagnostic projection must follow accepted runtime evidence."""

import json
from dataclasses import asdict

from pathlayer.detectors.proof import ProofSpec, compute_spec_hash
from pathlayer.events import ToolEvent
from pathlayer.policies import Action
from pathlayer.postmortem import TickRecord, build_coding_diagnostic


def event(sequence=1, value=False, task="one", claimed_hash=None):
    spec = ProofSpec(1, task, "proof", "workspace", "checkpoint_set", ("test",))
    return ToolEvent(
        tool="edit",
        metadata={
            "proof_spec": asdict(spec),
            "progress_proof": {
                "spec_hash": claimed_hash or compute_spec_hash(spec),
                "sequence": sequence,
                "checkpoints": {"test": value},
            },
        },
    )


def test_rejected_proofs_never_report_success():
    for invalid in (event(0, True), event(1, True), event(2, True, claimed_hash="wrong")):
        diagnostic = build_coding_diagnostic([event(), invalid], [])
        assert diagnostic.contract_state != "reached"
        assert diagnostic.last_progress is None


def test_epoch_switch_discards_success():
    diagnostic = build_coding_diagnostic([event(value=True), event(task="two")], [])
    assert diagnostic.contract_state == "unobserved"
    assert diagnostic.last_progress is None


def test_dry_run_is_not_an_intervention_and_full_reason_survives():
    ticks = [
        TickRecord(index=0, ts=0, action=Action.STOP, reason="loop: simulated", dry_run=True),
        TickRecord(index=1, ts=1, action=Action.WARN, reason="budget: limit exceeded"),
    ]
    diagnostic = build_coding_diagnostic([event(), event(2)], ticks)
    assert diagnostic.intervention_event == 1
    assert diagnostic.intervention_action == "warn"
    assert diagnostic.intervention_reason == "budget: limit exceeded"
    assert diagnostic.would_have_action == "stop"
    assert diagnostic.would_have_reason == "loop: simulated"
    assert diagnostic.would_have_event == 0


def test_unobserved_is_counted_as_a_tool_call():
    first = event(value=True)
    second = ToolEvent(
        tool="edit",
        metadata={
            "progress_proof_unobserved": {
                "spec_hash": first.metadata["progress_proof"]["spec_hash"],
                "unobserved": ["test"],
                "reason": "collection_failed",
            }
        },
    )
    diagnostic = build_coding_diagnostic([first, second], [])
    assert diagnostic.calls_since_progress == 1
    assert diagnostic.contract_state == "unobserved"


def test_multi_resource_outcome_names_the_changed_resource():
    records = []
    for resource, changed in (("unchanged.py", False), ("changed.py", True)):
        for phase in ("before", "after"):
            fingerprint = {"sha256": "new" if changed and phase == "after" else "old"}
            records.append(
                {
                    "schema_version": 1,
                    "tool_event_ref": "call",
                    "phase": phase,
                    "resource_type": "filesystem",
                    "resource_id": resource,
                    "status": "observed",
                    "raw_fingerprint": fingerprint,
                    "canonical_fingerprint": fingerprint,
                    "delta": {"changed": changed} if phase == "after" else None,
                    "collector_version": "filesystem/v1",
                }
            )
    tool_event = ToolEvent(
        tool="edit",
        run_id="call",
        metadata={
            "pathlayer_state_evidence": {
                "schema_version": 1,
                "outcome": "changed",
                "records": records,
            }
        },
    )
    diagnostic = build_coding_diagnostic([tool_event], [])
    assert diagnostic.state_resources == ("changed.py",)
    assert diagnostic.state_outcome == "changed"


def test_calls_are_absent_until_a_contract_is_observed_and_reset_per_epoch():
    no_contract = build_coding_diagnostic([ToolEvent(tool="edit") for _ in range(3)], [])
    assert no_contract.contract_state == "no contract observed"
    assert no_contract.calls_since_progress is None

    diagnostic = build_coding_diagnostic(
        [ToolEvent(tool="edit"), event(task="one"), event(task="two"), event(2, task="two")], []
    )
    assert diagnostic.contract_state == "incomplete"
    assert diagnostic.calls_since_progress == 2


def test_multiple_resources_that_establish_an_outcome_are_all_named():
    records = []
    for resource in ("one.py", "two.py"):
        for phase in ("before", "after"):
            fingerprint = {"sha256": f"{resource}-{phase}"}
            records.append(
                {
                    "schema_version": 1,
                    "tool_event_ref": "call",
                    "phase": phase,
                    "resource_type": "filesystem",
                    "resource_id": resource,
                    "status": "observed",
                    "raw_fingerprint": fingerprint,
                    "canonical_fingerprint": fingerprint,
                    "delta": {"changed": True} if phase == "after" else None,
                    "collector_version": "filesystem/v1",
                }
            )
    diagnostic = build_coding_diagnostic(
        [
            ToolEvent(
                tool="edit",
                run_id="call",
                metadata={
                    "pathlayer_state_evidence": {
                        "schema_version": 1,
                        "outcome": "changed",
                        "records": records,
                    }
                },
            )
        ],
        [],
    )
    assert diagnostic.state_resources == ("one.py", "two.py")
    assert json.loads(json.dumps(diagnostic.to_dict())) == diagnostic.to_dict()
