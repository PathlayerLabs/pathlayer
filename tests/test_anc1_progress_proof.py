"""ANC-1: deterministic ``checkpoint_set`` progress contract and adversarial replay."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace

import pytest

import pathlayer.detectors.proof as proof_module
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors, is_advancing, loop_suppression_anchors
from pathlayer.detectors.proof import (
    ProofProgressAnchor,
    ProofSpec,
    compute_spec_hash,
    validate_progress_proof,
    validate_proof_spec,
    validate_unobserved_marker,
)
from pathlayer.events import EventKind, ToolEvent

_ZERO_HASH = "sha256:" + "0" * 64


def _spec_dict(
    *,
    task_id: str = "task-1",
    proof_id: str = "accepted-deliverables",
    scope_id: str = "scope-1",
    kind: str = "checkpoint_set",
    required: tuple[str, ...] = ("a", "b", "c"),
    schema_version: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "task_id": task_id,
        "proof_id": proof_id,
        "scope_id": scope_id,
        "kind": kind,
        "required": list(required),
    }


def _hash_of(spec_dict: Mapping[str, object]) -> str:
    validated = validate_proof_spec(spec_dict)
    assert validated is not None
    return compute_spec_hash(validated)


def _proof_dict(
    spec_hash: str, sequence: int, checkpoints: Mapping[str, bool]
) -> dict[str, object]:
    return {"spec_hash": spec_hash, "sequence": sequence, "checkpoints": dict(checkpoints)}


def _unobserved_dict(
    spec_hash: str, unobserved: tuple[str, ...], reason: str = "timed out after 5s"
) -> dict[str, object]:
    return {"spec_hash": spec_hash, "unobserved": list(unobserved), "reason": reason}


def _event(
    *,
    spec: Mapping[str, object] | None = None,
    proof: Mapping[str, object] | None = None,
    unobserved: Mapping[str, object] | None = None,
    tool: str = "report_progress",
) -> ToolEvent:
    metadata: dict[str, object] = {}
    if spec is not None:
        metadata["proof_spec"] = spec
    if proof is not None:
        metadata["progress_proof"] = proof
    if unobserved is not None:
        metadata["progress_proof_unobserved"] = unobserved
    return ToolEvent(tool=tool, metadata=metadata)


def _edit(index: int) -> ToolEvent:
    return ToolEvent(tool="edit_file", args={"path": f"f{index}.py"})


# --- validate_proof_spec ------------------------------------------------


def test_valid_proof_spec_parses() -> None:
    spec = validate_proof_spec(_spec_dict(required=("b", "a", "c")))
    assert spec is not None
    assert spec.required == ("a", "b", "c")


@pytest.mark.parametrize(
    "missing", ["schema_version", "task_id", "proof_id", "scope_id", "kind", "required"]
)
def test_proof_spec_rejects_missing_field(missing: str) -> None:
    raw = _spec_dict()
    del raw[missing]
    assert validate_proof_spec(raw) is None


def test_proof_spec_rejects_unknown_field() -> None:
    raw = _spec_dict()
    raw["extra"] = "nope"
    assert validate_proof_spec(raw) is None


@pytest.mark.parametrize("version", [0, 2, "1", True])
def test_proof_spec_rejects_unsupported_schema_version(version: object) -> None:
    raw = _spec_dict()
    raw["schema_version"] = version
    assert validate_proof_spec(raw) is None


def test_proof_spec_rejects_non_checkpoint_set_kind() -> None:
    assert validate_proof_spec(_spec_dict(kind="numeric_score")) is None


def test_proof_spec_rejects_empty_required() -> None:
    raw = _spec_dict()
    raw["required"] = []
    assert validate_proof_spec(raw) is None


def test_proof_spec_rejects_duplicate_required_ids() -> None:
    raw = _spec_dict()
    raw["required"] = ["a", "a"]
    assert validate_proof_spec(raw) is None


def test_proof_spec_rejects_required_as_string_not_iterated_as_chars() -> None:
    raw = _spec_dict()
    raw["required"] = "abc"
    assert validate_proof_spec(raw) is None


@pytest.mark.parametrize("field", ["task_id", "proof_id", "scope_id"])
def test_proof_spec_rejects_empty_string_fields(field: str) -> None:
    raw = _spec_dict()
    raw[field] = ""
    assert validate_proof_spec(raw) is None


def test_proof_spec_rejects_overlong_required_id() -> None:
    raw = _spec_dict(required=("x" * (proof_module._MAX_CHECKPOINT_ID_LENGTH + 1),))
    assert validate_proof_spec(raw) is None


def test_proof_spec_required_over_capacity_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proof_module, "_MAX_REQUIRED_CHECKPOINTS", 2)
    assert validate_proof_spec(_spec_dict(required=("a", "b", "c"))) is None


def test_proof_spec_required_exactly_at_capacity_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proof_module, "_MAX_REQUIRED_CHECKPOINTS", 2)
    assert validate_proof_spec(_spec_dict(required=("a", "b"))) is not None


def test_proof_spec_required_order_independent_via_hash() -> None:
    spec1 = validate_proof_spec(_spec_dict(required=("a", "b", "c")))
    spec2 = validate_proof_spec(_spec_dict(required=("c", "a", "b")))
    assert spec1 is not None
    assert spec2 is not None
    assert compute_spec_hash(spec1) == compute_spec_hash(spec2)


@pytest.mark.parametrize("raw", [None, "not-a-mapping", 42, ["a", "list"]])
def test_proof_spec_rejects_non_mapping_input(raw: object) -> None:
    assert validate_proof_spec(raw) is None


# --- validate_progress_proof --------------------------------------------


def test_valid_progress_proof_parses() -> None:
    proof = validate_progress_proof(_proof_dict(_ZERO_HASH, 0, {"x": True}))
    assert proof is not None
    assert proof.checkpoints == (("x", True),)


def test_progress_proof_rejects_unknown_field() -> None:
    raw = _proof_dict(_ZERO_HASH, 0, {"x": True})
    raw["extra"] = 1
    assert validate_progress_proof(raw) is None


@pytest.mark.parametrize("missing", ["spec_hash", "sequence", "checkpoints"])
def test_progress_proof_rejects_missing_field(missing: str) -> None:
    raw = _proof_dict(_ZERO_HASH, 0, {"x": True})
    del raw[missing]
    assert validate_progress_proof(raw) is None


@pytest.mark.parametrize(
    "spec_hash",
    [
        "0" * 64,  # no prefix
        "sha256:" + "0" * 63,  # too short
        "sha256:" + "A" * 64,  # uppercase hex
        123,  # non-string
    ],
)
def test_progress_proof_rejects_malformed_spec_hash(spec_hash: object) -> None:
    raw = {"spec_hash": spec_hash, "sequence": 0, "checkpoints": {"x": True}}
    assert validate_progress_proof(raw) is None


@pytest.mark.parametrize("sequence", [-1, True, 1.0, "0"])
def test_progress_proof_rejects_bad_sequence(sequence: object) -> None:
    raw = {"spec_hash": _ZERO_HASH, "sequence": sequence, "checkpoints": {"x": True}}
    assert validate_progress_proof(raw) is None


def test_progress_proof_rejects_non_mapping_or_empty_checkpoints() -> None:
    base = {"spec_hash": _ZERO_HASH, "sequence": 0}
    assert validate_progress_proof({**base, "checkpoints": []}) is None
    assert validate_progress_proof({**base, "checkpoints": {}}) is None


@pytest.mark.parametrize("value", [1, "true", None, 0.9])
def test_progress_proof_rejects_non_bool_checkpoint_value(value: object) -> None:
    raw = {"spec_hash": _ZERO_HASH, "sequence": 0, "checkpoints": {"x": value}}
    assert validate_progress_proof(raw) is None


def test_progress_proof_rejects_overlong_checkpoints_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proof_module, "_MAX_CHECKPOINTS_PER_PROOF", 2)
    raw = _proof_dict(_ZERO_HASH, 0, {"a": True, "b": True, "c": True})
    assert validate_progress_proof(raw) is None


def test_progress_proof_checkpoints_exactly_at_capacity_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proof_module, "_MAX_CHECKPOINTS_PER_PROOF", 2)
    raw = _proof_dict(_ZERO_HASH, 0, {"a": True, "b": True})
    assert validate_progress_proof(raw) is not None


@pytest.mark.parametrize("raw", [None, "not-a-mapping", 42, ["a", "list"]])
def test_progress_proof_rejects_non_mapping_input(raw: object) -> None:
    assert validate_progress_proof(raw) is None


# --- compute_spec_hash ---------------------------------------------------


def test_spec_hash_is_deterministic() -> None:
    spec = validate_proof_spec(_spec_dict())
    assert spec is not None
    assert compute_spec_hash(spec) == compute_spec_hash(spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "other-task"),
        ("proof_id", "other-proof"),
        ("scope_id", "other-scope"),
        ("required", ("z",)),
        ("kind", "checkpoint_set_v2"),
    ],
)
def test_spec_hash_changes_with_any_field(field: str, value: object) -> None:
    base = validate_proof_spec(_spec_dict())
    assert base is not None
    changed: ProofSpec = replace(base, **{field: value})  # type: ignore[arg-type]
    assert compute_spec_hash(base) != compute_spec_hash(changed)


def test_spec_hash_has_sha256_prefix_and_length() -> None:
    spec = validate_proof_spec(_spec_dict())
    assert spec is not None
    digest = compute_spec_hash(spec)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_proof_spec_equality_and_hash_are_value_based() -> None:
    spec1 = validate_proof_spec(_spec_dict(required=("a", "b")))
    spec2 = validate_proof_spec(_spec_dict(required=("b", "a")))  # same set, different order
    spec3 = validate_proof_spec(_spec_dict(required=("a", "c")))
    assert spec1 is not None
    assert spec2 is not None
    assert spec3 is not None
    assert spec1 == spec2
    assert hash(spec1) == hash(spec2)
    assert spec1 != spec3


def test_progress_proof_equality_and_hash_are_value_based() -> None:
    proof1 = validate_progress_proof(_proof_dict(_ZERO_HASH, 0, {"a": True, "b": False}))
    proof2 = validate_progress_proof(_proof_dict(_ZERO_HASH, 0, {"b": False, "a": True}))
    proof3 = validate_progress_proof(_proof_dict(_ZERO_HASH, 1, {"a": True, "b": False}))
    assert proof1 is not None
    assert proof2 is not None
    assert proof3 is not None
    assert proof1 == proof2
    assert hash(proof1) == hash(proof2)
    assert proof1 != proof3


# --- level derivation / baseline / progression ---------------------------


def test_first_observation_is_real_baseline_no_artificial_credit() -> None:
    spec_a = _spec_dict(required=("x", "y", "z"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    assert anchor.measure(history) == 1.0


def test_measure_counts_only_required_checkpoints_true() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True, "z": True}))]
    assert anchor.measure(history) == 1.0  # "z" is not required, ignored


def test_level_rises_as_checkpoints_turn_true() -> None:
    spec_a = _spec_dict(required=("x", "y", "z"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    assert anchor.measure(history) == 1.0
    history.append(_event(proof=_proof_dict(hash_a, 1, {"y": True})))
    assert anchor.measure(history) == 2.0
    history.append(_event(proof=_proof_dict(hash_a, 2, {"z": True})))
    assert anchor.measure(history) == 3.0


def test_level_can_fall_on_reported_regression() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True, "y": True}))]
    assert anchor.measure(history) == 2.0
    history.append(_event(proof=_proof_dict(hash_a, 1, {"x": False})))
    assert anchor.measure(history) == 1.0


def test_is_satisfied_only_when_all_required_true() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    assert anchor.is_satisfied(history) is False
    history.append(_event(proof=_proof_dict(hash_a, 1, {"y": True})))
    assert anchor.is_satisfied(history) is True
    history.append(_event(proof=_proof_dict(hash_a, 2, {"x": False})))
    assert anchor.is_satisfied(history) is False


# --- scope / spec_hash change --------------------------------------------


def test_spec_hash_change_emits_gap_then_fresh_baseline() -> None:
    anchor = ProofProgressAnchor()
    spec_a = _spec_dict(scope_id="scope-a", required=("x", "y"))
    hash_a = _hash_of(spec_a)
    spec_b = _spec_dict(scope_id="scope-b", required=("p", "q", "r"))
    hash_b = _hash_of(spec_b)

    history = [
        _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True})),
        _event(proof=_proof_dict(hash_a, 1, {"x": True, "y": True})),
    ]
    assert anchor.measure(history) == 2.0

    history.append(_event(spec=spec_b, proof=_proof_dict(hash_b, 0, {"p": True})))
    assert anchor.measure(history) is None  # gap on the transition tick

    history.append(_edit(0))
    assert anchor.measure(history) == 1.0  # fresh epoch B baseline, never 2 + 1


def test_scope_id_change_implies_spec_hash_change() -> None:
    spec_a = validate_proof_spec(_spec_dict(scope_id="scope-a"))
    spec_b = validate_proof_spec(_spec_dict(scope_id="scope-b"))
    assert spec_a is not None
    assert spec_b is not None
    assert compute_spec_hash(spec_a) != compute_spec_hash(spec_b)


def test_epoch_switch_resets_no_progress_cusum() -> None:
    detector = NoProgressDetector([ProofProgressAnchor()], target_rate=0.5, threshold=2.0, window=8)
    spec_a = _spec_dict(scope_id="scope-a", required=("x", "y"))
    hash_a = _hash_of(spec_a)
    spec_b = _spec_dict(scope_id="scope-b", required=("p", "q"))
    hash_b = _hash_of(spec_b)

    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    history.extend(_edit(i) for i in range(3))
    assert detector.evaluate(history).confidence == pytest.approx(0.75)

    history.append(_event(spec=spec_b, proof=_proof_dict(hash_b, 0, {"p": True})))
    assert detector.evaluate(history).confidence == 0.0  # reset, not a continuation/spike


# --- absence / invalidity / out-of-order -> inert abstention ------------


def test_no_progress_proof_in_history_is_inapplicable() -> None:
    anchor = ProofProgressAnchor()
    history = [_edit(i) for i in range(5)]
    assert anchor.measure(history) is None
    assert anchor.is_satisfied(history) is False
    assert anchor.diagnose(history) == ()


def test_proof_spec_only_event_registers_but_does_not_drive_level() -> None:
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a)]
    assert anchor.measure(history) is None
    history.append(_event(proof=_proof_dict(hash_a, 0, {"x": True})))
    assert anchor.measure(history) == 1.0


def test_unknown_spec_hash_is_inert() -> None:
    anchor = ProofProgressAnchor()
    history = [_event(proof=_proof_dict(_ZERO_HASH, 0, {"x": True}))]
    assert anchor.measure(history) is None
    assert "no known proof_spec" in anchor.diagnose(history)[0]


def test_malformed_progress_proof_is_inert_not_reset() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    assert anchor.measure(history) == 1.0
    malformed = ToolEvent(
        tool="report_progress",
        metadata={
            "progress_proof": {"spec_hash": hash_a, "sequence": "bad", "checkpoints": {"x": True}}
        },
    )
    history.append(malformed)
    assert anchor.measure(history) == 1.0  # unchanged, not reset to None
    assert "malformed progress_proof ignored" in anchor.diagnose(history)[0]


def test_malformed_proof_spec_is_inert_progress_proof_still_applies() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    bad_spec = {**spec_a, "kind": "not_checkpoint_set"}
    history = [_event(spec=bad_spec, proof=_proof_dict(hash_a, 0, {"x": True}))]
    # the malformed spec is dropped, but an already-known spec_hash still works
    assert anchor.measure(history) is None
    assert "malformed proof_spec ignored" in anchor.diagnose(history)[0]

    history = [
        _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True})),
        _event(spec=bad_spec, proof=_proof_dict(hash_a, 1, {"y": True})),
    ]
    assert anchor.measure(history) == 2.0
    assert "malformed proof_spec ignored" in anchor.diagnose(history)[0]


def test_evidence_is_read_from_event_metadata_only() -> None:
    """``event.result`` is the observed agent's own tool output; a proof there
    is a self-reported claim, which the contract's trust boundary rules out.
    Only ``event.metadata`` -- written by an adapter or producer outside the
    agent's control -- can carry ``proof_spec``/``progress_proof``."""
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    in_result = ToolEvent(
        tool="report_progress",
        result={"proof_spec": spec_a, "progress_proof": _proof_dict(hash_a, 0, {"x": True})},
    )
    assert anchor.measure([in_result]) is None
    assert anchor.diagnose([in_result]) == ()

    in_metadata = ToolEvent(
        tool="report_progress",
        metadata={"proof_spec": spec_a, "progress_proof": _proof_dict(hash_a, 0, {"x": True})},
    )
    assert anchor.measure([in_metadata]) == 1.0


def test_tool_result_cannot_forge_or_override_the_producers_proof() -> None:
    """The forgery this closes: the producer observed ``x`` false, the tool's
    result claims it true. The anchor must keep the producer's word -- a forged
    satisfied proof would silence ``no_progress`` for the rest of the run."""
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    forged = ToolEvent(
        tool="report_progress",
        result={"progress_proof": _proof_dict(hash_a, 7, {"x": True})},
        metadata={"proof_spec": spec_a, "progress_proof": _proof_dict(hash_a, 0, {"x": False})},
    )
    assert anchor.measure([forged]) == 0.0
    assert anchor.is_satisfied([forged]) is False

    # Half a payload in the result is not completed by the other half in
    # metadata either: each key is resolved in metadata alone.
    split = ToolEvent(
        tool="report_progress",
        result={"proof_spec": spec_a},
        metadata={"progress_proof": _proof_dict(hash_a, 0, {"x": True})},
    )
    assert anchor.measure([split]) is None
    assert "no known proof_spec" in anchor.diagnose([split])[0]


def test_out_of_order_sequence_is_inert() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 5, {"x": True, "y": True}))]
    assert anchor.measure(history) == 2.0
    history.append(_event(proof=_proof_dict(hash_a, 3, {"x": False, "y": False})))
    assert anchor.measure(history) == 2.0
    assert "out-of-order" in anchor.diagnose(history)[0]


def test_same_sequence_restated_with_a_contradicting_value_is_inert() -> None:
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    assert anchor.measure(history) == 1.0
    history.append(_event(proof=_proof_dict(hash_a, 0, {"x": False})))
    assert anchor.measure(history) == 1.0
    assert "already accepted" in anchor.diagnose(history)[0]


def test_same_sequence_restated_with_purely_additive_state_is_also_inert() -> None:
    # Not a value-level conflict (no checkpoint id disagrees) -- "x" is simply
    # new information at an already-used sequence number. Still rejected
    # wholesale, never merged: a revision number must identify one coherent
    # snapshot, so the producer must bump `sequence` to add facts.
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    assert anchor.measure(history) == 1.0
    history.append(_event(proof=_proof_dict(hash_a, 0, {"y": True})))
    assert anchor.measure(history) == 1.0  # "y" is dropped, not merged in
    assert "already accepted" in anchor.diagnose(history)[0]


def test_spec_hash_mismatch_between_colocated_spec_and_proof_is_inert() -> None:
    spec_a = _spec_dict(scope_id="scope-a", required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    mismatched = _event(spec=spec_a, proof=_proof_dict(_ZERO_HASH, 0, {"x": True}))
    assert anchor.measure([mismatched]) is None

    # spec_a is still cached under its own correct hash, usable by a follow-up.
    follow_up = _event(proof=_proof_dict(hash_a, 0, {"x": True}))
    assert anchor.measure([mismatched, follow_up]) == 1.0


# --- duplicate idempotency ------------------------------------------------


def test_exact_duplicate_sequence_and_state_is_idempotent_no_op() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    proof = _proof_dict(hash_a, 0, {"x": True, "y": False})
    history = [_event(spec=spec_a, proof=proof)]
    assert anchor.measure(history) == 1.0
    history.append(_event(proof=dict(proof)))
    assert anchor.measure(history) == 1.0
    assert anchor.diagnose(history) == ()


# --- capacity bounds --------------------------------------------------


def test_spec_capacity_reached_new_spec_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proof_module, "_MAX_TRACKED_SPEC_HASHES", 1)
    spec_a = _spec_dict(scope_id="scope-a", required=("x",))
    hash_a = _hash_of(spec_a)
    spec_b = _spec_dict(scope_id="scope-b", required=("y",))
    hash_b = _hash_of(spec_b)
    anchor = ProofProgressAnchor()

    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    assert anchor.measure(history) == 1.0

    history.append(_event(spec=spec_b, proof=_proof_dict(hash_b, 0, {"y": True})))
    assert anchor.measure(history) == 1.0  # spec B dropped by capacity, epoch A untouched
    diagnostics = anchor.diagnose(history)
    assert any("capacity" in message for message in diagnostics)
    assert any("no known proof_spec" in message for message in diagnostics)


# --- PERF-2 equivalence + diagnose/step consistency -----------------------


def _assert_step_matches_batch(anchor: ProofProgressAnchor, events: list[ToolEvent]) -> None:
    state = anchor.create_state()
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, incremental = anchor.step(state, event)
        assert incremental.level == anchor.measure(history)
        assert incremental.satisfied == anchor.is_satisfied(history)
        assert incremental.diagnostics == anchor.diagnose(history)


def test_anchor_step_matches_batch_on_hand_built_adversarial_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proof_module, "_MAX_TRACKED_SPEC_HASHES", 2)
    spec_a = _spec_dict(scope_id="scope-a", required=("x", "y"))
    hash_a = _hash_of(spec_a)
    spec_b = _spec_dict(scope_id="scope-b", required=("p",))
    hash_b = _hash_of(spec_b)
    spec_c = _spec_dict(scope_id="scope-c", required=("q",))
    hash_c = _hash_of(spec_c)

    events = [
        _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True})),
        _event(proof=_proof_dict(hash_a, 1, {"x": True, "y": True})),  # satisfied
        _edit(0),
        _event(spec=spec_b, proof=_proof_dict(hash_b, 5, {"p": True})),  # epoch switch -> gap
        _event(proof=_proof_dict(hash_b, 3, {"p": False})),  # out-of-order, inert
        _event(proof=_proof_dict(hash_b, 5, {"p": False})),  # conflicting duplicate, inert
        ToolEvent(
            tool="bad",
            metadata={
                "progress_proof": {
                    "spec_hash": hash_b,
                    "sequence": "bad",
                    "checkpoints": {"p": True},
                }
            },
        ),  # malformed, inert
        _event(spec=spec_c, proof=_proof_dict(hash_c, 0, {"q": True})),  # capacity overflow
        _event(proof=_proof_dict(hash_b, 6, {"p": True})),  # genuine advance, resumes epoch B
        _event(spec=spec_b, unobserved=_unobserved_dict(hash_b, ("p",))),  # total outage: gap
        _event(unobserved=_unobserved_dict(hash_a, ("x",))),  # other epoch's marker: inert
        _event(unobserved={"spec_hash": hash_b, "unobserved": [], "reason": "x"}),  # malformed
        _event(
            proof=_proof_dict(hash_b, 7, {"p": False}),
            unobserved=_unobserved_dict(hash_b, ("p",), "did not report 1 of 1"),
        ),  # partial: ledger updated, still a gap
        _edit(1),
        _event(proof=_proof_dict(hash_b, 8, {"p": True})),  # fresh baseline after the gap
    ]
    _assert_step_matches_batch(ProofProgressAnchor(), events)


def _random_events(seed: int, count: int) -> list[ToolEvent]:
    rng = random.Random(seed)
    specs = [_spec_dict(scope_id=f"scope-{i}", required=(f"c{i}a", f"c{i}b")) for i in range(3)]
    hashes = [_hash_of(spec) for spec in specs]
    sequences = [0, 0, 0]
    events: list[ToolEvent] = []
    for i in range(count):
        roll = rng.random()
        if roll < 0.15:
            events.append(_edit(i))
        elif roll < 0.22:
            index = rng.randrange(3)
            events.append(
                _event(
                    unobserved=_unobserved_dict(
                        hashes[index], (f"c{index}a",) if rng.random() < 0.5 else (f"c{index}b",)
                    )
                )
            )
        elif roll < 0.3:
            events.append(
                ToolEvent(
                    tool="bad",
                    metadata={
                        "progress_proof": {
                            "spec_hash": hashes[0],
                            "sequence": "not-an-int",
                            "checkpoints": {},
                        }
                    },
                )
            )
        else:
            index = rng.randrange(3)
            if rng.random() < 0.7:
                sequences[index] += 1
            sequence = sequences[index] if rng.random() < 0.85 else max(0, sequences[index] - 1)
            checkpoints = {
                f"c{index}a": rng.random() < 0.6,
                f"c{index}b": rng.random() < 0.6,
            }
            include_spec = rng.random() < 0.3
            events.append(
                _event(
                    spec=specs[index] if include_spec else None,
                    proof=_proof_dict(hashes[index], sequence, checkpoints),
                )
            )
    return events


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 42])
def test_anchor_step_matches_batch_on_generated_corpus(seed: int) -> None:
    _assert_step_matches_batch(ProofProgressAnchor(), _random_events(seed, 60))


def test_anchor_step_matches_batch_on_empty_history() -> None:
    anchor = ProofProgressAnchor()
    assert anchor.measure([]) is None
    assert anchor.is_satisfied([]) is False
    assert anchor.diagnose([]) == ()
    assert hash(anchor.create_state()) == hash(anchor.create_state())


def test_state_is_hashable_after_nontrivial_sequence() -> None:
    anchor = ProofProgressAnchor()
    state = anchor.create_state()
    for event in _random_events(99, 40):
        state, _ = anchor.step(state, event)
    assert hash(state) is not None


# --- progress_proof_unobserved: a failed observation is a gap, not a freeze ---


def test_valid_unobserved_marker_parses() -> None:
    marker = validate_unobserved_marker(_unobserved_dict(_ZERO_HASH, ("b", "a")))
    assert marker is not None
    assert marker.spec_hash == _ZERO_HASH
    assert marker.unobserved == ("a", "b")
    assert marker.reason == "timed out after 5s"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not a mapping",
        {"spec_hash": _ZERO_HASH, "unobserved": ["a"]},  # missing reason
        {"spec_hash": _ZERO_HASH, "unobserved": ["a"], "reason": "r", "extra": 1},
        {"spec_hash": "sha256:short", "unobserved": ["a"], "reason": "r"},
        {"spec_hash": _ZERO_HASH, "unobserved": [], "reason": "r"},
        {"spec_hash": _ZERO_HASH, "unobserved": "abc", "reason": "r"},
        {"spec_hash": _ZERO_HASH, "unobserved": ["a", "a"], "reason": "r"},
        {"spec_hash": _ZERO_HASH, "unobserved": ["a", 1], "reason": "r"},
        {"spec_hash": _ZERO_HASH, "unobserved": [""], "reason": "r"},
        {"spec_hash": _ZERO_HASH, "unobserved": ["a"], "reason": ""},
        {"spec_hash": _ZERO_HASH, "unobserved": ["a"], "reason": 3},
        {"spec_hash": _ZERO_HASH, "unobserved": ["a"], "reason": "r" * 257},
        {"spec_hash": _ZERO_HASH, "unobserved": ["a" * 257], "reason": "r"},
        {"spec_hash": _ZERO_HASH, "unobserved": [f"c{i}" for i in range(257)], "reason": "r"},
    ],
)
def test_unobserved_marker_rejects_malformed_input(raw: object) -> None:
    assert validate_unobserved_marker(raw) is None


def test_unobserved_marker_is_a_gap_that_keeps_the_ledger() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    assert anchor.measure(history) == 1.0

    history.append(_event(spec=spec_a, unobserved=_unobserved_dict(hash_a, ("x", "y"))))
    assert anchor.measure(history) is None  # this tick: no level, CUSUM resets
    assert anchor.is_satisfied(history) is False
    (diagnostic,) = anchor.diagnose(history)
    assert "2 required checkpoint(s) unobserved" in diagnostic
    assert "timed out after 5s" in diagnostic
    assert "gap" in diagnostic

    history.append(_edit(0))  # a tick without any producer payload: last level is back
    assert anchor.measure(history) == 1.0
    assert anchor.diagnose(history) == ()

    history.append(_event(proof=_proof_dict(hash_a, 1, {"y": True})))
    assert anchor.measure(history) == 2.0  # the ledger was never erased
    assert anchor.is_satisfied(history) is True


def test_partial_proof_with_unobserved_marker_updates_ledger_but_gaps() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [
        _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": False, "y": False})),
        _event(
            proof=_proof_dict(hash_a, 1, {"x": True}),
            unobserved=_unobserved_dict(hash_a, ("y",), "did not report 1 of 2"),
        ),
    ]
    assert anchor.measure(history) is None
    history.append(_event(proof=_proof_dict(hash_a, 2, {"x": True, "y": False})))
    assert anchor.measure(history) == 1.0  # x's value from the partial proof was kept


def test_unobserved_marker_never_switches_epoch() -> None:
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    spec_b = _spec_dict(scope_id="scope-b", required=("p",))
    hash_b = _hash_of(spec_b)
    anchor = ProofProgressAnchor()
    history = [
        _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True})),
        _event(spec=spec_b, unobserved=_unobserved_dict(hash_b, ("p",))),
    ]
    assert anchor.measure(history) == 1.0  # inert: epoch A stays active and credited
    (diagnostic,) = anchor.diagnose(history)
    assert "another epoch" in diagnostic
    history.append(_event(proof=_proof_dict(hash_a, 1, {"x": True})))
    assert anchor.measure(history) == 1.0
    assert anchor.diagnose(history) == ()


def test_unobserved_marker_before_any_epoch_is_diagnosed_and_inapplicable() -> None:
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, unobserved=_unobserved_dict(hash_a, ("x",)))]
    assert anchor.measure(history) is None
    assert "gap" in anchor.diagnose(history)[0]
    # The spec was still cached: a later proof-only event starts the epoch.
    history.append(_event(proof=_proof_dict(hash_a, 0, {"x": True})))
    assert anchor.measure(history) == 1.0


def test_malformed_unobserved_marker_is_inert() -> None:
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [
        _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": False})),
        _event(unobserved={"spec_hash": hash_a, "unobserved": [], "reason": "r"}),
    ]
    assert anchor.measure(history) == 0.0
    assert "malformed progress_proof_unobserved ignored" in anchor.diagnose(history)[0]


def test_unobserved_marker_in_result_is_invisible() -> None:
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [
        _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": False})),
        ToolEvent(tool="t", result={"progress_proof_unobserved": _unobserved_dict(hash_a, ("x",))}),
    ]
    assert anchor.measure(history) == 0.0
    assert anchor.diagnose(history) == ()


def test_unobserved_diagnostic_bounds_the_listed_checkpoints() -> None:
    required = tuple(f"c{i:02d}" for i in range(8))
    spec_a = _spec_dict(required=required)
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [_event(spec=spec_a, unobserved=_unobserved_dict(hash_a, required))]
    (diagnostic,) = anchor.diagnose(history)
    assert "8 required checkpoint(s) unobserved" in diagnostic
    assert "c04" in diagnostic and "c05" not in diagnostic and "..." in diagnostic


def test_probe_outage_mid_run_never_stops_a_productive_agent() -> None:
    """The ANC-4 freeze: ten ticks the probe cannot observe used to count as
    ten steps without progress. With the marker they are gaps."""
    detector = NoProgressDetector([ProofProgressAnchor()])
    spec_a = _spec_dict(required=("x", "y", "z", "w"))
    hash_a = _hash_of(spec_a)
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))]
    history.extend(
        _event(spec=spec_a, unobserved=_unobserved_dict(hash_a, ("x", "y", "z", "w")))
        for _ in range(20)
    )
    confidences = [detector.evaluate(history[:i]).confidence for i in range(1, len(history) + 1)]
    assert max(confidences) == 0.0
    # Back online: a fresh baseline, then genuine stagnation is still caught.
    history.append(_event(proof=_proof_dict(hash_a, 1, {"x": True})))
    history.extend(_edit(i) for i in range(20))
    assert detector.evaluate(history).confidence >= 0.9


def test_probe_outage_gap_is_never_read_as_advancing() -> None:
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    history = [
        _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": False})),
        _event(spec=spec_a, unobserved=_unobserved_dict(hash_a, ("x", "y"))),
    ]
    assert is_advancing([anchor], history) is False
    history.append(_event(proof=_proof_dict(hash_a, 1, {"x": True})))
    assert is_advancing([anchor], history) is False  # a baseline after a gap is not a record
    history.append(_event(proof=_proof_dict(hash_a, 2, {"x": True, "y": True})))
    assert is_advancing([anchor], history) is True


# --- an outage in the production anchor set (coding anchors + proof anchor) ---


def _production_detector() -> NoProgressDetector:
    """The anchor set ``build_pipeline`` wires when ``progress_proof`` is enabled."""
    return NoProgressDetector([*coding_anchors(), ProofProgressAnchor()])


def _confidences(detector: NoProgressDetector, events: list[ToolEvent]) -> list[float]:
    state = detector.create_state()
    confidences = []
    for event in events:
        state, signal = detector.step(state, event)
        confidences.append(signal.confidence)
    return confidences


def _failing_command(metadata: Mapping[str, object]) -> ToolEvent:
    return ToolEvent(tool="shell", result={"exit_code": 1}, metadata=dict(metadata))


def test_outage_after_the_objective_never_unmasks_a_stalled_coding_anchor() -> None:
    """Objective reached, the agent keeps running a failing command, then the
    probe dies. The satisfied proof held no_progress silent; its outage must
    not hand the decision to the exit-code anchor, flat all along. As a plain
    gap it did: STOP on the very first unobserved call."""
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    events = [
        _failing_command(
            {"proof_spec": spec_a, "progress_proof": _proof_dict(hash_a, i, {"x": True, "y": True})}
            if i < 15
            else {
                "proof_spec": spec_a,
                "progress_proof_unobserved": _unobserved_dict(hash_a, ("x", "y")),
            }
        )
        for i in range(40)
    ]
    assert max(_confidences(_production_detector(), events)) == 0.0
    # The stall the proof was covering is real: the coding anchors alone fire.
    assert max(_confidences(NoProgressDetector(coding_anchors()), events)) == 1.0


def test_outage_of_an_advancing_proof_holds_then_restarts_patience() -> None:
    required = tuple(f"c{k}" for k in range(10))
    spec_a = _spec_dict(required=required)
    hash_a = _hash_of(spec_a)
    events = []
    for i in range(50):
        if 19 <= i < 30:  # the probe is down for eleven calls
            metadata: dict[str, object] = {
                "proof_spec": spec_a,
                "progress_proof_unobserved": _unobserved_dict(hash_a, required),
            }
        else:  # one checkpoint turns true every three calls, until c6
            checkpoints = {f"c{k}": i >= 3 * k and k < 7 for k in range(10)}
            metadata = {"proof_spec": spec_a, "progress_proof": _proof_dict(hash_a, i, checkpoints)}
        events.append(_failing_command(metadata))
    confidences = _confidences(_production_detector(), events)
    assert max(confidences[:30]) < 0.9  # no STOP before or during the outage
    assert confidences[19:30] == [0.0] * 11
    # Back online at call 31: a fresh baseline, then a full patience again.
    assert confidences[30] == 0.0
    assert max(confidences[30:39]) < 0.9
    assert confidences[39] == pytest.approx(0.9)


def test_a_probe_that_never_worked_leaves_the_coding_anchors_in_charge() -> None:
    """Nothing was ever measured, so there is no measurement to lose: the other
    anchors decide exactly as without a contract (zero-config outcome)."""
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    events = [
        _failing_command(
            {"proof_spec": spec_a, "progress_proof_unobserved": _unobserved_dict(hash_a, ("x",))}
        )
        for _ in range(20)
    ]
    production = _confidences(_production_detector(), events)
    assert production == _confidences(NoProgressDetector(coding_anchors()), events)
    assert max(production) == 1.0


def test_unavailable_is_flagged_only_when_an_epoch_was_being_measured() -> None:
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    spec_b = _spec_dict(scope_id="scope-b", required=("p",))
    hash_b = _hash_of(spec_b)
    anchor = ProofProgressAnchor()
    marker_a = _event(spec=spec_a, unobserved=_unobserved_dict(hash_a, ("x",)))
    assert anchor.is_unavailable([marker_a]) is False  # no epoch yet: inapplicable only

    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": False})), marker_a]
    assert anchor.is_unavailable(history) is True
    state = anchor.create_state()
    for event in history:
        state, snapshot = anchor.step(state, event)
    assert snapshot.unavailable is True and snapshot.level is None

    for inert in (
        _edit(0),
        _event(spec=spec_b, unobserved=_unobserved_dict(hash_b, ("p",))),
        _event(unobserved={"spec_hash": hash_a, "unobserved": [], "reason": "r"}),
    ):
        assert anchor.is_unavailable([*history[:1], inert]) is False


# --- NoProgressDetector integration --------------------------------------


def test_stagnant_after_baseline_fires_no_progress() -> None:
    detector = NoProgressDetector([ProofProgressAnchor()])
    spec_a = _spec_dict(required=("x", "y", "z", "w"))
    hash_a = _hash_of(spec_a)
    baseline = _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))
    stagnant = [baseline, *(_edit(i) for i in range(20))]
    confidences = [detector.evaluate(stagnant[:i]).confidence for i in range(1, len(stagnant) + 1)]
    assert confidences[-1] >= 0.9


def test_all_required_satisfied_suppresses_no_progress() -> None:
    detector = NoProgressDetector([ProofProgressAnchor()])
    spec_a = _spec_dict(required=("x", "y"))
    hash_a = _hash_of(spec_a)
    history = [_event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True, "y": True}))]
    history.extend(_edit(i) for i in range(20))
    assert detector.evaluate(history).confidence == 0.0


def test_non_tool_call_events_do_not_consume_patience() -> None:
    detector = NoProgressDetector([ProofProgressAnchor()])
    spec_a = _spec_dict(required=("x", "y", "z"))
    hash_a = _hash_of(spec_a)
    baseline = _event(spec=spec_a, proof=_proof_dict(hash_a, 0, {"x": True}))
    llm_events = [ToolEvent(tool="llm", kind=EventKind.LLM_CALL, cost_usd=1.0) for _ in range(20)]
    assert detector.evaluate([baseline, *llm_events]).confidence == 0.0


# --- non-wiring pin --------------------------------------------------------


def test_coding_anchors_and_loop_suppression_anchors_exclude_proof_anchor() -> None:
    assert not any(isinstance(anchor, ProofProgressAnchor) for anchor in coding_anchors())
    assert not any(isinstance(anchor, ProofProgressAnchor) for anchor in loop_suppression_anchors())
