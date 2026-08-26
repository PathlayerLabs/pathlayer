"""ANC-1: deterministic ``checkpoint_set`` progress contract and adversarial replay."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace

import pytest

import pathlayer.detectors.proof as proof_module
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors, loop_suppression_anchors
from pathlayer.detectors.proof import (
    ProofProgressAnchor,
    ProofSpec,
    compute_spec_hash,
    validate_progress_proof,
    validate_proof_spec,
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


def _event(
    *,
    spec: Mapping[str, object] | None = None,
    proof: Mapping[str, object] | None = None,
    tool: str = "report_progress",
) -> ToolEvent:
    payload: dict[str, object] = {}
    if spec is not None:
        payload["proof_spec"] = spec
    if proof is not None:
        payload["progress_proof"] = proof
    return ToolEvent(tool=tool, result=payload)


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
        result={
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


def test_evidence_is_also_read_from_event_metadata() -> None:
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()
    event = ToolEvent(
        tool="report_progress",
        metadata={"proof_spec": spec_a, "progress_proof": _proof_dict(hash_a, 0, {"x": True})},
    )
    assert anchor.measure([event]) == 1.0


def test_evidence_split_across_result_and_metadata_is_not_dropped() -> None:
    # proof_spec and progress_proof are independent keys and may legitimately
    # land on different sides of one event (e.g. an adapter attaching
    # progress_proof to metadata while the tool's own result carries something
    # else entirely) -- neither half should be silently invisible.
    spec_a = _spec_dict(required=("x",))
    hash_a = _hash_of(spec_a)
    anchor = ProofProgressAnchor()

    spec_in_result = ToolEvent(
        tool="report_progress",
        result={"proof_spec": spec_a},
        metadata={"progress_proof": _proof_dict(hash_a, 0, {"x": True})},
    )
    assert anchor.measure([spec_in_result]) == 1.0

    proof_in_result = ToolEvent(
        tool="report_progress",
        result={"progress_proof": _proof_dict(hash_a, 0, {"x": True})},
        metadata={"proof_spec": spec_a},
    )
    assert anchor.measure([proof_in_result]) == 1.0


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
            result={
                "progress_proof": {
                    "spec_hash": hash_b,
                    "sequence": "bad",
                    "checkpoints": {"p": True},
                }
            },
        ),  # malformed, inert
        _event(spec=spec_c, proof=_proof_dict(hash_c, 0, {"q": True})),  # capacity overflow
        _event(proof=_proof_dict(hash_b, 6, {"p": True})),  # genuine advance, resumes epoch B
        _edit(1),
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
        elif roll < 0.3:
            events.append(
                ToolEvent(
                    tool="bad",
                    result={
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
