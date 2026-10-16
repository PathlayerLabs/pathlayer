"""ARCH-1A: the minimal ``StateEvidence`` contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import pathlayer.detectors.state_evidence as state_evidence_module
from pathlayer.detectors.state_evidence import (
    StateEvidence,
    compute_evidence_hash,
    validate_state_evidence,
)

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "state_evidence"
_VALID_FIXTURES = sorted(_FIXTURES_DIR.glob("valid_*.json"))
_INVALID_FIXTURES = sorted(_FIXTURES_DIR.glob("invalid_*.json"))

_OBSERVED_FINGERPRINT = {"exists": True, "sha256": "9f2c" + "0" * 60}
_OBSERVED_CANONICAL = "sha256:9f2c" + "0" * 60


def _evidence_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "tool_event_ref": "call-0001-edit-target",
        "phase": "before",
        "resource_type": "filesystem",
        "resource_id": "src/target.py",
        "status": "observed",
        "raw_fingerprint": _OBSERVED_FINGERPRINT,
        "canonical_fingerprint": _OBSERVED_CANONICAL,
        "delta": None,
        "collector_version": "filesystem-collector@1",
    }
    base.update(overrides)
    return base


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fixture_file:
        payload: dict[str, Any] = json.load(fixture_file)
    return payload


# --- validate_state_evidence: closed schema ---------------------------------


def test_valid_evidence_parses() -> None:
    evidence = validate_state_evidence(_evidence_dict())
    assert evidence is not None
    assert evidence.phase == "before"
    assert evidence.status == "observed"


@pytest.mark.parametrize(
    "missing",
    [
        "schema_version",
        "tool_event_ref",
        "phase",
        "resource_type",
        "resource_id",
        "status",
        "raw_fingerprint",
        "canonical_fingerprint",
        "delta",
        "collector_version",
    ],
)
def test_rejects_missing_field(missing: str) -> None:
    raw = _evidence_dict()
    del raw[missing]
    assert validate_state_evidence(raw) is None


def test_rejects_unknown_field() -> None:
    raw = _evidence_dict()
    raw["extra"] = "nope"
    assert validate_state_evidence(raw) is None


def test_rejects_non_mapping_input() -> None:
    assert validate_state_evidence(["not", "a", "mapping"]) is None
    assert validate_state_evidence(None) is None


# --- validate_state_evidence: schema_version --------------------------------


@pytest.mark.parametrize("version", [0, 2, "1", True])
def test_rejects_unsupported_schema_version(version: object) -> None:
    assert validate_state_evidence(_evidence_dict(schema_version=version)) is None


# --- validate_state_evidence: phase / status enums --------------------------


@pytest.mark.parametrize("phase", ["during", "", "BEFORE", None])
def test_rejects_bad_phase(phase: object) -> None:
    assert validate_state_evidence(_evidence_dict(phase=phase)) is None


@pytest.mark.parametrize("status", ["unknown", "", "OBSERVED", None])
def test_rejects_bad_status(status: object) -> None:
    raw = _evidence_dict(status=status, raw_fingerprint=None, canonical_fingerprint=None)
    assert validate_state_evidence(raw) is None


# --- validate_state_evidence: string field bounds ---------------------------


@pytest.mark.parametrize(
    "field", ["tool_event_ref", "resource_type", "resource_id", "collector_version"]
)
def test_rejects_empty_string_field(field: str) -> None:
    assert validate_state_evidence(_evidence_dict(**{field: ""})) is None


@pytest.mark.parametrize(
    "field", ["tool_event_ref", "resource_type", "resource_id", "collector_version"]
)
def test_rejects_non_string_field(field: str) -> None:
    assert validate_state_evidence(_evidence_dict(**{field: 123})) is None


def test_rejects_oversized_resource_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_evidence_module, "_MAX_RESOURCE_ID_LENGTH", 4)
    assert validate_state_evidence(_evidence_dict(resource_id="way-too-long")) is None


# --- validate_state_evidence: observed <-> fingerprint tri-state ------------


def test_observed_requires_raw_fingerprint() -> None:
    raw = _evidence_dict(raw_fingerprint=None)
    assert validate_state_evidence(raw) is None


def test_observed_requires_canonical_fingerprint() -> None:
    raw = _evidence_dict(canonical_fingerprint=None)
    assert validate_state_evidence(raw) is None


@pytest.mark.parametrize("status", ["unavailable", "collection_failed"])
def test_non_observed_forbids_raw_fingerprint(status: str) -> None:
    raw = _evidence_dict(
        status=status, raw_fingerprint=_OBSERVED_FINGERPRINT, canonical_fingerprint=None
    )
    assert validate_state_evidence(raw) is None


@pytest.mark.parametrize("status", ["unavailable", "collection_failed"])
def test_non_observed_forbids_canonical_fingerprint(status: str) -> None:
    raw = _evidence_dict(
        status=status, raw_fingerprint=None, canonical_fingerprint=_OBSERVED_CANONICAL
    )
    assert validate_state_evidence(raw) is None


@pytest.mark.parametrize("status", ["unavailable", "collection_failed"])
def test_non_observed_with_no_fingerprint_parses(status: str) -> None:
    raw = _evidence_dict(status=status, raw_fingerprint=None, canonical_fingerprint=None)
    evidence = validate_state_evidence(raw)
    assert evidence is not None
    assert evidence.raw_fingerprint is None
    assert evidence.canonical_fingerprint is None
    assert evidence.delta is None


# --- validate_state_evidence: delta / phase interaction ---------------------


def test_before_phase_forbids_delta() -> None:
    raw = _evidence_dict(phase="before", delta={"changed": False})
    assert validate_state_evidence(raw) is None


def test_after_phase_allows_no_delta() -> None:
    raw = _evidence_dict(phase="after", delta=None)
    assert validate_state_evidence(raw) is not None


def test_after_phase_allows_bounded_delta() -> None:
    raw = _evidence_dict(phase="after", delta={"changed": True})
    evidence = validate_state_evidence(raw)
    assert evidence is not None
    assert evidence.delta == {"changed": True}


def test_after_phase_rejects_non_json_delta() -> None:
    raw = _evidence_dict(phase="after", delta={"not", "json", "shaped"})  # a set, not JSON
    assert validate_state_evidence(raw) is None


def test_non_observed_forbids_delta() -> None:
    raw = _evidence_dict(
        phase="after",
        status="collection_failed",
        raw_fingerprint=None,
        canonical_fingerprint=None,
        delta={"changed": True},
    )
    assert validate_state_evidence(raw) is None


# --- validate_state_evidence: JSON-shape bounding ---------------------------


def test_rejects_oversized_fingerprint_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_evidence_module, "_MAX_JSON_STRING_LENGTH", 4)
    raw = _evidence_dict(canonical_fingerprint="way-too-long")
    assert validate_state_evidence(raw) is None


def test_rejects_over_depth_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_evidence_module, "_MAX_JSON_DEPTH", 1)
    raw = _evidence_dict(canonical_fingerprint={"a": {"b": "too-deep"}})
    assert validate_state_evidence(raw) is None


def test_rejects_too_many_json_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_evidence_module, "_MAX_JSON_TOTAL_NODES", 3)
    raw = _evidence_dict(canonical_fingerprint={"a": 1, "b": 2, "c": 3, "d": 4})
    assert validate_state_evidence(raw) is None


def test_rejects_non_json_value_in_fingerprint() -> None:
    raw = _evidence_dict(canonical_fingerprint={"a", "b"})  # a set is not JSON-shaped
    assert validate_state_evidence(raw) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_float_in_fingerprint(value: float) -> None:
    raw = _evidence_dict(canonical_fingerprint={"score": value})
    assert validate_state_evidence(raw) is None


def test_rejects_oversized_integer_magnitude() -> None:
    # A huge int costs _is_bounded_json_value only one node, so without its own
    # magnitude bound this would validate and then crash compute_evidence_hash
    # (json.dumps hits Python's int-to-str conversion limit while hashing).
    raw = _evidence_dict(canonical_fingerprint={"huge": 10**5000})
    assert validate_state_evidence(raw) is None


def test_accepts_int_within_bit_bound() -> None:
    raw = _evidence_dict(canonical_fingerprint={"count": 2**200})
    evidence = validate_state_evidence(raw)
    assert evidence is not None
    compute_evidence_hash(evidence)  # does not raise


def test_accepts_nested_bounded_fingerprint() -> None:
    raw = _evidence_dict(
        canonical_fingerprint={"outcomes": [["test_a", "pass"], ["test_b", "fail"]]}
    )
    evidence = validate_state_evidence(raw)
    assert evidence is not None


# --- compute_evidence_hash: determinism -------------------------------------


def test_compute_evidence_hash_is_deterministic() -> None:
    evidence = validate_state_evidence(_evidence_dict())
    assert evidence is not None
    assert compute_evidence_hash(evidence) == compute_evidence_hash(evidence)


def test_compute_evidence_hash_changes_with_content() -> None:
    a = validate_state_evidence(_evidence_dict())
    b = validate_state_evidence(_evidence_dict(resource_id="src/other.py"))
    assert a is not None
    assert b is not None
    assert compute_evidence_hash(a) != compute_evidence_hash(b)


def test_compute_evidence_hash_has_sha256_prefix() -> None:
    evidence = validate_state_evidence(_evidence_dict())
    assert evidence is not None
    digest = compute_evidence_hash(evidence)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_compute_evidence_hash_is_independent_of_nested_key_order() -> None:
    ordered = _evidence_dict(canonical_fingerprint={"failing": [], "passing": ["t1"]})
    reordered = _evidence_dict(canonical_fingerprint={"passing": ["t1"], "failing": []})
    a = validate_state_evidence(ordered)
    b = validate_state_evidence(reordered)
    assert a is not None
    assert b is not None
    assert compute_evidence_hash(a) == compute_evidence_hash(b)


# --- StateEvidence: hashability divergence from proof.py's dataclasses -----


def test_state_evidence_with_dict_fingerprint_is_not_hashable() -> None:
    evidence = validate_state_evidence(
        _evidence_dict(canonical_fingerprint={"failing": [], "passing": []})
    )
    assert evidence is not None
    with pytest.raises(TypeError):
        hash(evidence)


def test_state_evidence_with_scalar_fingerprint_is_hashable() -> None:
    # raw_fingerprint/canonical_fingerprint are both plain strings here (no
    # dict/list) -- every dataclass field is then a hashable scalar.
    evidence = validate_state_evidence(_evidence_dict(raw_fingerprint=_OBSERVED_CANONICAL))
    assert evidence is not None
    hash(evidence)  # does not raise


def test_state_evidence_equality_is_value_based() -> None:
    a = validate_state_evidence(_evidence_dict())
    b = validate_state_evidence(_evidence_dict())
    assert a is not None
    assert b is not None
    assert a is not b
    assert a == b


# --- on-disk fixtures: valid/invalid round-trip -----------------------------


def test_fixture_directory_is_populated() -> None:
    assert _VALID_FIXTURES, f"no valid_*.json fixtures found under {_FIXTURES_DIR}"
    assert _INVALID_FIXTURES, f"no invalid_*.json fixtures found under {_FIXTURES_DIR}"


@pytest.mark.parametrize("path", _VALID_FIXTURES, ids=lambda p: p.name)
def test_valid_fixture_parses(path: Path) -> None:
    fixture = _load_fixture(path)
    assert fixture["expected_valid"] is True
    evidence = validate_state_evidence(fixture["evidence"])
    assert evidence is not None, f"{path} was expected to be valid but was rejected"
    assert isinstance(evidence, StateEvidence)


@pytest.mark.parametrize("path", _INVALID_FIXTURES, ids=lambda p: p.name)
def test_invalid_fixture_is_rejected(path: Path) -> None:
    fixture = _load_fixture(path)
    assert fixture["expected_valid"] is False
    assert validate_state_evidence(fixture["evidence"]) is None, (
        f"{path} was expected to be rejected but validated"
    )


def test_every_fixture_has_a_description() -> None:
    for path in (*_VALID_FIXTURES, *_INVALID_FIXTURES):
        assert _load_fixture(path)["description"].strip(), f"{path} is missing a description"


def test_valid_fixtures_hash_deterministically() -> None:
    for path in _VALID_FIXTURES:
        evidence = validate_state_evidence(_load_fixture(path)["evidence"])
        assert evidence is not None
        assert compute_evidence_hash(evidence) == compute_evidence_hash(evidence)
