"""ANC-4: durable state written before ANC-4 is never resumed with a proof contract.

ANC-4 changed how an *enabled* ``progress_proof`` section reads events: proofs
come from ``ToolEvent.metadata`` only, never ``ToolEvent.result``, and an
unobserved marker holds ``no_progress`` silent. Full prefix replay does not
always expose that (Kropemann's review on #192): a one-event prefix whose
``result``-side proof only set a baseline decides the same under both
readings, so a resume is accepted, yet the anchor restarts from a different
state and the *next* decision silently changes. The fingerprint of a
proof-enabled configuration therefore carries a proof semantics version; every
other configuration keeps its historical fingerprint and resumes as before.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import pathlayer.detectors.proof as proof_module
from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.proof import compute_spec_hash, validate_proof_spec
from pathlayer.events import ToolEvent
from pathlayer.store import (
    PersistentPipeline,
    SQLiteStateStore,
    StateCompatibilityError,
    configuration_fingerprint,
)

# Computed by the code on main immediately before ANC-4 and after OBS-2
# (cd1547a) for these exact configs. The revision matters: these literals were
# first frozen at 2370cee, before OBS-2 bumped STATE_SEMANTICS_VERSION from 1
# to 2 for its estimated-cost semantics. That bump deliberately changes every
# fingerprint, so ANC-4 -- branched before OBS-2 and merged after it -- carried
# literals describing a version that no longer existed, and both tests below
# failed on main without either pull request being wrong. Re-freezing them at
# cd1547a restores what they were always meant to assert: that *ANC-4* left
# non-proof fingerprints alone.
_PRE_ANC4_DEFAULT = "8e8d94244f0e3dc94e596c55ecbd2e1f957010c28549623790f8e95355433a65"
_PRE_ANC4_PROOF = "2aefccef7ddc2afa9e2cf88bcd445de4a2b6fa80514bc0ba4bbe4f3e2c7ef8d5"
_SECTION = {
    "enabled": True,
    "manifest_path": "proof_spec.json",
    "probe_command": ["python", "probe.py"],
}
_SPEC = {
    "schema_version": 1,
    "task_id": "task-1",
    "proof_id": "proof-1",
    "scope_id": "scope-1",
    "kind": "checkpoint_set",
    "required": ["done"],
}


def _proof_config() -> PathlayerConfig:
    return PathlayerConfig.from_mapping({"detectors": {"progress_proof": dict(_SECTION)}})


def _pre_anc4_find(event: ToolEvent, key: str) -> Any:
    """ANC-1 to ANC-3's lookup: the tool's own result first, then metadata."""
    result = event.result
    if isinstance(result, Mapping) and key in result:
        return result[key]
    return event.metadata.get(key)


def _no_progress(pipeline: PersistentPipeline) -> float:
    decision = pipeline.last_decision
    assert decision is not None
    (signal,) = [signal for signal in decision.signals if signal.name == "no_progress"]
    return signal.confidence


def test_only_proof_enabled_fingerprints_change() -> None:
    assert configuration_fingerprint(PathlayerConfig()) == _PRE_ANC4_DEFAULT
    disabled = PathlayerConfig.from_mapping(
        {"detectors": {"progress_proof": {**_SECTION, "enabled": False}}}
    )
    assert configuration_fingerprint(disabled) == _PRE_ANC4_DEFAULT
    assert configuration_fingerprint(_proof_config()) != _PRE_ANC4_PROOF


def test_resume_of_a_pre_anc4_zero_confidence_prefix_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # the config's relative manifest_path
    Path("proof_spec.json").write_text(json.dumps(_SPEC), encoding="utf-8")
    config = _proof_config()
    validated = validate_proof_spec(_SPEC)
    assert validated is not None
    first = ToolEvent(
        tool="report_progress",
        result={
            "proof_spec": _SPEC,
            "progress_proof": {
                "spec_hash": compute_spec_hash(validated),
                "sequence": 0,
                "checkpoints": {"done": False},
            },
        },
    )
    second = ToolEvent(tool="edit_file", args={"path": "a.py"})

    # A pre-ANC-4 runtime journals the first event under its own fingerprint.
    with monkeypatch.context() as old_runtime:
        old_runtime.setattr(proof_module, "_find", _pre_anc4_find)
        store = SQLiteStateStore(tmp_path / "state.db")
        before = PersistentPipeline(lambda: build_pipeline(config), store, "run", _PRE_ANC4_PROOF)
        before.process(first)
        assert _no_progress(before) == 0.0  # a baseline: nothing to replay against
        store.close()  # the process dies; the scope stays active
        uninterrupted = build_pipeline(config)
        uninterrupted.process(first)
        uninterrupted.process(second)
        decision = uninterrupted.last_decision
        assert decision is not None
        expected = [s.confidence for s in decision.signals if s.name == "no_progress"]
        assert expected == [pytest.approx(0.1)]
    shutil.copy(tmp_path / "state.db", tmp_path / "unguarded.db")

    # What the fingerprint now prevents: replay accepts the prefix, and the
    # next unchanged event silently decides differently.
    unguarded = SQLiteStateStore(tmp_path / "unguarded.db")
    resumed = PersistentPipeline(lambda: build_pipeline(config), unguarded, "run", _PRE_ANC4_PROOF)
    assert resumed.durable
    resumed.process(second)
    assert _no_progress(resumed) == 0.0
    unguarded.close()

    # This version: the proof-enabled scope is rejected explicitly instead.
    store = SQLiteStateStore(tmp_path / "state.db")
    with pytest.raises(StateCompatibilityError):
        PersistentPipeline(
            lambda: build_pipeline(config), store, "run", configuration_fingerprint(config)
        )
    store.close()


def test_pre_anc4_default_scope_still_resumes(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    before = PersistentPipeline(
        lambda: build_pipeline(PathlayerConfig()), store, "run", _PRE_ANC4_DEFAULT
    )
    before.process(ToolEvent(tool="read_file", args={"path": "a.py"}))
    store.close()

    store = SQLiteStateStore(tmp_path / "state.db")
    resumed = PersistentPipeline(
        lambda: build_pipeline(PathlayerConfig()),
        store,
        "run",
        configuration_fingerprint(PathlayerConfig()),
    )
    assert resumed.durable
    resumed.process(ToolEvent(tool="read_file", args={"path": "b.py"}))
    store.close()
