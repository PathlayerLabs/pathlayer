"""ANC-2: concrete producer contract and build/capture wiring.

Covers the trust boundary ANC-1 deliberately left open: loading/validating an
operator-supplied ``ProofSpec`` manifest (fail-fast, at build time), running an
executable probe (fail-*soft*, at capture time) to populate
``ToolEvent.metadata``, and the end-to-end path from a hand-written
manifest+probe through ``build_pipeline`` to a real ``no_progress`` signal.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_run_store import RunStore
from pathlayer.config import ConfigError, PathlayerConfig, build_pipeline
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.proof import ProofProgressAnchor
from pathlayer.detectors.proof_producer import (
    ManifestError,
    build_producer,
    load_proof_spec,
)
from pathlayer.events import EventKind, ToolEvent
from pathlayer.policies import Action


def _manifest(
    tmp_path: Path,
    *,
    name: str = "spec.json",
    required: tuple[str, ...] = ("done",),
    scope_id: str = "scope-1",
) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task-1",
                "proof_id": "accepted-deliverables",
                "scope_id": scope_id,
                "kind": "checkpoint_set",
                "required": list(required),
            }
        ),
        encoding="utf-8",
    )
    return path


def _probe_script(tmp_path: Path, body: str, *, name: str = "probe.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _static_probe(tmp_path: Path, checkpoints: dict[str, bool]) -> tuple[str, ...]:
    """A probe that always reports the same checkpoint states."""
    script = _probe_script(
        tmp_path,
        f"""
        import json, sys
        json.dump({{"checkpoints": {checkpoints!r}}}, sys.stdout)
        """,
    )
    return (sys.executable, str(script))


def _counting_probe(
    tmp_path: Path, counter_path: Path, checkpoints: dict[str, bool]
) -> tuple[str, ...]:
    """A probe that increments a counter file on every invocation."""
    script = _probe_script(
        tmp_path,
        f"""
        import json, sys
        path = {str(counter_path)!r}
        try:
            with open(path) as f:
                n = int(f.read() or "0")
        except FileNotFoundError:
            n = 0
        with open(path, "w") as f:
            f.write(str(n + 1))
        json.dump({{"checkpoints": {checkpoints!r}}}, sys.stdout)
        """,
    )
    return (sys.executable, str(script))


# --------------------------------------------------------------------------- #
# Manifest loading / validation (fail-fast, build time)
# --------------------------------------------------------------------------- #


def test_load_proof_spec_valid_manifest(tmp_path: Path) -> None:
    path = _manifest(tmp_path, required=("a", "b"))
    spec = load_proof_spec(path)
    assert spec.required == ("a", "b")
    assert spec.kind == "checkpoint_set"


def test_load_proof_spec_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="cannot read"):
        load_proof_spec(tmp_path / "missing.json")


def test_load_proof_spec_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_proof_spec(path)


def test_load_proof_spec_schema_violation_raises(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ManifestError, match="schema validation"):
        load_proof_spec(path)


def test_manifest_error_is_a_config_error(tmp_path: Path) -> None:
    assert issubclass(ManifestError, ConfigError)


def test_build_producer_end_to_end(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _static_probe(tmp_path, {"done": True})
    producer = build_producer(str(manifest_path), probe_command, 5.0)
    assert producer.spec.required == ("done",)
    assert producer.spec_hash.startswith("sha256:")


# --------------------------------------------------------------------------- #
# Probe invocation (fail-soft, capture time)
# --------------------------------------------------------------------------- #


def test_probe_success_increments_sequence(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _static_probe(tmp_path, {"done": True})
    producer = build_producer(str(manifest_path), probe_command, 5.0)

    first = producer.probe()
    second = producer.probe()
    assert first is not None and second is not None
    _, first_proof = first
    _, second_proof = second
    assert first_proof["sequence"] == 1
    assert second_proof["sequence"] == 2
    assert first_proof["checkpoints"] == {"done": True}


def test_probe_nonzero_exit_is_inert(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    script = _probe_script(tmp_path, "import sys; sys.exit(1)")
    producer = build_producer(str(manifest_path), (sys.executable, str(script)), 5.0)
    assert producer.probe() is None


def test_probe_timeout_is_inert(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    script = _probe_script(tmp_path, "import time; time.sleep(5)")
    producer = build_producer(str(manifest_path), (sys.executable, str(script)), 0.05)
    assert producer.probe() is None


def test_probe_malformed_stdout_is_inert(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    script = _probe_script(tmp_path, "print('not json')")
    producer = build_producer(str(manifest_path), (sys.executable, str(script)), 5.0)
    assert producer.probe() is None


def test_probe_missing_binary_is_inert(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    producer = build_producer(str(manifest_path), ("/no/such/binary-anc2",), 5.0)
    assert producer.probe() is None


def test_probe_non_utf8_stdout_is_inert(tmp_path: Path) -> None:
    """A probe writing undecodable bytes must degrade, not raise UnicodeDecodeError."""
    manifest_path = _manifest(tmp_path)
    script = _probe_script(
        tmp_path, "import sys; sys.stdout.buffer.write(b'\\xff\\xfe'); sys.stdout.buffer.flush()"
    )
    producer = build_producer(str(manifest_path), (sys.executable, str(script)), 5.0)
    assert producer.probe() is None


def test_probe_sequence_is_thread_safe(tmp_path: Path) -> None:
    """Concurrent runs sharing one producer must never race the counter."""
    import threading

    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _static_probe(tmp_path, {"done": True})
    producer = build_producer(str(manifest_path), probe_command, 5.0)

    sequences: list[int] = []
    lock = threading.Lock()

    def _call() -> None:
        result = producer.probe()
        assert result is not None
        _, progress_proof = result
        with lock:
            sequences.append(progress_proof["sequence"])

    threads = [threading.Thread(target=_call) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(sequences) == list(range(1, 21))


def test_probe_drops_checkpoints_outside_required(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _static_probe(tmp_path, {"done": True, "unrelated": True})
    producer = build_producer(str(manifest_path), probe_command, 5.0)
    result = producer.probe()
    assert result is not None
    _, progress_proof = result
    assert progress_proof["checkpoints"] == {"done": True}


def test_probe_rejects_whole_payload_on_one_malformed_checkpoint(tmp_path: Path) -> None:
    """A malformed entry invalidates the whole tick -- it must not be silently
    dropped while a well-formed sibling entry is still credited."""
    manifest_path = _manifest(tmp_path, required=("done", "broken"))
    script = _probe_script(
        tmp_path,
        'import json, sys; json.dump({"checkpoints": {"done": True, "broken": "yes"}}, sys.stdout)',
    )
    producer = build_producer(str(manifest_path), (sys.executable, str(script)), 5.0)
    assert producer.probe() is None


def test_attach_writes_metadata_not_result(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _static_probe(tmp_path, {"done": True})
    producer = build_producer(str(manifest_path), probe_command, 5.0)

    event = ToolEvent(tool="write_file", result={"ok": True})
    producer.attach(event)

    assert event.result == {"ok": True}
    assert "proof_spec" in event.metadata
    assert "progress_proof" in event.metadata
    assert event.metadata["progress_proof"]["checkpoints"] == {"done": True}


def test_attach_is_noop_on_probe_failure(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    script = _probe_script(tmp_path, "import sys; sys.exit(1)")
    producer = build_producer(str(manifest_path), (sys.executable, str(script)), 5.0)

    event = ToolEvent(tool="write_file")
    producer.attach(event)
    assert event.metadata == {}


# --------------------------------------------------------------------------- #
# End-to-end DoD scenario: manifest + probe through build_pipeline to a real
# no_progress signal, with no other progress evidence present.
# --------------------------------------------------------------------------- #


def test_end_to_end_stall_fires_no_progress(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _static_probe(tmp_path, {"done": False})

    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": str(manifest_path),
                    "probe_command": list(probe_command),
                }
            }
        }
    )
    pipeline = build_pipeline(config)
    no_progress = next(d for d in pipeline._detectors if isinstance(d, NoProgressDetector))
    assert any(isinstance(a, ProofProgressAnchor) for a in no_progress._anchors)

    producer = build_producer(str(manifest_path), probe_command, 5.0)
    action = Action.NONE
    for _ in range(30):
        if pipeline.stopped:
            break
        event = ToolEvent(tool="do_work", kind=EventKind.TOOL_CALL)
        producer.attach(event)
        action = pipeline.process(event)

    assert pipeline.stopped
    assert action == Action.STOP


def test_end_to_end_zero_config_stays_unchanged(tmp_path: Path) -> None:
    """A malformed/absent manifest never affects a run with the section disabled."""
    pipeline = build_pipeline(PathlayerConfig())
    no_progress = next(d for d in pipeline._detectors if isinstance(d, NoProgressDetector))
    assert not any(isinstance(a, ProofProgressAnchor) for a in no_progress._anchors)


def test_build_pipeline_fails_fast_on_malformed_manifest(tmp_path: Path) -> None:
    """build_pipeline itself validates the manifest -- not just OpenClawRunRecorder.

    Any caller that builds a pipeline directly (bypassing the recorder) must
    get the same fail-fast guarantee; otherwise a malformed manifest would
    only ever surface as a permanently-inert anchor, never an error.
    """
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text("not json", encoding="utf-8")
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": str(bad_manifest),
                    "probe_command": ["echo"],
                }
            }
        }
    )
    with pytest.raises(ConfigError, match="not valid JSON"):
        build_pipeline(config)


def test_openclaw_run_recorder_fails_fast_on_malformed_manifest(tmp_path: Path) -> None:
    """OpenClawRunRecorder also fails fast, via its own build_producer call --

    even when detectors.no_progress is disabled, so build_pipeline's own
    validation (nested under `if spec.no_progress.enabled`) never runs.
    """
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text("not json", encoding="utf-8")
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "no_progress": {"enabled": False},
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": str(bad_manifest),
                    "probe_command": ["echo"],
                },
            }
        }
    )
    build_pipeline(config)  # succeeds: no_progress disabled, nothing to validate here
    with pytest.raises(ManifestError):
        OpenClawRunRecorder(tmp_path / "runs", config=config)


# --------------------------------------------------------------------------- #
# Live-capture-only injection: never re-probed on restore/replay.
# --------------------------------------------------------------------------- #


def _ctx(run_id: str) -> dict[str, Any]:
    return {"runId": run_id, "sessionKey": "sess-1", "agentId": "agent-1"}


def _before(run_id: str) -> dict[str, Any]:
    return {
        "hook": "before_agent_run",
        "event": {"prompt": "fix the bug", "messages": []},
        "ctx": _ctx(run_id),
    }


def _tool(run_id: str, *, tool_call_id: str = "call-1") -> dict[str, Any]:
    return {
        "hook": "after_tool_call",
        "event": {
            "toolName": "read",
            "params": {},
            "result": "ok",
            "toolCallId": tool_call_id,
            "runId": run_id,
        },
        "ctx": _ctx(run_id),
    }


def test_restore_never_reinvokes_the_probe(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    counter_path = tmp_path / "counter"
    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _counting_probe(tmp_path, counter_path, {"done": True})

    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": str(manifest_path),
                    "probe_command": list(probe_command),
                }
            }
        }
    )

    recorder = OpenClawRunRecorder(runs_root, config=config)
    recorder.process_payload(_before("run-cut"))
    recorder.process_payload(_tool("run-cut"))

    assert counter_path.read_text(encoding="utf-8") == "1"

    restarted = OpenClawRunRecorder(runs_root, config=config)
    restarted.startup()

    run = RunStore(runs_root).resolve_run("run-cut")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    # The unfinished run's raw journal was replayed by startup(), but the probe
    # must not have been re-invoked: replaying against *current* world state
    # for a *historical* event would break replay determinism.
    assert counter_path.read_text(encoding="utf-8") == "1"
    # Restoring must not corrupt the durable SQLite state: the replayed event
    # (with no proof re-attached) must reconstruct byte-identical to what a
    # live PersistentPipeline already durably committed (with the proof it
    # actually saw), via the durable proof sidecar record -- not diverge into
    # a StateConcurrencyError that silently degrades the run to memory-only.
    assert manifest.get("persistence_error") is None
    assert manifest.get("state_durable") is not False


def test_live_proof_is_durably_journaled_for_replay(tmp_path: Path) -> None:
    """The exact live proof is durably recorded, correlated to its event.

    This is the mechanism the byte-identical-replay property depends on: a
    raw hook payload alone can never reproduce a live probe result, so
    ``_handle_tool`` must durably journal a sidecar record carrying it,
    keyed by the same ``event_id`` a later restore's replay will recompute --
    otherwise a resumed run's reconstructed event diverges from what a live
    ``PersistentPipeline`` already durably committed, and its duplicate-tick
    idempotency check raises ``StateConcurrencyError`` (see the ANC-2 PR
    review finding this test, and ``test_restore_never_reinvokes_the_probe``'s
    ``persistence_error`` assertion, lock in).
    """
    runs_root = tmp_path / "runs"
    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _static_probe(tmp_path, {"done": True})

    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": str(manifest_path),
                    "probe_command": list(probe_command),
                }
            }
        }
    )

    recorder = OpenClawRunRecorder(runs_root, config=config)
    recorder.process_payload(_before("run-cut"))
    recorder.process_payload(_tool("run-cut"))
    live_event = recorder._runs["run-cut"].events[0]
    assert live_event.metadata["progress_proof"] == {
        "spec_hash": live_event.metadata["progress_proof"]["spec_hash"],
        "sequence": 1,
        "checkpoints": {"done": True},
    }

    run = RunStore(runs_root).resolve_run("run-cut")
    assert run is not None
    records = [json.loads(line) for line in run.raw_events_path.read_text().splitlines()]
    proof_records = [r for r in records if r.get("hook") == "pathlayer_progress_proof"]
    assert len(proof_records) == 1
    sidecar = proof_records[0]["payload"]
    assert sidecar["for_event_id"] == "openclaw:2:0"
    assert sidecar["proof_spec"] == live_event.metadata["proof_spec"]
    assert sidecar["progress_proof"] == live_event.metadata["progress_proof"]


def test_restore_counts_proof_sidecar_when_manifest_lags(tmp_path: Path) -> None:
    """A crash before the manifest update must not reuse a sidecar sequence."""
    runs_root = tmp_path / "runs"
    manifest_path = _manifest(tmp_path, required=("done",))
    probe_command = _static_probe(tmp_path, {"done": True})
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": str(manifest_path),
                    "probe_command": list(probe_command),
                }
            }
        }
    )

    recorder = OpenClawRunRecorder(runs_root, config=config)
    recorder.process_payload(_before("run-cut"))
    recorder.process_payload(_tool("run-cut", tool_call_id="call-1"))

    run = RunStore(runs_root).resolve_run("run-cut")
    assert run is not None
    # Simulate the exact crash window: the raw tool + proof sidecar and SQLite
    # tick are durable, but the final manifest event_count update did not land.
    run.update_manifest(event_count=1)

    restarted = OpenClawRunRecorder(runs_root, config=config)
    restarted.process_payload(_tool("run-cut", tool_call_id="call-2"))

    records = [json.loads(line) for line in run.raw_events_path.read_text().splitlines()]
    assert [record["sequence"] for record in records] == list(range(1, len(records) + 1))
    assert run.read_manifest()["event_count"] == len(records)
