"""Tests for OpenClaw per-run recording (lifecycle + durable run directories)."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import stat
import sys
import tarfile
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import pathlayer.adapters.openclaw_recorder as recorder_module
from pathlayer.adapters.openclaw import translate_openclaw_event
from pathlayer.adapters.openclaw_export import export_run
from pathlayer.adapters.openclaw_lifecycle import (
    extract_openclaw_run_id,
    sanitize_run_id_for_path,
)
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_run_store import RunStore, atomic_write_json
from pathlayer.config import (
    ConfigError,
    DetectorsConfig,
    LoopConfig,
    PathlayerConfig,
    PipelineSection,
    PolicySection,
)
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action
from pathlayer.store import SQLiteStateStore, StateWriteError, configuration_fingerprint
from pathlayer.validation.replay import replay_trace
from pathlayer.validation.traces import (
    UNKNOWN_LABEL,
    Trace,
    load_trace,
    trace_to_dict,
)

_RECORD_LISTENER = (
    Path(__file__).resolve().parents[1] / "examples" / "openclaw_plugin" / "record.py"
)

# A run of >= 5 identical tool calls reaches loop confidence
# 1 - 0.5**(reps - k + 1) >= 0.9375, above the default stop_threshold (0.9), so
# the default pipeline STOPs -- no test double needed.
_STOP_REPEATS = 6


def _ctx(run_id: str = "run-a", session: str = "sess-1", agent: str = "agent-1") -> dict[str, Any]:
    return {"runId": run_id, "sessionKey": session, "agentId": agent}


def _before(run_id: str = "run-a", prompt: str = "fix the bug") -> dict[str, Any]:
    return {
        "hook": "before_agent_run",
        "event": {"prompt": prompt, "messages": [{"role": "user", "content": prompt}]},
        "ctx": _ctx(run_id),
    }


def _tool(
    tool: str,
    *,
    run_id: str = "run-a",
    tool_call_id: str = "call-1",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "hook": "after_tool_call",
        "event": {
            "toolName": tool,
            "params": params or {},
            "result": "ok",
            "toolCallId": tool_call_id,
            "runId": run_id,
        },
        "ctx": _ctx(run_id),
    }


def _end(run_id: str = "run-a", *, success: bool = True) -> dict[str, Any]:
    return {
        "hook": "agent_end",
        "event": {"success": success, "durationMs": 1200, "messages": [{"role": "assistant"}]},
        "ctx": _ctx(run_id),
    }


def test_ctx_run_id_is_canonical() -> None:
    payload = _tool("read", run_id="event-run", tool_call_id="call-9")
    payload["ctx"]["runId"] = "ctx-run"
    assert extract_openclaw_run_id(payload) == "ctx-run"
    events = translate_openclaw_event(payload)
    assert events[0].run_id == "ctx-run"
    assert events[0].metadata["tool_call_id"] == "call-9"


def test_tool_call_id_is_never_run_id() -> None:
    payload = {
        "hook": "after_tool_call",
        "event": {"toolName": "x", "params": {}, "toolCallId": "call-9"},
        "ctx": {},
    }
    assert extract_openclaw_run_id(payload) is None
    events = translate_openclaw_event(payload)
    assert events[0].run_id is None
    assert events[0].metadata["tool_call_id"] == "call-9"


def test_event_run_id_used_when_ctx_missing() -> None:
    payload = {
        "hook": "after_tool_call",
        "event": {"toolName": "x", "params": {}, "runId": "event-run"},
        "ctx": {},
    }
    assert extract_openclaw_run_id(payload) == "event-run"


def test_two_interleaved_runs_stay_separated(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-a"))
    recorder.process_payload(_before("run-b"))
    recorder.process_payload(_tool("read", run_id="run-a", tool_call_id="a1"))
    recorder.process_payload(_tool("edit", run_id="run-b", tool_call_id="b1"))
    recorder.process_payload(_end("run-a"))
    recorder.process_payload(_end("run-b"))

    store = RunStore(tmp_path)
    run_a = store.resolve_run("run-a")
    run_b = store.resolve_run("run-b")
    assert run_a is not None and run_b is not None
    trace_a = load_trace(run_a.trace_path)
    trace_b = load_trace(run_b.trace_path)
    assert [event.tool for event in trace_a.events] == ["read"]
    assert [event.tool for event in trace_b.events] == ["edit"]


def test_lifecycle_start_tool_end(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-1", prompt="hello"))
    recorder.process_payload(_tool("grep", run_id="run-1", tool_call_id="c1"))
    recorder.process_payload(_end("run-1"))

    run = RunStore(tmp_path).resolve_run("run-1")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["event_count"] == 3
    request = json.loads(run.request_path.read_text(encoding="utf-8"))
    assert request["prompt"] == "hello"
    assert run.raw_events_path.read_text(encoding="utf-8").count("\n") == 3
    trace = load_trace(run.trace_path)
    assert len(trace.events) == 1
    final = json.loads(run.final_path.read_text(encoding="utf-8"))
    assert final["success"] is True


def test_orphan_tool_before_start(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(
        {
            "hook": "after_tool_call",
            "event": {"toolName": "read", "params": {}, "toolCallId": "call-x"},
            "ctx": {"sessionKey": "sess-orphan", "runId": "run-orphan"},
        }
    )
    runs = RunStore(tmp_path).list_runs()
    assert len(runs) == 1
    manifest = runs[0].read_manifest()
    assert manifest["run_id"] == "run-orphan"
    assert manifest["status"] == "orphan"


def test_atomic_manifest_write(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"status": "started", "n": 1})
    atomic_write_json(path, {"status": "completed", "n": 2})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "completed"


def test_trace_json_is_replayable(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-replay"))
    for _ in range(4):
        recorder.process_payload(_tool("read", run_id="run-replay", tool_call_id="c"))
    recorder.process_payload(_end("run-replay"))

    run = RunStore(tmp_path).resolve_run("run-replay")
    assert run is not None
    report = replay_trace(load_trace(run.trace_path))
    assert report.events_processed >= 1


def test_export_one_run_only(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-export"))
    recorder.process_payload(_tool("read", run_id="run-export"))
    recorder.process_payload(_end("run-export"))

    archive = tmp_path / "out" / "run.tar.gz"
    export_run("run-export", archive, runs_root=tmp_path)
    assert archive.is_file()
    checksum = json.loads(Path(f"{archive}.sha256.json").read_text(encoding="utf-8"))
    assert checksum["run_id"] == "run-export"
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
    assert any(name.endswith("trace.json") for name in names)


def test_export_refuses_unknown_run_id(tmp_path: Path) -> None:
    with pytest.raises(LookupError, match="no recorded run"):
        export_run("missing-run", tmp_path / "x.tar.gz", runs_root=tmp_path)


def test_export_refuses_ambiguous_run_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    for suffix in ("a", "b"):
        run_id = "same-id"
        manifest = {"schema_version": 1, "run_id": run_id, "status": "started"}
        directory = store.open_or_create(f"{run_id}-{suffix}", manifest=manifest)
        atomic_write_json(
            directory.manifest_path,
            {**manifest, "run_id": run_id, "dir_name": directory.dir_name},
        )
    with pytest.raises(LookupError, match="ambiguous"):
        export_run("same-id", tmp_path / "x.tar.gz", runs_root=tmp_path)


@pytest.mark.parametrize("bad_id", ["", "..", "../etc/passwd", "a/b"])
def test_unsafe_run_ids_rejected(bad_id: str) -> None:
    with pytest.raises(ValueError):
        sanitize_run_id_for_path(bad_id)


def test_listener_restart_marks_unfinished_runs(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    directory = store.open_or_create(
        "run-interrupted",
        manifest={
            "schema_version": 1,
            "run_id": "run-interrupted",
            "status": "started",
            "event_count": 1,
        },
    )
    directory.update_manifest(status="started")

    recorder = OpenClawRunRecorder(tmp_path)
    recorder.startup()

    manifest = directory.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["ended_at"] is not None


def test_listener_restart_preserves_completed_runs(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-done"))
    recorder.process_payload(_end("run-done"))
    run = RunStore(tmp_path).resolve_run("run-done")
    assert run is not None
    ended_at = run.read_manifest()["ended_at"]

    recorder2 = OpenClawRunRecorder(tmp_path)
    recorder2.startup()
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["ended_at"] == ended_at


def test_recorder_http_ingest_end_to_end(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.startup()
    adapter = recorder.attach_http(port=0)
    try:
        body = json.dumps({"source": "openclaw", "payload": _before("run-http")}).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{adapter.port}/ingest",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
    finally:
        adapter.detach()
    assert RunStore(tmp_path).resolve_run("run-http") is not None


def _load_listener() -> ModuleType:
    """Load the recording listener example by path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("openclaw_record_listener", _RECORD_LISTENER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(run: Any) -> dict[str, str]:
    """Byte-for-byte content of every file of a run directory, keyed by name."""
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(run.root.iterdir())
        if path.is_file()
    }


def _decisions(run: Any) -> list[dict[str, Any]]:
    lines = run.decisions_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def _looping_payloads(run_id: str, count: int = _STOP_REPEATS) -> list[dict[str, Any]]:
    return [
        _tool("read", run_id=run_id, tool_call_id=f"c{i}", params={"path": "a.py"})
        for i in range(count)
    ]


def test_second_agent_end_changes_nothing(tmp_path: Path) -> None:
    # Regression: the duplicate used to reopen the finished run, restart the
    # sequence at 1 and overwrite trace.json with an empty trace.
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-dup"))
    for name in ("read", "edit", "test"):
        recorder.process_payload(_tool(name, run_id="run-dup", tool_call_id=name))
    recorder.process_payload(_end("run-dup"))

    run = RunStore(tmp_path).resolve_run("run-dup")
    assert run is not None
    before = _snapshot(run)
    mtimes = {path.name: path.stat().st_mtime_ns for path in run.root.iterdir() if path.is_file()}
    sequence_before = run.read_manifest()["event_count"]

    recorder.process_payload(_end("run-dup"))

    assert _snapshot(run) == before
    assert {p.name: p.stat().st_mtime_ns for p in run.root.iterdir() if p.is_file()} == mtimes
    manifest = run.read_manifest()
    assert manifest["event_count"] == sequence_before
    assert len(load_trace(run.trace_path).events) == 3


def test_late_tool_call_never_reopens_a_finished_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-late"))
    recorder.process_payload(_tool("read", run_id="run-late"))
    recorder.process_payload(_end("run-late"))

    run = RunStore(tmp_path).resolve_run("run-late")
    assert run is not None
    before = _snapshot(run)

    recorder.process_payload(_tool("edit", run_id="run-late", tool_call_id="late"))

    assert _snapshot(run) == before


def test_agent_end_after_restart_does_not_resurrect_interrupted_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-cut"))
    recorder.process_payload(_tool("read", run_id="run-cut"))

    restarted = OpenClawRunRecorder(tmp_path)
    restarted.startup()
    run = RunStore(tmp_path).resolve_run("run-cut")
    assert run is not None
    assert run.read_manifest()["status"] == "interrupted"
    before = _snapshot(run)

    restarted.process_payload(_end("run-cut"))

    assert _snapshot(run) == before
    assert not run.trace_path.exists()


def test_trace_json_goes_through_the_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[str] = []
    real_replace = os.replace

    def _spy(src: Any, dst: Any) -> None:
        replaced.append(Path(dst).name)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _spy)
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-atomic"))
    recorder.process_payload(_tool("read", run_id="run-atomic"))
    recorder.process_payload(_end("run-atomic"))

    assert "trace.json" in replaced


def test_failed_write_keeps_the_previous_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trace.json"
    atomic_write_json(path, trace_to_dict(Trace(label="healthy", events=[ToolEvent(tool="read")])))
    before = path.read_text(encoding="utf-8")

    def _fail(src: Any, dst: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", _fail)
    with pytest.raises(OSError, match="no space left"):
        atomic_write_json(path, trace_to_dict(Trace(label="loop")))

    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["trace.json"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not meaningful on Windows")
def test_exported_files_are_owner_only_whatever_the_umask(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-perm"))
    recorder.process_payload(_tool("read", run_id="run-perm"))
    recorder.process_payload(_end("run-perm"))

    archive = tmp_path / "out" / "run.tar.gz"
    checksums = Path(f"{archive}.sha256.json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    # A world-readable leftover from a previous export must not survive either.
    archive.write_bytes(b"stale")
    archive.chmod(0o644)
    previous_umask = os.umask(0o000)
    try:
        export_run("run-perm", archive, runs_root=tmp_path)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    assert stat.S_IMODE(checksums.stat().st_mode) == 0o600
    run = RunStore(tmp_path).resolve_run("run-perm")
    assert run is not None
    assert stat.S_IMODE(run.trace_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(run.manifest_path.stat().st_mode) == 0o600


def test_finished_run_is_not_healthy_without_a_verdict(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-nothing"))
    recorder.process_payload(_end("run-nothing"))

    run = RunStore(tmp_path).resolve_run("run-nothing")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["health"] == UNKNOWN_LABEL
    assert manifest["health_reason"] == "no tool call was analyzed"
    assert load_trace(run.trace_path).label == UNKNOWN_LABEL


def test_unsuccessful_run_is_not_healthy(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-failed"))
    recorder.process_payload(_tool("read", run_id="run-failed"))
    recorder.process_payload(_end("run-failed", success=False))

    run = RunStore(tmp_path).resolve_run("run-failed")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["health"] == UNKNOWN_LABEL
    assert "did not report success" in manifest["health_reason"]


def test_stopped_run_remains_unlabelled_until_human_curation(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-loop-stop"))
    for payload in _looping_payloads("run-loop-stop"):
        recorder.process_payload(payload)
    recorder.process_payload(_end("run-loop-stop"))

    run = RunStore(tmp_path).resolve_run("run-loop-stop")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["stopped"] is True
    assert manifest["health"] == UNKNOWN_LABEL
    assert manifest["health_reason"].startswith("pathlayer stopped: loop")
    assert load_trace(run.trace_path).label == UNKNOWN_LABEL


def test_interrupted_run_health_stays_unknown(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-killed"))
    recorder.process_payload(_tool("read", run_id="run-killed"))

    OpenClawRunRecorder(tmp_path).startup()

    run = RunStore(tmp_path).resolve_run("run-killed")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["health"] == UNKNOWN_LABEL


def test_clean_live_run_remains_unlabelled_until_human_curation(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-clean"))
    for name, path in (("read", "a.py"), ("edit", "b.py"), ("test", "c.py")):
        recorder.process_payload(
            _tool(name, run_id="run-clean", tool_call_id=name, params={"path": path})
        )
    recorder.process_payload(_end("run-clean", success=True))

    run = RunStore(tmp_path).resolve_run("run-clean")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["health"] == UNKNOWN_LABEL
    assert manifest["health_reason"] == "live recording requires human curation"
    assert manifest["stopped"] is False
    assert load_trace(run.trace_path).label == UNKNOWN_LABEL


def test_events_after_a_passive_stop_are_still_recorded(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-after-stop"))
    for payload in _looping_payloads("run-after-stop"):
        recorder.process_payload(payload)
    tail = ("edit", "run_tests", "commit")
    for name in tail:
        recorder.process_payload(
            _tool(name, run_id="run-after-stop", tool_call_id=name, params={"path": name})
        )
    recorder.process_payload(_end("run-after-stop"))

    run = RunStore(tmp_path).resolve_run("run-after-stop")
    assert run is not None
    trace = load_trace(run.trace_path)
    assert [event.tool for event in trace.events] == ["read"] * _STOP_REPEATS + list(tail)

    decisions = _decisions(run)
    # One decision per recorded event, in order, so trace.json and
    # decisions.jsonl stay aligned across the STOP.
    assert [decision["event_index"] for decision in decisions] == list(range(len(trace.events)))
    first_stop = next(i for i, d in enumerate(decisions) if d["action"] == "STOP")
    assert decisions[first_stop]["analyzed"] is True
    assert decisions[first_stop]["reason"].startswith("loop")
    assert all(not d["analyzed"] for d in decisions[first_stop + 1 :])
    assert run.read_manifest()["event_count"] == len(decisions) + 2  # + before_agent_run, agent_end


def test_intervention_below_stop_is_not_healthy(tmp_path: Path) -> None:
    # A WARN tier active from the first tick: an action was emitted, so the run
    # is not claimed healthy even though it completed successfully.
    config = PathlayerConfig(policy=PolicySection(tier_thresholds={"warn": 0.0}))
    recorder = OpenClawRunRecorder(tmp_path, config=config)
    recorder.process_payload(_before("run-warned"))
    recorder.process_payload(_tool("read", run_id="run-warned"))
    recorder.process_payload(_end("run-warned", success=True))

    run = RunStore(tmp_path).resolve_run("run-warned")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["stopped"] is False
    assert manifest["health"] == UNKNOWN_LABEL
    assert manifest["health_reason"] == "pathlayer intervened below STOP: WARN"


def test_dry_run_would_be_action_is_not_healthy(tmp_path: Path) -> None:
    # Dry-run emits nothing, so only ``would_be_actions`` shows the run was not
    # clean -- claiming healthy here would hide exactly what dry-run is for.
    config = PathlayerConfig(
        pipeline=PipelineSection(dry_run=True),
        policy=PolicySection(stop_threshold=0.0),
    )
    recorder = OpenClawRunRecorder(tmp_path, config=config)
    recorder.process_payload(_before("run-dry"))
    recorder.process_payload(_tool("read", run_id="run-dry"))
    recorder.process_payload(_end("run-dry", success=True))

    run = RunStore(tmp_path).resolve_run("run-dry")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["stopped"] is False
    assert manifest["health"] == UNKNOWN_LABEL
    assert manifest["health_reason"] == "dry-run would have acted: STOP"


def test_unfinished_run_is_resumed_without_restarting_its_sequence(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-resume"))
    recorder.process_payload(_tool("read", run_id="run-resume", tool_call_id="first"))
    run = RunStore(tmp_path).resolve_run("run-resume")
    assert run is not None
    started_at = run.read_manifest()["started_at"]

    # A new listener that never ran the startup sweep still finds the directory
    # in a non-terminal state: it must continue the run, not restart it.
    resumed = OpenClawRunRecorder(tmp_path)
    resumed.process_payload(_tool("edit", run_id="run-resume", tool_call_id="second"))
    resumed.process_payload(_end("run-resume"))

    manifest = run.read_manifest()
    assert manifest["started_at"] == started_at
    assert manifest["event_count"] == 4
    sequences = [
        json.loads(line)["sequence"]
        for line in run.raw_events_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert sequences == [1, 2, 3, 4]
    trace = load_trace(run.trace_path)
    assert [event.tool for event in trace.events] == ["read", "edit"]
    assert [event.ts for event in trace.events] == [0.0, 1.0]
    assert [decision["event_index"] for decision in _decisions(run)] == [0, 1]


def test_unfinished_run_resumes_sqlite_without_double_counting_raw_replay(
    tmp_path: Path,
) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-sqlite-resume"))
    first.process_payload(_tool("read", run_id="run-sqlite-resume", tool_call_id="first"))

    resumed = OpenClawRunRecorder(tmp_path)
    resumed.process_payload(_tool("edit", run_id="run-sqlite-resume", tool_call_id="second"))

    connection = sqlite3.connect(tmp_path / "state.db")
    rows = connection.execute(
        "SELECT event_index, event_id FROM events WHERE state_key=? ORDER BY event_index",
        ("run-sqlite-resume",),
    ).fetchall()
    connection.close()

    assert rows == [(0, "openclaw:2:0"), (1, "openclaw:3:0")]


def test_reattach_store_failure_rebuilds_a_fresh_memory_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-replay-memory"))
    first.process_payload(_tool("read", run_id="run-replay-memory", tool_call_id="first"))
    first.process_payload(_tool("edit", run_id="run-replay-memory", tool_call_id="second"))

    restarted = OpenClawRunRecorder(tmp_path)
    assert restarted._state_store is not None
    original_lookup = restarted._state_store.lookup_event
    calls = 0

    def fail_second_lookup(state_key: str, event_id: str) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StateWriteError("SQLite unavailable during raw replay")
        return original_lookup(state_key, event_id)

    monkeypatch.setattr(restarted._state_store, "lookup_event", fail_second_lookup)
    restarted.process_payload(_tool("test", run_id="run-replay-memory", tool_call_id="third"))

    active = restarted._runs["run-replay-memory"]
    assert not isinstance(active.pipeline, recorder_module.PersistentPipeline)
    assert [event.tool for event in active.pipeline.history] == ["read", "edit", "test"]
    manifest = active.directory.read_manifest()
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateWriteError"


def test_reattach_translation_divergence_interrupts_and_poisons_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-translation-drift"))
    first.process_payload(_tool("read", run_id="run-translation-drift", tool_call_id="first"))
    run = RunStore(tmp_path).resolve_run("run-translation-drift")
    assert run is not None
    raw_before = run.raw_events_path.read_text(encoding="utf-8")
    original_translate = recorder_module.translate_openclaw_event

    def changed_translation(payload: dict[str, Any]) -> list[ToolEvent]:
        events = original_translate(payload)
        for event in events:
            event.tool = f"changed-{event.tool}"
        return events

    monkeypatch.setattr(recorder_module, "translate_openclaw_event", changed_translation)
    restarted = OpenClawRunRecorder(tmp_path)
    restarted.process_payload(_tool("edit", run_id="run-translation-drift", tool_call_id="second"))

    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateConcurrencyError"
    assert run.raw_events_path.read_text(encoding="utf-8") == raw_before
    assert restarted._state_store is not None
    snapshot = restarted._state_store.read_scope("run-translation-drift")
    assert not snapshot.durable


def test_reattach_corrupt_state_interrupts_instead_of_falling_back_to_memory(
    tmp_path: Path,
) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-corrupt-state"))
    first.process_payload(_tool("read", run_id="run-corrupt-state", tool_call_id="first"))
    run = RunStore(tmp_path).resolve_run("run-corrupt-state")
    assert run is not None
    raw_before = run.raw_events_path.read_text(encoding="utf-8")

    connection = sqlite3.connect(tmp_path / "state.db")
    connection.execute(
        "UPDATE ticks SET fused = 2 WHERE state_key = ?",
        ("run-corrupt-state",),
    )
    connection.commit()
    connection.close()

    restarted = OpenClawRunRecorder(tmp_path)
    restarted.process_payload(_tool("edit", run_id="run-corrupt-state", tool_call_id="second"))

    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateCorruptionError"
    assert run.raw_events_path.read_text(encoding="utf-8") == raw_before
    assert "run-corrupt-state" not in restarted._runs


def test_new_raw_directory_does_not_reopen_a_nondurable_sqlite_scope(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    fingerprint = configuration_fingerprint(PathlayerConfig())
    store.load_or_create("orphan-scope", fingerprint)
    store.mark_nondurable("orphan-scope", fingerprint)
    store.close()

    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("orphan-scope"))

    run = RunStore(tmp_path).resolve_run("orphan-scope")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateIncompleteError"
    assert "orphan-scope" not in recorder._runs


def test_sqlite_unavailable_at_startup_keeps_raw_recording_and_memory_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise StateWriteError("read-only filesystem")

    monkeypatch.setattr(recorder_module, "SQLiteStateStore", unavailable)
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-memory"))
    recorder.process_payload(_tool("read", run_id="run-memory"))
    recorder.process_payload(_end("run-memory"))

    run = RunStore(tmp_path).resolve_run("run-memory")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateWriteError"
    assert run.raw_events_path.is_file()
    assert len(load_trace(run.trace_path).events) == 1
    assert not (tmp_path / "state.db").exists()


def test_sqlite_unavailable_during_startup_marks_existing_run_non_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-startup-memory"))
    first.process_payload(_tool("read", run_id="run-startup-memory"))

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise StateWriteError("read-only filesystem")

    monkeypatch.setattr(recorder_module, "SQLiteStateStore", unavailable)
    restarted = OpenClawRunRecorder(tmp_path)
    restarted.startup()

    run = RunStore(tmp_path).resolve_run("run-startup-memory")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateWriteError"


def test_runtime_sqlite_failure_marks_manifest_and_keeps_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-degraded"))
    assert recorder._state_store is not None

    def fail_append(*args: Any, **kwargs: Any) -> Any:
        raise StateWriteError("disk full")

    monkeypatch.setattr(recorder._state_store, "append_tick", fail_append)
    recorder.process_payload(_tool("read", run_id="run-degraded"))

    run = RunStore(tmp_path).resolve_run("run-degraded")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateWriteError"
    assert len(_decisions(run)) == 1
    raw_before_restart = run.raw_events_path.read_text(encoding="utf-8")

    restarted = OpenClawRunRecorder(tmp_path)
    restarted.process_payload(_tool("edit", run_id="run-degraded", tool_call_id="second"))
    connection = sqlite3.connect(tmp_path / "state.db")
    durable_events = connection.execute(
        "SELECT count(*) FROM events WHERE state_key=?", ("run-degraded",)
    ).fetchone()[0]
    connection.close()
    assert durable_events == 0
    assert run.read_manifest()["status"] == "interrupted"
    assert len(_decisions(run)) == 1
    assert run.raw_events_path.read_text(encoding="utf-8") == raw_before_restart


def test_live_event_identity_collision_interrupts_both_listeners(tmp_path: Path) -> None:
    first = OpenClawRunRecorder(tmp_path)
    second = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-collision"))
    second.process_payload(_before("run-collision"))
    # Model two live listeners that both allocated the next sequence before
    # either observed the other's raw append.
    second._runs["run-collision"].sequence = 1
    first.process_payload(_tool("read", run_id="run-collision", tool_call_id="first"))
    second.process_payload(_tool("edit", run_id="run-collision", tool_call_id="second"))

    run = RunStore(tmp_path).resolve_run("run-collision")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateConcurrencyError"
    raw_after_collision = run.raw_events_path.read_text(encoding="utf-8")
    assert len(raw_after_collision.splitlines()) == 4
    assert "run-collision" not in second._runs

    second.process_payload(_tool("later", run_id="run-collision"))
    first.process_payload(_tool("later", run_id="run-collision"))
    assert run.raw_events_path.read_text(encoding="utf-8") == raw_after_collision
    assert "run-collision" not in first._runs


def test_agent_end_close_failure_still_completes_non_durable_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-close-failure"))
    recorder.process_payload(_tool("read", run_id="run-close-failure"))
    active = recorder._runs["run-close-failure"]
    assert isinstance(active.pipeline, recorder_module.PersistentPipeline)

    def fail_close(_status: str) -> None:
        raise StateWriteError("scope was closed concurrently")

    monkeypatch.setattr(active.pipeline, "close", fail_close)
    recorder.process_payload(_end("run-close-failure"))

    manifest = active.directory.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateWriteError"
    assert active.directory.final_path.is_file()
    assert active.directory.trace_path.is_file()
    assert "run-close-failure" not in recorder._runs


def test_incompatible_config_interrupts_existing_run_without_reusing_scope(
    tmp_path: Path,
) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-incompatible"))
    first.process_payload(_tool("read", run_id="run-incompatible"))
    run = RunStore(tmp_path).resolve_run("run-incompatible")
    assert run is not None
    lines_before = run.raw_events_path.read_text(encoding="utf-8").splitlines()

    changed = PathlayerConfig(policy=PolicySection(stop_threshold=0.5))
    restarted = OpenClawRunRecorder(tmp_path, config=changed)
    restarted.process_payload(_tool("edit", run_id="run-incompatible"))

    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateCompatibilityError"
    assert run.raw_events_path.read_text(encoding="utf-8").splitlines() == lines_before


def test_startup_interrupt_sweep_runs_even_if_raw_replay_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-broken-replay"))
    first.process_payload(_tool("read", run_id="run-broken-replay"))
    run = RunStore(tmp_path).resolve_run("run-broken-replay")
    assert run is not None

    restarted = OpenClawRunRecorder(tmp_path)

    def fail_replay(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("bad raw journal")

    monkeypatch.setattr(restarted, "_restore_recorded_events", fail_replay)
    with pytest.raises(ValueError, match="bad raw journal"):
        restarted.startup()
    assert run.read_manifest()["status"] == "interrupted"


def test_startup_marks_sqlite_scope_interrupted(tmp_path: Path) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_before("run-sqlite-interrupted"))
    first.process_payload(_tool("read", run_id="run-sqlite-interrupted"))

    restarted = OpenClawRunRecorder(tmp_path)
    restarted.startup()

    connection = sqlite3.connect(tmp_path / "state.db")
    status = connection.execute(
        "SELECT status FROM scopes WHERE state_key=?", ("run-sqlite-interrupted",)
    ).fetchone()[0]
    connection.close()
    assert status == "interrupted"


def test_startup_reconciles_manifest_when_sqlite_was_closed_first(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-close-crash"))
    recorder.process_payload(_tool("read", run_id="run-close-crash"))
    recorder.process_payload(_end("run-close-crash"))
    run = RunStore(tmp_path).resolve_run("run-close-crash")
    assert run is not None
    run.update_manifest(status="started", ended_at=None)

    OpenClawRunRecorder(tmp_path).startup()

    assert run.read_manifest()["status"] == "completed"
    assert run.read_manifest()["ended_at"] is not None


def test_custom_pipeline_factory_keeps_sqlite_opt_out(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, pipeline_factory=lambda: Pipeline([]))
    recorder.process_payload(_before("custom-no-sqlite"))
    recorder.process_payload(_tool("read", run_id="custom-no-sqlite"))

    assert not (tmp_path / "state.db").exists()


def test_custom_pipeline_factory_can_keep_legacy_process_signature(tmp_path: Path) -> None:
    class _LegacyPipeline(Pipeline):
        def process(self, event: ToolEvent) -> Action:
            return super().process(event)

    recorder = OpenClawRunRecorder(tmp_path, pipeline_factory=lambda: _LegacyPipeline([]))
    recorder.process_payload(_before("custom-legacy"))
    recorder.process_payload(_tool("read", run_id="custom-legacy"))

    run = RunStore(tmp_path).resolve_run("custom-legacy")
    assert run is not None
    assert len(_decisions(run)) == 1


def test_tool_event_timestamps_follow_per_run_arrival_order(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-ts"))
    for name in ("read", "edit", "test"):
        recorder.process_payload(_tool(name, run_id="run-ts", tool_call_id=name))
    recorder.process_payload(_end("run-ts"))

    run = RunStore(tmp_path).resolve_run("run-ts")
    assert run is not None
    assert [event.ts for event in load_trace(run.trace_path).events] == [0.0, 1.0, 2.0]


def test_zero_duration_is_preserved(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-zero-duration"))
    payload = _end("run-zero-duration")
    payload["event"]["durationMs"] = 0
    recorder.process_payload(payload)

    run = RunStore(tmp_path).resolve_run("run-zero-duration")
    assert run is not None
    final = json.loads(run.final_path.read_text(encoding="utf-8"))
    assert final["duration_ms"] == 0


def test_resumed_run_with_unusable_counter_restarts_from_zero(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    directory = store.open_or_create(
        "run-corrupt",
        manifest={"schema_version": 1, "run_id": "run-corrupt", "status": "started"},
    )
    atomic_write_json(
        directory.manifest_path,
        {
            "schema_version": 1,
            "run_id": "run-corrupt",
            "dir_name": directory.dir_name,
            "status": "started",
            "event_count": "not a number",
        },
    )

    OpenClawRunRecorder(tmp_path).process_payload(_tool("read", run_id="run-corrupt"))

    assert directory.read_manifest()["event_count"] == 1


def test_recorder_uses_builtin_defaults_without_a_config(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-default"))
    recorder.process_payload(_tool("read", run_id="run-default"))
    recorder.process_payload(_end("run-default"))

    run = RunStore(tmp_path).resolve_run("run-default")
    assert run is not None
    assert run.read_manifest()["stopped"] is False


def test_custom_config_value_changes_recorded_behaviour(tmp_path: Path) -> None:
    # stop_threshold 0.0 means "act on the first tick"; the default 0.9 does not
    # stop on a single read, so any difference here comes from the config.
    config = PathlayerConfig(policy=PolicySection(stop_threshold=0.0))
    recorder = OpenClawRunRecorder(tmp_path, config=config)
    recorder.process_payload(_before("run-configured"))
    recorder.process_payload(_tool("read", run_id="run-configured"))
    recorder.process_payload(_end("run-configured"))

    run = RunStore(tmp_path).resolve_run("run-configured")
    assert run is not None
    manifest = run.read_manifest()
    assert manifest["stopped"] is True
    assert manifest["health"] == UNKNOWN_LABEL


def test_invalid_config_value_fails_at_construction(tmp_path: Path) -> None:
    # k=1 is rejected by LoopDetector itself: the recorder builds a pipeline
    # eagerly so this surfaces at startup, not on the first forwarded hook.
    config = PathlayerConfig(detectors=DetectorsConfig(loop=LoopConfig(k=1)))
    with pytest.raises(ConfigError):
        OpenClawRunRecorder(tmp_path, config=config)


def test_config_and_pipeline_factory_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not both"):
        OpenClawRunRecorder(
            tmp_path, config=PathlayerConfig(), pipeline_factory=lambda: Pipeline([])
        )


def test_listener_without_config_keeps_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listener = _load_listener()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PATHLAYER_CONFIG", raising=False)
    runs_root = tmp_path / "runs"

    recorder = listener.build_recorder(listener.parse_args(["--runs-root", str(runs_root)]))
    recorder.process_payload(_before("run-listener"))
    recorder.process_payload(_tool("read", run_id="run-listener"))
    recorder.process_payload(_end("run-listener"))

    run = RunStore(runs_root).resolve_run("run-listener")
    assert run is not None
    assert run.read_manifest()["stopped"] is False


def test_listener_discovers_and_applies_a_custom_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listener = _load_listener()
    config_dir = tmp_path / ".pathlayer"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"policy": {"stop_threshold": 0.0}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PATHLAYER_CONFIG", raising=False)
    runs_root = tmp_path / "runs"

    recorder = listener.build_recorder(listener.parse_args(["--runs-root", str(runs_root)]))
    recorder.process_payload(_before("run-discovered"))
    recorder.process_payload(_tool("read", run_id="run-discovered"))
    recorder.process_payload(_end("run-discovered"))

    run = RunStore(runs_root).resolve_run("run-discovered")
    assert run is not None
    assert run.read_manifest()["stopped"] is True


def test_listener_reads_config_path_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listener = _load_listener()
    config = tmp_path / "custom.json"
    config.write_text(json.dumps({"policy": {"stop_threshold": 0.0}}), encoding="utf-8")
    monkeypatch.setenv("PATHLAYER_CONFIG", str(config))

    args = listener.parse_args(["--runs-root", str(tmp_path / "runs")])
    assert args.config == str(config)
    assert listener.load_listener_config(args.config).policy.stop_threshold == 0.0


def test_listener_rejects_an_invalid_config_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listener = _load_listener()
    config = tmp_path / "broken.json"
    config.write_text(json.dumps({"policy": {"nope": 1}}), encoding="utf-8")
    monkeypatch.delenv("PATHLAYER_CONFIG", raising=False)

    args = listener.parse_args(
        ["--runs-root", str(tmp_path / "runs"), "--config", str(config)],
    )
    with pytest.raises(SystemExit, match="invalid Pathlayer config"):
        listener.build_recorder(args)


def test_listener_rejects_a_missing_config_file(tmp_path: Path) -> None:
    listener = _load_listener()
    with pytest.raises(SystemExit, match="invalid Pathlayer config"):
        listener.load_listener_config(str(tmp_path / "absent.yaml"))


def test_decisions_recorded_on_pipeline_action(tmp_path: Path) -> None:
    loops = [
        _tool("read", run_id="run-loop", tool_call_id=f"c{i}", params={"path": "a.py"})
        for i in range(6)
    ]

    class _Factory:
        def __call__(self) -> Pipeline:
            from pathlayer.detectors.loop import LoopDetector

            return Pipeline([LoopDetector(k=2, window=32)])

    recorder = OpenClawRunRecorder(tmp_path, pipeline_factory=_Factory())
    recorder.process_payload(_before("run-loop"))
    for payload in loops:
        recorder.process_payload(payload)
    recorder.process_payload(_end("run-loop"))

    run = RunStore(tmp_path).resolve_run("run-loop")
    assert run is not None
    lines = run.decisions_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    last = json.loads(lines[-1])
    assert "action" in last
