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
from collections.abc import Callable, Sequence
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
from pathlayer.adapters.openclaw_run_store import RunDirectory, RunStore, atomic_write_json
from pathlayer.config import (
    BudgetConfig,
    ConfigError,
    DetectorsConfig,
    LoopConfig,
    PathlayerConfig,
    PipelineSection,
    PolicySection,
    ResourceConfig,
    build_pipeline,
)
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.resource import ResourceDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.store import (
    SQLiteStateStore,
    StateConcurrencyError,
    StateWriteError,
    configuration_fingerprint,
)
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


class _WatchdogHandle:
    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _WatchdogScheduler:
    def __init__(self) -> None:
        self.handle: _WatchdogHandle | None = None
        self.delay = 0.0
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def call_later(self, delay_seconds: float, callback: Callable[[], None]) -> _WatchdogHandle:
        assert delay_seconds == 10
        self.delay = delay_seconds
        self.handle = _WatchdogHandle(callback)
        return self.handle

    def fire(self) -> None:
        assert self.handle is not None and not self.handle.cancelled
        self.now += self.delay
        self.handle.callback()


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
    assert run.report_path.is_file()


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
    assert run.report_path.is_file()


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
    assert run.report_path.is_file()


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
    assert run.report_path.is_file()


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
    assert run.report_path.is_file()

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
    assert run.report_path.is_file()


def test_controlled_interruption_persists_the_failing_events_last_known_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PathlayerConfig(
        detectors=DetectorsConfig(
            budget=BudgetConfig(enabled=True, max_cost_usd=10.0, default_cost_usd=0.25)
        )
    )
    recorder = OpenClawRunRecorder(tmp_path, config=config)
    recorder.process_payload(_before("run-cost-interrupted"))
    active = recorder._runs["run-cost-interrupted"]

    def conflict(*args: Any, **kwargs: Any) -> Action:
        raise StateConcurrencyError("conflicting event identity")

    monkeypatch.setattr(active.pipeline, "process", conflict)
    recorder.process_payload(_tool("read", run_id="run-cost-interrupted"))

    manifest = active.directory.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["cost_estimated_usd"] == pytest.approx(0.25)
    stored_report = json.loads(active.directory.report_path.read_text(encoding="utf-8"))
    assert stored_report["cost"]["estimated_usd"] == pytest.approx(0.25)


def test_finalize_interrupted_run_writes_terminal_manifest_before_popping_active_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent hook must never see the run gone from ``_runs`` while its
    on-disk manifest is still non-terminal -- that window is what would let a
    re-attach reopen a run that is mid-finalization."""
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_before("run-finalize-order"))
    active = recorder._runs["run-finalize-order"]

    def conflict(*args: Any, **kwargs: Any) -> Action:
        raise StateConcurrencyError("conflicting event identity")

    monkeypatch.setattr(active.pipeline, "process", conflict)

    original_update_manifest = RunDirectory.update_manifest

    def spying_update_manifest(self: RunDirectory, **fields: Any) -> dict[str, Any]:
        # The manifest write must happen while the run is still tracked as
        # active: popping first would let a concurrent hook re-attach to a
        # run whose on-disk status is not yet terminal.
        assert "run-finalize-order" in recorder._runs
        return original_update_manifest(self, **fields)

    monkeypatch.setattr(RunDirectory, "update_manifest", spying_update_manifest)
    recorder.process_payload(_tool("read", run_id="run-finalize-order"))

    assert "run-finalize-order" not in recorder._runs
    assert active.directory.read_manifest()["status"] == "interrupted"


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


def test_openclaw_timeout_is_journaled_reported_and_exactly_replayable(tmp_path: Path) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("run-timeout"))

    scheduler.fire()
    scheduler.fire()  # replayed/concurrent expiry is idempotent

    run = RunStore(tmp_path).resolve_run("run-timeout")
    assert run is not None
    assert run.trace_path.is_file()
    assert run.report_path.is_file()
    timeout_report = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert "selected STOP and stopped analysis" in timeout_report["outcome_meaning"]
    assert "depends on the integration's actuator" in timeout_report["outcome_meaning"]
    trace = load_trace(run.trace_path)
    assert [event.kind for event in trace.events] == [
        EventKind.RUN_START,
        EventKind.RUN_TIMEOUT,
    ]
    decisions = _decisions(run)
    assert [decision["action"] for decision in decisions] == ["NONE", "STOP"]
    assert run.read_manifest()["stopped"] is True
    raw_records = [
        json.loads(line) for line in run.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        sum(
            record.get("payload", {}).get("event", {}).get("kind") == "run_timeout"
            for record in raw_records
        )
        == 1
    )

    replay = replay_trace(
        trace,
        detectors=[ResourceDetector(max_duration_seconds=10)],
        policy=Policy(),
        progress_anchors=[],
    )
    assert replay.intervened
    assert replay.detected_at_index == 1

    recorder.process_payload(_end("run-timeout"))


def test_openclaw_rss_over_threshold_stops_the_run(tmp_path: Path) -> None:
    # RES-3: inject a deterministic sampler (same DI idiom as clock/monotonic)
    # rather than faking the platform read -- the real sampler
    # (sample_rss_bytes) already has its own dedicated unit tests.
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_rss_bytes=1000))
    )
    recorder = OpenClawRunRecorder(tmp_path, config=config, sample_rss=lambda: 2000)
    recorder.process_payload(_before("run-rss"))
    recorder.process_payload(_tool("read", run_id="run-rss"))
    recorder.process_payload(_end("run-rss"))

    run = RunStore(tmp_path).resolve_run("run-rss")
    assert run is not None
    assert run.read_manifest()["stopped"] is True
    assert [decision["action"] for decision in _decisions(run)][-1] == "STOP"


def test_reserved_lifecycle_hook_cannot_be_injected_or_replayed(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    payload = {
        "hook": "pathlayer_lifecycle",
        "event": {"kind": "run_timeout", "metadata": {}},
        "ctx": _ctx("forged-timeout"),
    }

    with pytest.raises(ValueError, match="reserved"):
        recorder.process_payload(payload)
    assert recorder_module._translate_recorded_payload(payload) == []
    assert RunStore(tmp_path).resolve_run("forged-timeout") is None


def test_orphan_tool_starts_watchdog_at_first_observed_event(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )

    with caplog.at_level("WARNING"):
        recorder.process_payload(_tool("read", run_id="orphan-timeout"))
    active = recorder._runs["orphan-timeout"]
    assert [event.kind for event in active.events[:2]] == [
        EventKind.RUN_START,
        EventKind.TOOL_CALL,
    ]
    assert "earlier duration was not measured" in caplog.text

    scheduler.fire()
    assert active.pipeline.stopped


def test_openclaw_http_boundary_enforces_due_delayed_timer(tmp_path: Path) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0 + scheduler.now,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("delayed-openclaw-timer"))
    scheduler.now = 11.0

    recorder.process_payload(_tool("read", run_id="delayed-openclaw-timer"))

    active = recorder._runs["delayed-openclaw-timer"]
    assert active.pipeline.stopped
    live_kinds = [event.kind for event in active.events]
    assert live_kinds == [
        EventKind.RUN_START,
        EventKind.RUN_TIMEOUT,
        EventKind.TOOL_CALL,
    ]
    decisions = _decisions(active.directory)
    assert [decision["analyzed"] for decision in decisions] == [True, True, False]

    raw_records = [
        json.loads(line)
        for line in active.directory.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    raw_kinds = [
        event.kind
        for record in raw_records
        for event in recorder_module._translate_recorded_payload(
            record["payload"], internal_lifecycle=record.get("internal_lifecycle") is True
        )
    ]
    assert raw_kinds == live_kinds

    restarted = OpenClawRunRecorder(
        tmp_path,
        pipeline_factory=lambda: build_pipeline(config),
        clock=lambda: 100.0 + scheduler.now,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    restarted.process_payload(_tool("write", run_id="delayed-openclaw-timer"))
    replayed = restarted._runs["delayed-openclaw-timer"]
    assert [event.kind for event in replayed.events[:3]] == live_kinds


def test_triggering_payload_is_journaled_even_when_it_crosses_the_deadline(
    tmp_path: Path,
) -> None:
    # Same overdue-deadline setup as test_openclaw_http_boundary_enforces_due_
    # delayed_timer: the tool payload itself synchronously completes the run
    # via check_deadline(). Unlike that test, this one asserts the triggering
    # payload was still durably appended to the raw journal -- it must not
    # vanish just because handling it also happened to end the run.
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0 + scheduler.now,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("triggering-payload"))
    scheduler.now = 11.0

    recorder.process_payload(_tool("read", run_id="triggering-payload", tool_call_id="call-9"))

    active = recorder._runs["triggering-payload"]
    assert active.pipeline.stopped
    raw_records = [
        json.loads(line)
        for line in active.directory.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        record.get("payload", {}).get("event", {}).get("toolCallId") == "call-9"
        for record in raw_records
    )


def test_controlled_interruption_cancels_runtime_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("interrupted-watchdog"))
    active = recorder._runs["interrupted-watchdog"]

    def conflict(*args: Any, **kwargs: Any) -> Action:
        raise StateConcurrencyError("conflicting event identity")

    monkeypatch.setattr(active.pipeline, "process", conflict)
    recorder.process_payload(_tool("read", run_id="interrupted-watchdog"))

    assert scheduler.handle is not None and scheduler.handle.cancelled
    assert active.directory.read_manifest()["status"] == "interrupted"


def test_timeout_retry_reuses_one_openclaw_lifecycle_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("retry-timeout"))
    active = recorder._runs["retry-timeout"]
    original_process = active.pipeline.process
    failed = False

    def flaky_process(event: ToolEvent, *, event_id: str | None = None) -> Action:
        nonlocal failed
        if event.kind is EventKind.RUN_TIMEOUT and not failed:
            failed = True
            raise RuntimeError("transient detector failure")
        return original_process(event, event_id=event_id)

    monkeypatch.setattr(active.pipeline, "process", flaky_process)
    scheduler.fire()

    raw_records = [
        json.loads(line)
        for line in active.directory.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        sum(
            record.get("payload", {}).get("event", {}).get("kind") == "run_timeout"
            for record in raw_records
        )
        == 1
    )
    assert [event.kind for event in active.events].count(EventKind.RUN_TIMEOUT) == 1
    assert active.pipeline.stopped


def test_run_start_pipeline_rollback_retries_pending_record_and_arms_timer(
    tmp_path: Path,
) -> None:
    scheduler = _WatchdogScheduler()
    failed = False

    class _FailOnceAtStart(Detector):
        def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
            nonlocal failed
            if history and history[-1].kind is EventKind.RUN_START and not failed:
                failed = True
                raise RuntimeError("transient start failure")
            return Signal("start_probe", 0.0)

    recorder = OpenClawRunRecorder(
        tmp_path,
        pipeline_factory=lambda: Pipeline(
            [ResourceDetector(max_duration_seconds=10), _FailOnceAtStart()], Policy()
        ),
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )

    recorder.process_payload(_before("retry-run-start"))

    active = recorder._runs["retry-run-start"]
    assert scheduler.handle is not None
    assert [event.kind for event in active.pipeline.history] == [EventKind.RUN_START]
    raw_records = [
        json.loads(line)
        for line in active.directory.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        sum(
            record.get("payload", {}).get("event", {}).get("kind") == "run_start"
            for record in raw_records
        )
        == 1
    )

    scheduler.fire()
    assert active.pipeline.stopped


def test_lifecycle_fsync_failure_interrupts_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("fsync-timeout"))
    active = recorder._runs["fsync-timeout"]
    original_append = recorder_module.append_jsonl
    failed = False

    def append_then_fail(path: Path, record: dict[str, Any]) -> None:
        nonlocal failed
        original_append(path, record)
        payload = record.get("payload", {})
        if payload.get("event", {}).get("kind") == "run_timeout" and not failed:
            failed = True
            raise OSError("simulated fsync failure after write")

    monkeypatch.setattr(recorder_module, "append_jsonl", append_then_fail)
    scheduler.fire()

    raw_records = [
        json.loads(line)
        for line in active.directory.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        sum(
            record.get("payload", {}).get("event", {}).get("kind") == "run_timeout"
            for record in raw_records
        )
        == 1
    )
    assert not active.pipeline.stopped
    assert active.completed
    assert active.directory.read_manifest()["status"] == "interrupted"
    assert "fsync-timeout" not in recorder._runs


def test_run_start_fsync_failure_cannot_resurrect_interrupted_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    original_append = recorder_module.append_jsonl

    def append_start_then_fail(path: Path, record: dict[str, Any]) -> None:
        original_append(path, record)
        payload = record.get("payload", {})
        if payload.get("event", {}).get("kind") == "run_start":
            raise OSError("simulated RUN_START fsync failure after write")

    monkeypatch.setattr(recorder_module, "append_jsonl", append_start_then_fail)
    recorder.process_payload(_before("fsync-run-start"))

    run = RunStore(tmp_path).resolve_run("fsync-run-start")
    assert run is not None
    assert run.read_manifest()["status"] == "interrupted"
    assert not run.request_path.exists()
    assert "fsync-run-start" not in recorder._runs


def test_orphan_start_failure_cannot_append_payload_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    original_append = recorder_module.append_jsonl

    def append_start_then_fail(path: Path, record: dict[str, Any]) -> None:
        original_append(path, record)
        if record.get("payload", {}).get("event", {}).get("kind") == "run_start":
            raise OSError("simulated orphan RUN_START fsync failure after write")

    monkeypatch.setattr(recorder_module, "append_jsonl", append_start_then_fail)
    recorder.process_payload(_tool("read", run_id="orphan-start-failure"))

    run = RunStore(tmp_path).resolve_run("orphan-start-failure")
    assert run is not None
    raw_records = [json.loads(line) for line in run.raw_events_path.read_text().splitlines()]
    assert [record["sequence"] for record in raw_records] == [1]
    assert run.read_manifest()["status"] == "interrupted"
    assert run.read_manifest()["event_count"] == len(raw_records)
    assert "orphan-start-failure" not in recorder._runs


@pytest.mark.parametrize("artifact", ["decision", "manifest", "trace", "report"])
def test_timeout_artifact_failure_interrupts_after_durable_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    run_id = f"timeout-{artifact}-failure"
    recorder.process_payload(_before(run_id))
    active = recorder._runs[run_id]
    failed = False

    if artifact == "decision":
        original_record_decision = recorder._record_decision

        def fail_timeout_decision(
            active_run: Any,
            record: dict[str, Any],
            event: ToolEvent,
            action: Action,
            *,
            index: int,
            analyzed: bool = True,
        ) -> None:
            nonlocal failed
            if event.kind is EventKind.RUN_TIMEOUT and not failed:
                failed = True
                raise OSError("simulated timeout decision failure")
            original_record_decision(
                active_run, record, event, action, index=index, analyzed=analyzed
            )

        monkeypatch.setattr(recorder, "_record_decision", fail_timeout_decision)
    elif artifact == "manifest":
        original_update_manifest = RunDirectory.update_manifest

        def fail_timeout_manifest(self: RunDirectory, **fields: Any) -> dict[str, Any]:
            nonlocal failed
            if "stopped" in fields and "status" not in fields and not failed:
                failed = True
                raise OSError("simulated timeout manifest failure")
            return original_update_manifest(self, **fields)

        monkeypatch.setattr(RunDirectory, "update_manifest", fail_timeout_manifest)
    elif artifact == "trace":
        original_atomic_write = recorder_module.atomic_write_json

        def fail_timeout_trace(path: Path, payload: Any) -> None:
            nonlocal failed
            if path == active.directory.trace_path and not failed:
                failed = True
                raise OSError("simulated timeout trace failure")
            original_atomic_write(path, payload)

        monkeypatch.setattr(recorder_module, "atomic_write_json", fail_timeout_trace)
    else:
        original_write_report = recorder_module.write_run_report

        def fail_timeout_report(directory: RunDirectory) -> Any:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("simulated timeout report failure")
            return original_write_report(directory)

        monkeypatch.setattr(recorder_module, "write_run_report", fail_timeout_report)

    scheduler.fire()

    assert failed
    assert active.pipeline.stopped
    assert active.completed
    assert [event.kind for event in active.pipeline.history].count(EventKind.RUN_TIMEOUT) == 1
    assert run_id not in recorder._runs
    manifest = active.directory.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["stopped"] is True
    assert manifest.get("state_durable") is not False
    assert active.directory.trace_path.is_file()
    assert active.directory.report_path.is_file()
    raw_records = [
        json.loads(line)
        for line in active.directory.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        sum(
            record.get("payload", {}).get("event", {}).get("kind") == "run_timeout"
            for record in raw_records
        )
        == 1
    )
    assert recorder._state_store is not None
    assert recorder._state_store.read_scope(run_id).status == "stopped"


def test_timeout_durability_manifest_failure_interrupts_memory_only_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    run_id = "timeout-durability-manifest-failure"
    recorder.process_payload(_before(run_id))
    active = recorder._runs[run_id]
    assert recorder._state_store is not None

    def fail_timeout_append(*args: Any, **kwargs: Any) -> Any:
        raise StateWriteError("simulated timeout SQLite failure")

    original_update_manifest = RunDirectory.update_manifest
    failed_manifest = False

    def fail_first_durability_manifest(self: RunDirectory, **fields: Any) -> dict[str, Any]:
        nonlocal failed_manifest
        if fields.get("state_durable") is False and not failed_manifest:
            failed_manifest = True
            raise OSError("simulated durability manifest failure")
        return original_update_manifest(self, **fields)

    monkeypatch.setattr(recorder._state_store, "append_tick", fail_timeout_append)
    monkeypatch.setattr(RunDirectory, "update_manifest", fail_first_durability_manifest)
    scheduler.fire()

    assert failed_manifest
    assert active.pipeline.stopped
    assert not active.pipeline.durable
    assert active.completed
    assert run_id not in recorder._runs
    manifest = active.directory.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["state_durable"] is False
    assert manifest["persistence_error"] == "StateWriteError"
    assert active.directory.trace_path.is_file()
    assert active.directory.report_path.is_file()
    raw_records = [
        json.loads(line)
        for line in active.directory.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        sum(
            record.get("payload", {}).get("event", {}).get("kind") == "run_timeout"
            for record in raw_records
        )
        == 1
    )


def test_lifecycle_prewrite_failure_retries_with_contiguous_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("prewrite-timeout"))
    active = recorder._runs["prewrite-timeout"]
    original_append = recorder_module.append_jsonl
    failed = False

    def fail_before_append(path: Path, record: dict[str, Any]) -> None:
        nonlocal failed
        payload = record.get("payload", {})
        if payload.get("event", {}).get("kind") == "run_timeout" and not failed:
            failed = True
            raise OSError("simulated pre-write failure")
        original_append(path, record)

    monkeypatch.setattr(recorder_module, "append_jsonl", fail_before_append)
    scheduler.fire()

    raw_records = [
        json.loads(line)
        for line in active.directory.raw_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sequence"] for record in raw_records] == list(range(1, len(raw_records) + 1))
    assert (
        sum(
            record.get("payload", {}).get("event", {}).get("kind") == "run_timeout"
            for record in raw_records
        )
        == 1
    )
    assert active.directory.read_manifest()["event_count"] == len(raw_records)
    assert active.pipeline.stopped


def test_partial_lifecycle_write_interrupts_instead_of_appending_after_torn_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("partial-timeout"))
    active = recorder._runs["partial-timeout"]
    original_append = recorder_module.append_jsonl

    def partial_timeout_append(path: Path, record: dict[str, Any]) -> None:
        payload = record.get("payload", {})
        if payload.get("event", {}).get("kind") == "run_timeout":
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"partial_timeout":')
                handle.flush()
            raise OSError("simulated partial write")
        original_append(path, record)

    monkeypatch.setattr(recorder_module, "append_jsonl", partial_timeout_append)
    scheduler.fire()

    assert active.completed
    assert active.directory.read_manifest()["status"] == "interrupted"
    assert "partial-timeout" not in recorder._runs
    assert (
        active.directory.raw_events_path.read_text(encoding="utf-8").count('"partial_timeout"') == 1
    )


def test_interruption_cleanup_survives_manifest_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _WatchdogScheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    recorder.process_payload(_before("failed-interruption-manifest"))
    active = recorder._runs["failed-interruption-manifest"]

    def conflict(*args: Any, **kwargs: Any) -> Action:
        raise StateConcurrencyError("conflicting event identity")

    original_update = RunDirectory.update_manifest
    calls = 0

    def fail_first_manifest(self: RunDirectory, **fields: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient disk error")
        return original_update(self, **fields)

    monkeypatch.setattr(active.pipeline, "process", conflict)
    monkeypatch.setattr(RunDirectory, "update_manifest", fail_first_manifest)
    recorder.process_payload(_tool("read", run_id="failed-interruption-manifest"))

    assert active.completed
    assert scheduler.handle is not None and scheduler.handle.cancelled
    assert "failed-interruption-manifest" not in recorder._runs
    assert active.directory.read_manifest()["status"] == "interrupted"


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
