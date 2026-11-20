"""Blocking the tool calls of a stopped OpenClaw run (opt-in STOP enforcement).

The reference plugin asks the recorder before each tool call through a
``before_tool_call`` query; :meth:`OpenClawRunRecorder.tool_call_decision`
answers it. These tests pin the three promises of that answer: it blocks only
a run Pathlayer stopped, only when enforcement is on, and it never changes what
is recorded.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pathlayer.adapters import openclaw_cli
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_report import build_run_report
from pathlayer.adapters.openclaw_run_store import RunStore
from pathlayer.config import DetectorsConfig, PathlayerConfig, ResourceConfig


def _ctx(run_id: str) -> dict[str, Any]:
    return {"runId": run_id, "sessionKey": "sess-1", "agentId": "agent-1"}


def _start(run_id: str) -> dict[str, Any]:
    return {"hook": "before_agent_run", "event": {"prompt": "fix the bug"}, "ctx": _ctx(run_id)}


def _read(run_id: str, index: int) -> dict[str, Any]:
    # Identical calls: the default pipeline STOPs on the loop by the 6th.
    return {
        "hook": "after_tool_call",
        "event": {
            "toolName": "read",
            "params": {"path": "a.py"},
            "result": "ok",
            "toolCallId": f"call-{index}",
            "runId": run_id,
        },
        "ctx": _ctx(run_id),
    }


def _end(run_id: str) -> dict[str, Any]:
    # A successful end: a failed one ends only an attempt, and a STOP keeps
    # blocking the next attempt (see test_openclaw_multi_attempt.py).
    return {"hook": "agent_end", "event": {"success": True}, "ctx": _ctx(run_id)}


def _query(run_id: str) -> dict[str, Any]:
    """What the plugin sends before a tool call: identifiers only."""
    return {
        "hook": "before_tool_call",
        "event": {"toolName": "exec", "toolCallId": "next"},
        "ctx": _ctx(run_id),
    }


def _decisions(recorder: OpenClawRunRecorder, run_id: str) -> list[dict[str, Any]]:
    run = RunStore(recorder.runs_root).resolve_run(run_id)
    assert run is not None
    lines = run.decisions_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _snapshot(recorder: OpenClawRunRecorder) -> dict[str, str]:
    """Every recorded file except the gate log, which logs the queries themselves."""
    root = recorder.runs_root
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix not in {".db", ".db-wal", ".db-shm"}
        and path.name != "gate.jsonl"
    }


def _gate(recorder: OpenClawRunRecorder, run_id: str) -> list[dict[str, Any]]:
    run = RunStore(recorder.runs_root).resolve_run(run_id)
    assert run is not None
    if not run.gate_path.is_file():
        return []
    return [json.loads(line) for line in run.gate_path.read_text(encoding="utf-8").splitlines()]


def _loop_until_stop(recorder: OpenClawRunRecorder, run_id: str) -> list[dict[str, Any] | None]:
    """Feed identical calls; return the decision a tool call would get after each."""
    recorder.process_payload(_start(run_id))
    answers = []
    for index in range(8):
        recorder.process_payload(_read(run_id, index))
        answers.append(recorder.tool_call_decision(_query(run_id)))
    return answers


def test_an_enforced_stop_blocks_every_later_tool_call_of_that_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()

    answers = _loop_until_stop(recorder, "run-a")

    first_stop = next(
        index
        for index, line in enumerate(
            line for line in _decisions(recorder, "run-a") if line["tool"] == "read"
        )
        if line["action"] == "STOP"
    )
    assert answers[:first_stop] == [None] * first_stop
    blocked = answers[first_stop:]
    assert blocked and all(answer == blocked[0] for answer in blocked)
    assert blocked[0] is not None and blocked[0]["block"] is True
    reason = blocked[0]["blockReason"]
    assert reason.startswith("Pathlayer stopped this run (loop: ")
    assert "Do not call more tools" in reason


def test_a_stop_blocks_only_the_run_it_stopped(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    recorder.process_payload(_start("run-b"))
    recorder.process_payload(_read("run-b", 0))

    _loop_until_stop(recorder, "run-a")

    assert recorder.tool_call_decision(_query("run-a")) is not None
    assert recorder.tool_call_decision(_query("run-b")) is None
    assert recorder.tool_call_decision(_query("never-seen")) is None
    no_run_id = {"hook": "before_tool_call", "event": {"toolName": "exec"}, "ctx": {}}
    assert recorder.tool_call_decision(no_run_id) is None


def test_without_enforcement_nothing_is_ever_blocked(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.startup()

    answers = _loop_until_stop(recorder, "run-a")

    assert any(line["action"] == "STOP" for line in _decisions(recorder, "run-a"))
    assert answers == [None] * len(answers)


def test_only_a_before_tool_call_query_gets_a_decision(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    _loop_until_stop(recorder, "run-a")

    assert recorder.tool_call_decision(_read("run-a", 99)) is None
    assert recorder.tool_call_decision({**_query("run-a"), "hook": "unknown"}) is None
    assert recorder.tool_call_decision("not a payload") is None


def test_a_finished_run_is_not_blocked(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    _loop_until_stop(recorder, "run-a")

    recorder.process_payload(_end("run-a"))

    assert recorder.tool_call_decision(_query("run-a")) is None


def test_queries_never_change_what_is_recorded(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    _loop_until_stop(recorder, "run-a")
    before = _snapshot(recorder)

    for run_id in ("run-a", "run-a", "ghost"):
        recorder.tool_call_decision(_query(run_id))
        # A query that reaches the ingest path is dropped before anything is
        # written, and never opens a run.
        recorder.process_payload(_query(run_id))

    assert _snapshot(recorder) == before
    assert RunStore(tmp_path).resolve_run("ghost") is None
    # Only the two queries about the recorded run are logged.
    assert len(_gate(recorder, "run-a")) == 8 + 2


# --------------------------------------------------------------------------- #
# Gate log (gate.jsonl)
# --------------------------------------------------------------------------- #


def test_every_gate_query_is_logged_with_its_answer(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    answers = _loop_until_stop(recorder, "run-a")

    gate = _gate(recorder, "run-a")
    assert [line["decision"] for line in gate] == [
        "block" if answer is not None else "allow" for answer in answers
    ]
    assert all(line["enforced"] is True for line in gate)
    assert all(line["tool_name"] == "exec" for line in gate)
    blocked = [line for line in gate if line["decision"] == "block"]
    assert blocked and all(line["run_stopped"] is True for line in blocked)
    assert all(line["reason"].startswith("loop:") for line in blocked)
    assert all(line["reason"] is None for line in gate if not line["run_stopped"])

    recorder.process_payload(_end("run-a"))
    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None
    report = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert report["tool_call_gate"]["queries"] == len(gate)
    assert report["tool_call_gate"]["enforced"] is True
    assert len(report["tool_call_gate"]["blocked"]) == len(blocked)
    assert report["tool_call_gate"]["would_have_blocked"] == 0
    text = build_run_report(run).format()
    assert (
        f"tool-call gate (enforced): {len(gate)} tool call(s) checked, {len(blocked)} blocked"
        in text
    )
    assert "  blocked exec (next) at " in text


def test_a_passive_recording_logs_what_enforcement_would_have_blocked(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.startup()
    answers = _loop_until_stop(recorder, "run-a")
    assert answers == [None] * len(answers)

    gate = _gate(recorder, "run-a")
    assert all(line["decision"] == "allow" and line["enforced"] is False for line in gate)
    stopped = [line for line in gate if line["run_stopped"]]
    assert stopped

    recorder.process_payload(_end("run-a"))
    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None
    report = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert report["tool_call_gate"]["blocked"] == []
    assert report["tool_call_gate"]["would_have_blocked"] == len(stopped)
    assert "would have been blocked with enforcement on" in build_run_report(run).format()


def test_a_finished_or_unknown_run_logs_nothing(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a"))

    assert recorder.tool_call_decision(_query("run-a")) is None
    assert recorder.tool_call_decision(_query("ghost")) is None
    assert _gate(recorder, "run-a") == []
    assert RunStore(tmp_path).resolve_run("ghost") is None


def test_a_gate_log_failure_never_changes_the_answer(tmp_path: Path, monkeypatch: Any) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    _loop_until_stop(recorder, "run-a")

    real_open = Path.open

    def fail(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == "gate.jsonl":
            raise OSError("disk full")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    answer = recorder.tool_call_decision(_query("run-a"))
    assert answer is not None and answer["block"] is True


def test_a_report_without_a_gate_log_has_no_gate_section(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_read("run-a", 0))
    recorder.process_payload(_end("run-a"))

    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None
    assert json.loads(run.report_path.read_text(encoding="utf-8"))["tool_call_gate"] is None


def test_a_corrupt_gate_log_costs_only_the_gate_section(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    _loop_until_stop(recorder, "run-a")
    recorder.process_payload(_end("run-a"))
    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None
    run.gate_path.write_text("not json\n{}\n", encoding="utf-8")

    report = build_run_report(run)
    assert report.tool_call_gate is None
    assert any("tool-call gate log is corrupt" in note for note in report.notes)
    assert report.actions


class _Handle:
    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback

    def cancel(self) -> None:
        pass


class _Scheduler:
    """Fires the run's duration watchdog on demand."""

    def __init__(self) -> None:
        self.now = 0.0
        self.delay = 0.0
        self.handle: _Handle | None = None

    def monotonic(self) -> float:
        return self.now

    def call_later(self, delay_seconds: float, callback: Callable[[], None]) -> _Handle:
        self.delay = delay_seconds
        self.handle = _Handle(callback)
        return self.handle

    def fire(self) -> None:
        assert self.handle is not None
        self.now += self.delay
        self.handle.callback()


def test_a_duration_timeout_stop_is_enforced_too(tmp_path: Path) -> None:
    scheduler = _Scheduler()
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        tmp_path,
        config=config,
        clock=lambda: 100.0,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
        enforce=True,
    )
    recorder.process_payload(_start("run-t"))
    assert recorder.tool_call_decision(_query("run-t")) is None

    scheduler.fire()

    decision = recorder.tool_call_decision(_query("run-t"))
    assert decision is not None and decision["block"] is True
    assert decision["blockReason"].startswith("Pathlayer stopped this run (")


def test_a_listener_restart_ends_the_block_of_an_interrupted_run(tmp_path: Path) -> None:
    # Documented limit: the startup sweep marks runs in progress as interrupted,
    # a terminal status. They are never analyzed again, so no longer blocked.
    first = OpenClawRunRecorder(tmp_path, enforce=True)
    first.startup()
    _loop_until_stop(first, "run-a")
    assert first.tool_call_decision(_query("run-a")) is not None

    restarted = OpenClawRunRecorder(tmp_path, enforce=True)
    restarted.startup()
    restarted.process_payload(_read("run-a", 50))

    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None and run.read_manifest()["status"] == "interrupted"
    assert restarted.tool_call_decision(_query("run-a")) is None


def _post(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/ingest",
        data=json.dumps({"source": "openclaw", "payload": payload}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body: dict[str, Any] = json.loads(response.read())
        return body


def test_the_http_ingest_returns_the_block_to_the_plugin(tmp_path: Path) -> None:
    for enforce in (True, False):
        recorder = OpenClawRunRecorder(tmp_path / str(enforce), enforce=enforce)
        recorder.startup()
        adapter = recorder.attach_http(port=0)
        try:
            _post(adapter.port, _start("run-a"))
            assert _post(adapter.port, _query("run-a")) == {"ok": True}
            for index in range(8):
                _post(adapter.port, _read("run-a", index))
            answer = _post(adapter.port, _query("run-a"))
        finally:
            adapter.detach()
        if enforce:
            assert answer["ok"] is True
            assert answer["decision"]["block"] is True
            assert answer["decision"]["blockReason"].startswith("Pathlayer stopped this run")
        else:
            assert answer == {"ok": True}


def test_the_recording_listener_enforces_only_with_enforce(tmp_path: Path) -> None:
    for argv, expected in (([], False), (["--enforce"], True)):
        args = openclaw_cli.parse_args(
            ["record", "--runs-root", str(tmp_path / str(expected)), *argv]
        )
        assert args.enforce is expected
        recorder = openclaw_cli.build_recorder(args)
        recorder.startup()
        answers = _loop_until_stop(recorder, "run-a")
        assert (answers[-1] is not None) is expected


def test_the_text_report_lists_the_first_blocked_calls_only(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    _loop_until_stop(recorder, "run-a")
    for _ in range(20):
        recorder.tool_call_decision(_query("run-a"))
    recorder.process_payload(_end("run-a"))
    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None

    report = build_run_report(run)
    assert report.tool_call_gate is not None
    blocked = len(report.tool_call_gate.blocked)
    assert blocked > 10
    text = report.format()
    assert text.count("  blocked exec") == 10
    assert f"... and {blocked - 10} more blocked call(s)" in text
    listed = json.loads(run.report_path.read_text(encoding="utf-8"))["tool_call_gate"]["blocked"]
    assert len(listed) == blocked
