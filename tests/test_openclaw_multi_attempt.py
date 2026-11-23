"""OpenClaw runs made of several attempts (issue #53).

An OpenClaw run is a loop of attempts sharing one ``runId``: after a context
compaction, a timeout compaction or a provider failover, OpenClaw starts another
attempt, and every attempt sends its own ``agent_end`` (``success: false`` when
it failed) followed by its own ``llm_output``. These tests pin what the recorder
owes such a run:

- a failed attempt ends the attempt, not the run: the next attempt continues the
  same pipeline, sequence, cost and STOP, and an enforced STOP keeps blocking;
- the run ends for good at any other ``agent_end``, when the next run of its
  session starts, or at the startup sweep -- with a final trace holding every
  event, whose replay reproduces the live decisions;
- the duration watchdog never fires on a run that may already have ended.

The payloads are synthetic: they reproduce the hook order and field names read
in OpenClaw's source (2026.7.1 and 2026.10.10), not a recorded compaction.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import pathlayer.adapters.openclaw_recorder as recorder_module
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_run_store import RunDirectory, RunStore
from pathlayer.config import (
    DetectorsConfig,
    PathlayerConfig,
    PricingSection,
    ResourceConfig,
    build_pipeline,
)
from pathlayer.events import EventKind
from pathlayer.pipeline import Pipeline
from pathlayer.policies.actions import action_name
from pathlayer.postmortem import SessionOutcome, build_session_report
from pathlayer.pricing import ModelPrice
from pathlayer.validation.traces import load_trace

# 1000 total tokens at the default 0.8 input ratio: 800 * $1/M + 200 * $2/M.
_CALL_USD = 0.0012
_PRICED = PathlayerConfig(pricing=PricingSection(models={"model": ModelPrice(1.0, 2.0)}))


def _ctx(run_id: str, session: str) -> dict[str, Any]:
    return {"runId": run_id, "sessionKey": session, "agentId": "agent-1"}


def _start(run_id: str, *, session: str = "sess-1", prompt: str = "fix the bug") -> dict[str, Any]:
    return {"hook": "before_agent_run", "event": {"prompt": prompt}, "ctx": _ctx(run_id, session)}


def _tool(run_id: str, call_id: str, *, session: str = "sess-1") -> dict[str, Any]:
    # Identical calls: the default pipeline STOPs on the loop by the 6th.
    return {
        "hook": "after_tool_call",
        "event": {
            "toolName": "read",
            "params": {"path": "a.py"},
            "result": "ok",
            "toolCallId": call_id,
            "runId": run_id,
        },
        "ctx": _ctx(run_id, session),
    }


def _end(
    run_id: str, *, success: Any, duration_ms: int = 1000, session: str = "sess-1"
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "success": success,
        "durationMs": duration_ms,
        "messages": [{"role": "assistant", "content": f"attempt of {duration_ms} ms"}],
    }
    if success is False:
        event["error"] = "context overflow"
    return {"hook": "agent_end", "event": event, "ctx": _ctx(run_id, session)}


def _llm(run_id: str, *, session: str = "sess-1") -> dict[str, Any]:
    return {
        "hook": "llm_output",
        "event": {"model": "model", "usage": {"total": 1000}},
        "ctx": _ctx(run_id, session),
    }


def _query(run_id: str) -> dict[str, Any]:
    return {
        "hook": "before_tool_call",
        "event": {"toolName": "read", "toolCallId": "next"},
        "ctx": _ctx(run_id, "sess-1"),
    }


def _run(root: Path, run_id: str) -> RunDirectory:
    run = RunStore(root).resolve_run(run_id)
    assert run is not None
    return run


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _report(run: RunDirectory) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(run.report_path.read_text(encoding="utf-8"))
    return report


def _files(run: RunDirectory) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(run.root.iterdir()) if path.is_file()}


def _replayed_actions(pipeline: Pipeline, run: RunDirectory) -> list[str]:
    """Replay the final trace through a fresh pipeline, as an offline reader would."""
    actions = []
    for event in load_trace(run.trace_path).events:
        if pipeline.stopped:
            break
        actions.append(action_name(pipeline.process(event)))
    return actions


def _live_actions(run: RunDirectory) -> list[str]:
    return [line["action"] for line in _jsonl(run.decisions_path) if line["analyzed"]]


class _Handle:
    def __init__(self, delay: float, callback: Callable[[], None]) -> None:
        self.delay = delay
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _Scheduler:
    """Records every timer the watchdog arms; nothing fires unless a test says so."""

    def __init__(self) -> None:
        self.handles: list[_Handle] = []
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def call_later(self, delay_seconds: float, callback: Callable[[], None]) -> _Handle:
        handle = _Handle(delay_seconds, callback)
        self.handles.append(handle)
        return handle

    def live(self) -> list[_Handle]:
        return [handle for handle in self.handles if not handle.cancelled]


class _Clock:
    """Wall clock and monotonic clock moving together, under test control."""

    def __init__(self, scheduler: _Scheduler, start: float = 100.0) -> None:
        self.scheduler = scheduler
        self.offset = start

    def __call__(self) -> float:
        return self.offset + self.scheduler.now

    def advance(self, seconds: float) -> None:
        self.scheduler.now += seconds


def _timed_recorder(root: Path, *, enforce: bool = False) -> tuple[OpenClawRunRecorder, _Clock]:
    scheduler = _Scheduler()
    clock = _Clock(scheduler)
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10))
    )
    recorder = OpenClawRunRecorder(
        root,
        config=config,
        clock=clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
        enforce=enforce,
    )
    return recorder, clock


# --------------------------------------------------------------------------- #
# One run across attempts
# --------------------------------------------------------------------------- #


def test_a_failed_attempt_keeps_the_run_open_with_a_provisional_report(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_end("run-a", success=False))

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "awaiting_retry"
    assert manifest["failed_attempts"] == 1
    assert manifest["ended_at"] is None
    assert not run.final_path.exists()
    assert not run.trace_path.exists()
    report = _report(run)
    assert report["outcome"] == "in_progress"
    assert "provisional" in report["outcome_meaning"]
    assert any(note.startswith("provisional report: attempt 1") for note in report["notes"])
    assert not any("never reached agent_end" in note for note in report["notes"])
    assert report["duration_seconds"] is None


def test_the_next_attempt_continues_the_same_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a", prompt="first"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_end("run-a", success=False, duration_ms=1500))
    recorder.process_payload(_start("run-a", prompt="compacted"))
    assert _run(tmp_path, "run-a").read_manifest()["status"] == "started"
    recorder.process_payload(_tool("run-a", "c2"))
    recorder.process_payload(_end("run-a", success=True, duration_ms=2500))

    run = _run(tmp_path, "run-a")
    assert [other.original_run_id for other in RunStore(tmp_path).list_runs()] == ["run-a"]
    raw = _jsonl(run.raw_events_path)
    assert [record["sequence"] for record in raw] == list(range(1, len(raw) + 1))
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["failed_attempts"] == 1
    assert manifest["event_count"] == len(raw)
    # The first attempt's request stays the run's request.
    assert json.loads(run.request_path.read_text(encoding="utf-8"))["prompt"] == "first"
    assert json.loads(run.final_path.read_text(encoding="utf-8"))["success"] is True
    trace = load_trace(run.trace_path)
    assert [event.tool for event in trace.events] == ["read", "read"]
    assert trace.metadata["failed_attempts"] == 1
    assert [line["event_index"] for line in _jsonl(run.decisions_path)] == [0, 1]
    report = _report(run)
    assert report["outcome"] == "completed"
    assert report["duration_seconds"] == pytest.approx(4.0)
    assert any(
        "2 attempts, 1 of which did not succeed" in note
        and "without the time between attempts" in note
        for note in report["notes"]
    )
    assert not run.late_events_path.exists()


def test_an_attempt_without_a_duration_leaves_the_run_duration_unknown(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    failed = _end("run-a", success=False)
    del failed["event"]["durationMs"]
    recorder.process_payload(failed)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a", success=True))

    report = _report(_run(tmp_path, "run-a"))
    assert report["duration_seconds"] is None
    assert any("at least one attempt reported none" in note for note in report["notes"])


def test_replaying_the_final_trace_reproduces_the_live_decisions(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-loop"))
    for index in range(3):
        recorder.process_payload(_tool("run-loop", f"a{index}"))
    recorder.process_payload(_end("run-loop", success=False))
    recorder.process_payload(_start("run-loop"))
    for index in range(6):
        recorder.process_payload(_tool("run-loop", f"b{index}"))
    recorder.process_payload(_end("run-loop", success=True))

    run = _run(tmp_path, "run-loop")
    live = _live_actions(run)
    assert "STOP" in live
    assert _replayed_actions(build_pipeline(PathlayerConfig()), run) == live
    # Every event is in the trace, the ones past the STOP included.
    assert len(load_trace(run.trace_path).events) == len(_jsonl(run.decisions_path)) == 9


# --------------------------------------------------------------------------- #
# STOP across attempts
# --------------------------------------------------------------------------- #


def _stop_in_first_attempt(recorder: OpenClawRunRecorder, run_id: str) -> None:
    recorder.process_payload(_start(run_id))
    for index in range(8):
        recorder.process_payload(_tool(run_id, f"a{index}"))
    recorder.process_payload(_end(run_id, success=False))


def test_an_enforced_stop_keeps_blocking_the_next_attempt(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    _stop_in_first_attempt(recorder, "run-a")

    blocked = recorder.tool_call_decision(_query("run-a"))
    assert blocked is not None and blocked["block"] is True
    recorder.process_payload(_start("run-a"))
    assert recorder.tool_call_decision(_query("run-a")) == blocked
    recorder.process_payload(_tool("run-a", "b0"))
    assert recorder.tool_call_decision(_query("run-a")) == blocked

    run = _run(tmp_path, "run-a")
    last = _jsonl(run.decisions_path)[-1]
    assert (last["analyzed"], last["action"], last["stopped"]) == (False, "STOP", True)

    recorder.process_payload(_end("run-a", success=True))
    assert recorder.tool_call_decision(_query("run-a")) is None
    report = _report(run)
    assert report["outcome"] == "stopped"


def test_without_enforcement_the_next_attempt_is_recorded_but_not_analyzed(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.startup()
    _stop_in_first_attempt(recorder, "run-a")
    assert _report(_run(tmp_path, "run-a"))["outcome"] == "stopped"

    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "b0"))

    assert recorder.tool_call_decision(_query("run-a")) is None
    lines = _jsonl(_run(tmp_path, "run-a").decisions_path)
    assert len(lines) == 9
    assert (lines[-1]["analyzed"], lines[-1]["action"]) == (False, "STOP")


# --------------------------------------------------------------------------- #
# Cost across attempts
# --------------------------------------------------------------------------- #


def test_a_usage_report_alone_does_not_resume_the_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_end("run-a", success=False))
    before = _report(_run(tmp_path, "run-a"))

    recorder.process_payload(_llm("run-a"))

    run = _run(tmp_path, "run-a")
    assert run.read_manifest()["status"] == "awaiting_retry"
    # Still this run's spend, analyzed in-run: in the raw journal, not the annex.
    assert [record["hook"] for record in _jsonl(run.raw_events_path)][-1] == "llm_output"
    assert not run.late_events_path.exists()
    after = _report(run)
    assert after["outcome"] == "in_progress"
    assert after["cost"]["estimated_usd"] == pytest.approx(_CALL_USD)
    assert after["cost"] != before["cost"]


def test_usage_between_attempts_is_in_run_and_usage_after_the_end_is_post_run(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_end("run-a", success=False))
    recorder.process_payload(_llm("run-a"))  # first attempt's usage
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c2"))
    recorder.process_payload(_end("run-a", success=True))
    recorder.process_payload(_llm("run-a"))  # last attempt's usage, after the end

    run = _run(tmp_path, "run-a")
    raw_hooks = [record["hook"] for record in _jsonl(run.raw_events_path)]
    assert raw_hooks.count("llm_output") == 1
    assert len(_jsonl(run.late_events_path)) == 1
    cost = _report(run)["cost"]
    assert cost["estimated_usd"] == pytest.approx(_CALL_USD)
    assert cost["post_run_estimated_usd"] == pytest.approx(_CALL_USD)
    assert cost["total_usd"] == pytest.approx(2 * _CALL_USD)


# --------------------------------------------------------------------------- #
# Duplicates and older payload shapes
# --------------------------------------------------------------------------- #


def test_a_duplicated_failed_agent_end_writes_nothing(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    failed = _end("run-a", success=False)
    recorder.process_payload(failed)
    run = _run(tmp_path, "run-a")
    files = _files(run)

    recorder.process_payload(json.loads(json.dumps(failed)))
    assert _files(run) == files
    # The usage report OpenClaw sends after the end proves no new attempt.
    recorder.process_payload(_llm("run-a"))
    files = _files(run)
    recorder.process_payload(json.loads(json.dumps(failed)))
    assert _files(run) == files
    assert run.read_manifest()["status"] == "awaiting_retry"
    assert run.read_manifest()["failed_attempts"] == 1


def test_identical_failed_ends_of_two_attempts_are_two_attempts(tmp_path: Path) -> None:
    # Two attempts can fail with the same error, snapshot and duration: once a
    # new attempt has started, the identical end is that attempt's own.
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    failed = _end("run-a", success=False)
    recorder.process_payload(failed)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(json.loads(json.dumps(failed)))

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "awaiting_retry"
    assert manifest["failed_attempts"] == 2
    ends = [record for record in _jsonl(run.raw_events_path) if record["hook"] == "agent_end"]
    assert len(ends) == 2

    # The waiting run can therefore still be ended by the next run of its session.
    recorder.process_payload(_start("run-b"))
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["ended_by"] == "next_run_in_session"
    assert manifest["failed_attempts"] == 2


@pytest.mark.parametrize("success", [None, "false", 0, "missing"])
def test_only_success_false_ends_an_attempt(tmp_path: Path, success: Any) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    end = _end("run-a", success=success)
    if success == "missing":
        del end["event"]["success"]
    recorder.process_payload(end)

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert "failed_attempts" not in manifest
    assert run.final_path.is_file() and run.trace_path.is_file()


# --------------------------------------------------------------------------- #
# Ending a waiting run for good
# --------------------------------------------------------------------------- #


def test_the_next_run_of_the_session_ends_a_waiting_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_end("run-a", success=False, duration_ms=1500))

    recorder.process_payload(_start("run-b"))

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["ended_by"] == "next_run_in_session"
    assert manifest["superseded_by_run_id"] == "run-b"
    assert manifest["health_reason"] == "openclaw reported an error: context overflow"
    assert manifest["outcome"] == "error"
    assert manifest["error"] == "context overflow"
    final = json.loads(run.final_path.read_text(encoding="utf-8"))
    assert final["success"] is False
    assert final["sequence"] == _jsonl(run.raw_events_path)[-1]["sequence"]
    trace = load_trace(run.trace_path)
    assert [event.tool for event in trace.events] == ["read"]
    assert trace.metadata["ended_by"] == "next_run_in_session"
    report = _report(run)
    assert report["outcome"] == "completed"
    assert report["duration_seconds"] == pytest.approx(1.5)
    assert any("next run of its session started" in note for note in report["notes"])
    assert _run(tmp_path, "run-b").read_manifest()["status"] == "started"

    # The failed attempt's usage arrives after: post-run spend of the ended run.
    # The report and the resume dossier are derived, and rewritten with the cost.
    derived = {"report.json", "resume.json", "resume.md"}
    files = {name: data for name, data in _files(run).items() if name not in derived}
    recorder.process_payload(_llm("run-a"))
    assert len(_jsonl(run.late_events_path)) == 1
    assert {
        name: data for name, data in _files(run).items() if name not in {*derived, "late.jsonl"}
    } == files
    assert json.loads(run.resume_json_path.read_text())["cost"]["status"] == "observed"
    assert _report(run)["cost"]["post_run_estimated_usd"] == pytest.approx(_CALL_USD)


def test_a_run_of_another_session_leaves_a_waiting_run_open(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a", success=False))

    recorder.process_payload(_start("run-b", session="sess-2"))
    recorder.process_payload(_start("run-a"))  # the waiting run's own next attempt

    assert _run(tmp_path, "run-a").read_manifest()["status"] == "started"


def test_a_duplicated_start_of_a_finished_run_ends_nothing(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-old"))
    recorder.process_payload(_end("run-old", success=True))
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a", success=False))

    recorder.process_payload(_start("run-old"))

    assert _run(tmp_path, "run-a").read_manifest()["status"] == "awaiting_retry"


@pytest.mark.parametrize("durable", [True, False])
def test_the_startup_sweep_interrupts_a_waiting_run_with_its_full_trace(
    tmp_path: Path, durable: bool
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_start("run-a"))
    for index in range(7):
        recorder.process_payload(_tool("run-a", f"a{index}"))
    recorder.process_payload(_end("run-a", success=False))
    recorder.process_payload(_llm("run-a"))
    live = _live_actions(_run(tmp_path, "run-a"))
    if not durable:
        # A run whose SQLite state was lost: the sweep replays it in memory.
        _run(tmp_path, "run-a").update_manifest(state_durable=False)

    OpenClawRunRecorder(tmp_path, config=_PRICED).startup()

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["interrupted_while_awaiting_retry"] is True
    trace = load_trace(run.trace_path)
    assert [event.kind for event in trace.events].count(EventKind.TOOL_CALL) == 7
    assert [event.kind for event in trace.events][-1] is EventKind.LLM_CALL
    assert trace.metadata["interrupted_while_awaiting_retry"] is True
    assert trace.metadata["failed_attempts"] == 1
    assert _replayed_actions(build_pipeline(_PRICED), run) == live
    report = _report(run)
    assert report["outcome"] == "stopped"  # the loop STOP outranks the interruption
    assert report["status"] == "interrupted"
    assert any("waited for another attempt" in note for note in report["notes"])
    assert not any("trace.json is absent" in note for note in report["notes"])


def test_the_startup_sweep_seals_a_run_cut_during_a_later_attempt(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "a1"))
    recorder.process_payload(_end("run-a", success=False))
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "a2"))
    assert _run(tmp_path, "run-a").read_manifest()["status"] == "started"

    OpenClawRunRecorder(tmp_path, config=_PRICED).startup()

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert "interrupted_while_awaiting_retry" not in manifest
    trace = load_trace(run.trace_path)
    assert [event.kind for event in trace.events].count(EventKind.TOOL_CALL) == 2
    assert trace.metadata["failed_attempts"] == 1
    assert "interrupted_while_awaiting_retry" not in trace.metadata
    report = _report(run)
    assert report["status"] == "interrupted"
    assert not any("waited for another attempt" in note for note in report["notes"])
    assert not any("trace.json is absent" in note for note in report["notes"])


def test_a_run_cut_during_a_later_attempt_without_a_trace_reached_an_agent_end(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a", success=False))
    recorder.process_payload(_start("run-a"))
    run = _run(tmp_path, "run-a")
    run.update_manifest(status="interrupted")  # cut, and never sealed

    notes = _report_notes(run)

    assert "trace.json is absent: this run was cut before its final agent_end" in notes
    assert "trace.json is absent: this run never reached agent_end" not in notes


def _report_notes(run: RunDirectory) -> list[str]:
    from pathlayer.adapters.openclaw_report import build_run_report

    return list(build_run_report(run).notes)


def test_a_sweep_that_cannot_write_the_trace_still_interrupts_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a", success=False))
    real_write = recorder_module.atomic_write_json

    def fail_trace(path: Path, payload: Any) -> None:
        if path.name == "trace.json":
            raise OSError("disk full")
        real_write(path, payload)

    monkeypatch.setattr(recorder_module, "atomic_write_json", fail_trace)
    OpenClawRunRecorder(tmp_path).startup()

    run = _run(tmp_path, "run-a")
    assert run.read_manifest()["status"] == "interrupted"
    assert not run.trace_path.exists()
    assert _report(run)["outcome"] == "interrupted"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_a_restarted_listener_resumes_a_waiting_run(tmp_path: Path) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_start("run-a"))
    first.process_payload(_tool("run-a", "c1"))
    failed = _end("run-a", success=False)
    first.process_payload(failed)

    # A second listener on the same directory, without the startup sweep.
    second = OpenClawRunRecorder(tmp_path)
    second.process_payload(_llm("run-a"))
    assert _run(tmp_path, "run-a").read_manifest()["status"] == "awaiting_retry"
    files = _files(_run(tmp_path, "run-a"))
    second.process_payload(json.loads(json.dumps(failed)))  # duplicate, recognized after restart
    assert _files(_run(tmp_path, "run-a")) == files
    second.process_payload(_start("run-a"))
    second.process_payload(_tool("run-a", "c2"))
    second.process_payload(_end("run-a", success=True))

    run = _run(tmp_path, "run-a")
    raw = _jsonl(run.raw_events_path)
    assert [record["sequence"] for record in raw] == list(range(1, len(raw) + 1))
    assert run.read_manifest()["failed_attempts"] == 1
    tools = [
        event for event in load_trace(run.trace_path).events if event.kind is EventKind.TOOL_CALL
    ]
    assert len(tools) == 2


def test_a_restarted_listener_ends_a_waiting_run_when_its_session_moves_on(
    tmp_path: Path,
) -> None:
    first = OpenClawRunRecorder(tmp_path)
    first.process_payload(_start("run-a"))
    first.process_payload(_end("run-a", success=False))

    second = OpenClawRunRecorder(tmp_path)
    second.process_payload(_llm("run-a"))  # reattaches the waiting run
    second.process_payload(_start("run-b"))

    assert _run(tmp_path, "run-a").read_manifest()["ended_by"] == "next_run_in_session"


def test_a_provisional_report_failure_keeps_the_run_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))

    def fail(_run: RunDirectory) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(recorder_module, "write_run_report", fail)
    recorder.process_payload(_end("run-a", success=False))
    run = _run(tmp_path, "run-a")
    assert run.read_manifest()["status"] == "awaiting_retry"
    assert not run.report_path.exists()

    monkeypatch.undo()
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a", success=True))
    assert _report(run)["outcome"] == "completed"


def test_a_manifest_failure_on_resume_leaves_the_run_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    _stop_in_first_attempt(recorder, "run-a")
    run = _run(tmp_path, "run-a")
    journaled = len(_jsonl(run.raw_events_path))
    real_update = RunDirectory.update_manifest

    def fail_once(self: RunDirectory, **fields: Any) -> dict[str, Any]:
        monkeypatch.setattr(RunDirectory, "update_manifest", real_update)
        raise OSError("disk full")

    monkeypatch.setattr(RunDirectory, "update_manifest", fail_once)
    with pytest.raises(OSError, match="disk full"):
        recorder.process_payload(_start("run-a"))

    assert len(_jsonl(run.raw_events_path)) == journaled
    assert run.read_manifest()["status"] == "awaiting_retry"
    assert recorder.tool_call_decision(_query("run-a")) is not None
    recorder.process_payload(_start("run-a"))
    assert run.read_manifest()["status"] == "started"


# --------------------------------------------------------------------------- #
# Duration watchdog
# --------------------------------------------------------------------------- #


def test_a_failed_attempt_pauses_the_duration_watchdog(tmp_path: Path) -> None:
    recorder, clock = _timed_recorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    scheduler = clock.scheduler
    assert [handle.delay for handle in scheduler.live()] == [10]

    clock.advance(3)
    recorder.process_payload(_end("run-a", success=False))
    assert scheduler.live() == []

    # Neither the attempt's usage report nor elapsed time re-arms it: nothing
    # proves the run is still alive.
    clock.advance(60)
    recorder.process_payload(_llm("run-a"))
    assert scheduler.live() == []
    run = _run(tmp_path, "run-a")
    assert _lifecycle_kinds(run) == [EventKind.RUN_START]
    assert run.read_manifest()["stopped"] is False


def test_a_stop_selected_while_the_run_waits_is_written_at_once(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    for index in range(3):
        recorder.process_payload(_tool("run-a", f"a{index}"))
    recorder.process_payload(_end("run-a", success=False))
    run = _run(tmp_path, "run-a")
    assert run.read_manifest()["stopped"] is False
    # The attempt's last calls, delivered after its end, complete the loop.
    for index in range(3, 8):
        recorder.process_payload(_tool("run-a", f"a{index}"))

    manifest = run.read_manifest()
    assert manifest["status"] == "awaiting_retry"
    assert manifest["stopped"] is True
    assert _report(run)["outcome"] == "stopped"


def test_a_tool_call_arriving_after_the_failed_end_does_not_resume_the_run(
    tmp_path: Path,
) -> None:
    # The plugin posts hooks without ordering them: an aborted attempt's last
    # tool call can reach Pathlayer after its failed agent_end.
    recorder, clock = _timed_recorder(tmp_path)
    scheduler = clock.scheduler
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_end("run-a", success=False))
    recorder.process_payload(_tool("run-a", "c2"))

    run = _run(tmp_path, "run-a")
    assert run.read_manifest()["status"] == "awaiting_retry"
    assert [record["hook"] for record in _jsonl(run.raw_events_path)][-1] == "after_tool_call"
    assert scheduler.live() == []
    # No deadline can fire on a run that has ended, and the next run of the
    # session still closes it.
    clock.advance(60)
    recorder.process_payload(_start("run-b"))
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["ended_by"] == "next_run_in_session"
    assert manifest["stopped"] is False
    tools = [e for e in load_trace(run.trace_path).events if e.kind is EventKind.TOOL_CALL]
    assert len(tools) == 2


def _lifecycle_kinds(run: RunDirectory) -> list[EventKind]:
    """The run boundaries Pathlayer journaled itself (start, timeout)."""
    return [
        EventKind(record["payload"]["event"]["kind"])
        for record in _jsonl(run.raw_events_path)
        if record["payload"].get("hook") == "pathlayer_lifecycle"
    ]


def test_the_next_attempt_rearms_the_original_deadline(tmp_path: Path) -> None:
    recorder, clock = _timed_recorder(tmp_path, enforce=True)
    recorder.process_payload(_start("run-a"))
    clock.advance(3)
    recorder.process_payload(_end("run-a", success=False))
    clock.advance(3)

    recorder.process_payload(_start("run-a"))

    (handle,) = clock.scheduler.live()
    assert handle.delay == pytest.approx(4)  # 10 s from the first start, not from now
    clock.advance(4)
    handle.callback()
    assert recorder.tool_call_decision(_query("run-a")) is not None
    run = _run(tmp_path, "run-a")
    assert _lifecycle_kinds(run) == [EventKind.RUN_START, EventKind.RUN_TIMEOUT]


def test_an_overdue_deadline_fires_when_the_next_attempt_starts(tmp_path: Path) -> None:
    recorder, clock = _timed_recorder(tmp_path, enforce=True)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a", success=False))
    clock.advance(120)

    recorder.process_payload(_start("run-a"))

    assert recorder.tool_call_decision(_query("run-a")) is not None
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_end("run-a", success=True))
    run = _run(tmp_path, "run-a")
    trace = load_trace(run.trace_path)
    assert [event.kind for event in trace.events] == [
        EventKind.RUN_START,
        EventKind.RUN_TIMEOUT,
        EventKind.TOOL_CALL,
    ]
    assert _jsonl(run.decisions_path)[-1]["analyzed"] is False


def test_a_provisional_report_ignores_an_older_timeout_snapshot(tmp_path: Path) -> None:
    recorder, clock = _timed_recorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    clock.advance(10)
    (handle,) = clock.scheduler.live()
    handle.callback()  # timeout: STOP and a trace.json snapshot
    run = _run(tmp_path, "run-a")
    assert len(load_trace(run.trace_path).events) == 3
    recorder.process_payload(_tool("run-a", "c2"))  # recorded past the STOP
    recorder.process_payload(_end("run-a", success=False))

    report = _report(run)
    assert report["events_recorded"] == 4
    assert report["outcome"] == "stopped"
    assert report["status"] == "awaiting_retry"


# --------------------------------------------------------------------------- #
# Report outcome
# --------------------------------------------------------------------------- #


def test_awaiting_retry_is_the_only_in_progress_status() -> None:
    def outcome(status: str, *, stopped: bool = False) -> SessionOutcome:
        return build_session_report(
            run_id="r", status=status, ticks=[], events_recorded=0, stopped=stopped
        ).outcome

    assert outcome("awaiting_retry") is SessionOutcome.IN_PROGRESS
    assert outcome("awaiting_retry", stopped=True) is SessionOutcome.STOPPED
    # A report cannot tell a live run from one whose listener died.
    assert outcome("started") is SessionOutcome.INTERRUPTED
    assert outcome("completed") is SessionOutcome.COMPLETED


def test_a_failure_closing_a_waiting_run_never_costs_the_next_run_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a", success=False))
    real_write = recorder_module.atomic_write_json

    def fail_final(path: Path, payload: Any) -> None:
        if path.name == "final.json":
            raise OSError("disk full")
        real_write(path, payload)

    monkeypatch.setattr(recorder_module, "atomic_write_json", fail_final)
    recorder.process_payload(_start("run-b"))

    assert _run(tmp_path, "run-b").request_path.is_file()
    assert _run(tmp_path, "run-a").read_manifest()["status"] == "awaiting_retry"
    monkeypatch.undo()
    recorder.process_payload(_start("run-c"))
    assert _run(tmp_path, "run-a").read_manifest()["superseded_by_run_id"] == "run-c"


def test_a_failed_end_without_a_run_id_ends_its_orphan_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    end = _end("ignored", success=False)
    del end["ctx"]["runId"]
    recorder.process_payload(end)

    (orphan,) = RunStore(tmp_path).list_runs()
    assert orphan.original_run_id.startswith("orphan-")
    assert orphan.read_manifest()["status"] == "completed"


def test_a_compaction_that_rereads_the_same_files_is_not_stopped(tmp_path: Path) -> None:
    # Healthy counterexample: after a compaction the agent has lost its context
    # and re-reads what it had read, then moves on. Stitching the attempts into
    # one pipeline must not turn that re-read into a loop STOP.
    def read(run_id: str, path: str, call_id: str) -> dict[str, Any]:
        payload = _tool(run_id, call_id)
        payload["event"]["params"] = {"path": path}
        payload["event"]["result"] = f"contents of {path}"
        return payload

    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    for attempt in range(3):
        if attempt:
            recorder.process_payload(_start("run-a", prompt="compacted"))
        for name in ("a.py", "b.py", "c.py"):
            recorder.process_payload(read("run-a", name, f"{attempt}-{name}"))
        if attempt < 2:
            # Each real attempt reports its own duration and messages.
            recorder.process_payload(_end("run-a", success=False, duration_ms=1000 + attempt))
    edit = _tool("run-a", "edit")
    edit["event"].update(toolName="edit", params={"path": "a.py"}, result="patched")
    recorder.process_payload(edit)
    recorder.process_payload(_end("run-a", success=True))

    run = _run(tmp_path, "run-a")
    assert run.read_manifest()["failed_attempts"] == 2
    assert "STOP" not in _live_actions(run)
    assert run.read_manifest()["stopped"] is False


# --------------------------------------------------------------------------- #
# Attempts that end in error without success=false (2026-11-18 retest)
# --------------------------------------------------------------------------- #


def _error_end(
    run_id: str, *, error_message: str | None = "Context overflow", session: str = "sess-1"
) -> dict[str, Any]:
    """An ``agent_end`` after a provider error: ``success`` stays true.

    agent-core turns a provider error into an assistant message with
    ``stopReason: "error"`` instead of raising, so OpenClaw computes
    ``success = !aborted && !promptError`` as true.
    """
    last: dict[str, Any] = {"role": "assistant", "content": [], "stopReason": "error"}
    if error_message is not None:
        last["errorMessage"] = error_message
    return {
        "hook": "agent_end",
        "event": {
            "success": True,
            "durationMs": 1000,
            "messages": [{"role": "user", "content": "go"}, last],
        },
        "ctx": _ctx(run_id, session),
    }


def test_a_provider_error_ends_an_attempt_not_the_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_error_end("run-a"))

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "awaiting_retry"
    assert manifest["failed_attempts"] == 1
    assert not run.final_path.exists()
    assert any(
        note.startswith(
            "provisional report: attempt 1 of this run did not succeed (Context overflow)"
        )
        for note in _report(run)["notes"]
    )

    # OpenClaw compacts and retries under the same runId.
    recorder.process_payload(_start("run-a", prompt="compacted"))
    recorder.process_payload(_tool("run-a", "c2"))
    recorder.process_payload(_end("run-a", success=True))

    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["failed_attempts"] == 1
    assert manifest["outcome"] == "ok"
    assert "error" not in manifest
    assert [event.tool for event in load_trace(run.trace_path).events] == ["read", "read"]


def test_a_provider_error_on_the_last_attempt_is_the_run_outcome(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_error_end("run-a"))
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_error_end("run-a"))
    # OpenClaw gave up and told the user, with no further hook: the next run of
    # the session is what proves the run is over.
    recorder.process_payload(_start("run-b"))

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["outcome"] == "error"
    assert manifest["error"] == "Context overflow"
    assert manifest["failed_attempts"] == 2
    assert manifest["health_reason"] == "openclaw reported an error: Context overflow"
    trace = load_trace(run.trace_path)
    assert trace.metadata["outcome"] == "error"
    report = _report(run)
    assert report["outcome"] == "completed"
    assert any(
        note.startswith("OpenClaw reported an error on this run's last attempt: Context overflow")
        for note in report["notes"]
    )


def test_a_precheck_overflow_ends_in_error(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    end = _end("run-a", success=False)
    end["event"]["error"] = "Context overflow: prompt too large for the model (precheck)."
    recorder.process_payload(end)
    recorder.process_payload(_start("run-b"))

    manifest = _run(tmp_path, "run-a").read_manifest()
    assert manifest["outcome"] == "error"
    assert manifest["error"] == "Context overflow: prompt too large for the model (precheck)."


def test_a_provider_error_without_a_message_still_ends_the_attempt(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_error_end("run-a", error_message=None))
    recorder.process_payload(_start("run-b"))

    manifest = _run(tmp_path, "run-a").read_manifest()
    assert manifest["outcome"] == "error"
    assert manifest["error"] == "the provider ended the reply with an error"


@pytest.mark.parametrize("stop_reason", ["stop", "toolUse", "aborted", "length", None])
def test_a_reply_that_did_not_end_in_error_ends_the_run(
    tmp_path: Path, stop_reason: str | None
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    end = _end("run-a", success=True)
    end["event"]["messages"] = [{"role": "assistant", "content": [], "stopReason": stop_reason}]
    recorder.process_payload(end)

    manifest = _run(tmp_path, "run-a").read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["outcome"] == "ok"
    assert "failed_attempts" not in manifest


def test_an_earlier_assistant_error_does_not_fail_the_attempt(tmp_path: Path) -> None:
    # Only the reply that ended the attempt counts: an error the agent recovered
    # from earlier in the session history says nothing about this attempt.
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    end = _end("run-a", success=True)
    end["event"]["messages"] = [
        {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "overloaded"},
        {"role": "user", "content": "retry"},
        {"role": "assistant", "content": [], "stopReason": "stop"},
    ]
    recorder.process_payload(end)

    assert _run(tmp_path, "run-a").read_manifest()["outcome"] == "ok"


def test_a_stopped_run_reports_the_stop_as_its_outcome(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.process_payload(_start("run-a"))
    _stop_in_first_attempt(recorder, "run-a")
    recorder.process_payload(_end("run-a", success=True))

    assert _run(tmp_path, "run-a").read_manifest()["outcome"] == "stopped"


# --------------------------------------------------------------------------- #
# Compaction hooks
# --------------------------------------------------------------------------- #


def _compaction(hook: str, *, session: str = "sess-1") -> dict[str, Any]:
    # OpenClaw sends compaction hooks with the session key only, no runId.
    return {
        "hook": hook,
        "event": {"messageCount": 63, "compactedCount": 1},
        "ctx": {"sessionKey": session},
    }


def test_compaction_hooks_are_journaled_in_the_sessions_run_and_never_analyzed(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_error_end("run-a"))
    recorder.process_payload(_compaction("before_compaction"))
    recorder.process_payload(_compaction("after_compaction"))
    run = _run(tmp_path, "run-a")
    # A compaction is no new attempt: the run keeps waiting.
    assert run.read_manifest()["status"] == "awaiting_retry"
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c2"))
    recorder.process_payload(_end("run-a", success=True))

    hooks = [record["payload"]["hook"] for record in _jsonl(run.raw_events_path)]
    assert hooks.count("before_compaction") == 1
    assert hooks.count("after_compaction") == 1
    assert [line["event_index"] for line in _jsonl(run.decisions_path)] == [0, 1]
    assert [event.tool for event in load_trace(run.trace_path).events] == ["read", "read"]
    assert [other.original_run_id for other in RunStore(tmp_path).list_runs()] == ["run-a"]


def test_a_compaction_hook_without_a_known_run_is_dropped(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_compaction("before_compaction"))
    recorder.process_payload(_start("run-a", session="other"))
    recorder.process_payload(_compaction("after_compaction", session="sess-1"))

    assert [run.original_run_id for run in RunStore(tmp_path).list_runs()] == ["run-a"]
    hooks = [
        record["payload"]["hook"] for record in _jsonl(_run(tmp_path, "run-a").raw_events_path)
    ]
    assert hooks == ["before_agent_run"]


def test_a_compaction_hook_is_not_attributed_when_the_session_has_two_open_runs(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    # Not supposed to happen (one run at a time per session), so no guess.
    recorder.process_payload(_tool("run-x", "c1"))
    recorder.process_payload(_compaction("before_compaction"))

    for run_id in ("run-a", "run-x"):
        hooks = [r["payload"]["hook"] for r in _jsonl(_run(tmp_path, run_id).raw_events_path)]
        assert "before_compaction" not in hooks


def test_an_attempt_error_is_truncated() -> None:
    from pathlayer.adapters.openclaw_lifecycle import (
        MAX_ATTEMPT_ERROR_CHARS,
        attempt_end_error,
        attempt_end_failed,
    )

    long_error = "x" * (MAX_ATTEMPT_ERROR_CHARS + 50)
    text = attempt_end_error({"success": False, "error": long_error})
    assert text is not None and len(text) == MAX_ATTEMPT_ERROR_CHARS and text.endswith("...")
    assert attempt_end_error({"success": False, "error": "  "}) is None
    assert attempt_end_failed({"success": False, "error": "  "})
    assert not attempt_end_failed("not an event")
    assert attempt_end_error(["not", "an", "event"]) is None
    assert not attempt_end_failed({"success": True, "messages": "not a list"})
    assert not attempt_end_failed({"success": True, "messages": [{"role": "user"}]})


def test_a_user_abort_is_not_reported_as_an_error(tmp_path: Path) -> None:
    # OpenClaw reports an abort with success=false and no error, on purpose: it
    # is not an errored completion.
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    end = _end("run-a", success=False)
    del end["event"]["error"]
    end["event"]["messages"] = [{"role": "assistant", "content": [], "stopReason": "aborted"}]
    recorder.process_payload(end)
    recorder.process_payload(_start("run-b"))

    manifest = _run(tmp_path, "run-a").read_manifest()
    assert manifest["outcome"] == "unknown"
    assert "error" not in manifest
    assert "did not report success" in manifest["health_reason"]


def test_an_error_before_any_tool_call_is_named(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_error_end("run-a"))
    recorder.process_payload(_start("run-b"))

    manifest = _run(tmp_path, "run-a").read_manifest()
    assert manifest["health_reason"] == "openclaw reported an error: Context overflow"


def test_the_sweep_keeps_the_error_of_a_run_left_waiting(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    recorder.process_payload(_error_end("run-a"))

    OpenClawRunRecorder(tmp_path).startup()

    run = _run(tmp_path, "run-a")
    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest["error"] == "Context overflow"
    trace = load_trace(run.trace_path)
    assert trace.metadata["health_reason"] == "openclaw reported an error: Context overflow"


def test_a_compaction_reaches_a_run_restored_after_a_restart(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_tool("run-a", "c1"))
    restarted = OpenClawRunRecorder(tmp_path)
    # A hook of the run reattaches it in the new process, with its session.
    restarted.process_payload(_tool("run-a", "c2"))
    restarted.process_payload(_compaction("before_compaction"))

    hooks = [r["payload"]["hook"] for r in _jsonl(_run(tmp_path, "run-a").raw_events_path)]
    assert hooks[-1] == "before_compaction"
