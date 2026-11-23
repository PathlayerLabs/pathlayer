"""The plain-language OpenClaw run report (``pathlayer-report <run>``).

One status line, clock times instead of event positions, the calls Pathlayer
blocked said as such, and the internal vocabulary kept in the technical
appendix.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from pathlayer.adapters.openclaw_readable import (
    display_status,
    event_times,
    first_stop,
    render_run_text,
    report_payload,
    tool_results_after,
)
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_report import build_run_report, main
from pathlayer.adapters.openclaw_run_store import RunDirectory, RunStore
from pathlayer.policies import Action
from pathlayer.postmortem import (
    BlockedToolCall,
    SessionReport,
    TickRecord,
    ToolCallGate,
    build_session_report,
)

FIXTURES = Path(__file__).parent / "fixtures" / "openclaw_report"

# Words a non-developer should never have to read outside the appendix.
INTERNAL_WORDS = ("tick", "fused", "peak", "wired", "arbitrat", "confidence", "t=")


def _ctx(run_id: str) -> dict[str, Any]:
    return {"runId": run_id, "sessionKey": "sess-1", "agentId": "agent-1"}


def _start(run_id: str) -> dict[str, Any]:
    return {"hook": "before_agent_run", "event": {"prompt": "fix the bug"}, "ctx": _ctx(run_id)}


def _read(run_id: str, index: int) -> dict[str, Any]:
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


def _end(run_id: str, **event: Any) -> dict[str, Any]:
    body = {"success": True, "durationMs": 61_000, **event}
    return {"hook": "agent_end", "event": body, "ctx": _ctx(run_id)}


def _query(run_id: str, call: str = "next") -> dict[str, Any]:
    return {
        "hook": "before_tool_call",
        "event": {"toolName": "exec", "toolCallId": call},
        "ctx": _ctx(run_id),
    }


def _run(root: Path, run_id: str) -> RunDirectory:
    run = RunStore(root).resolve_run(run_id)
    assert run is not None
    return run


def _stopped_run(root: Path, *, enforce: bool = True, run_id: str = "run-a") -> RunDirectory:
    """A run the loop detector stops, with tool calls asked about after the STOP."""
    recorder = OpenClawRunRecorder(root, enforce=enforce)
    recorder.startup()
    recorder.process_payload(_start(run_id))
    for index in range(8):
        recorder.process_payload(_read(run_id, index))
        recorder.tool_call_decision(_query(run_id, f"next-{index}"))
    recorder.process_payload(_end(run_id))
    return _run(root, run_id)


def _pin_times(run: RunDirectory) -> None:
    """Replace every wall-clock stamp with a fixed one, so the text is stable."""
    run.update_manifest(
        started_at="2026-11-20T17:00:00+00:00", ended_at="2026-11-20T17:01:01+00:00"
    )
    for path, key in ((run.decisions_path, "received_at"), (run.gate_path, "answered_at")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for offset, record in enumerate(records):
            record[key] = f"2026-11-20T17:00:{10 + offset:02d}+00:00"
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _report(status: str = "completed", **kwargs: Any) -> SessionReport:
    return build_session_report(run_id="r", status=status, ticks=[], events_recorded=0, **kwargs)


# --------------------------------------------------------------------------- #
# One status line
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ({"status": "completed", "outcome": "ok"}, "Finished (OpenClaw reported success)"),
        (
            {"status": "completed", "outcome": "error", "error": "Context overflow: too big"},
            "Ended with an error reported by OpenClaw: Context overflow: too big",
        ),
        (
            {"status": "completed", "outcome": "unknown"},
            "Ended without success, and without an error (for example, aborted)",
        ),
        ({"status": "completed"}, "Finished"),
        (
            {"status": "awaiting_retry", "failed_attempts": 1},
            "Waiting for a retry: attempt 1 did not succeed",
        ),
        (
            {"status": "interrupted"},
            "Interrupted: the run never reached its end (the listener stopped or restarted)",
        ),
        (
            {"status": "interrupted", "interrupted_while_awaiting_retry": True},
            "Interrupted while waiting for a retry: the listener restarted",
        ),
        (
            {"status": "started"},
            "No end recorded: the run may still be running, or its listener stopped",
        ),
        (
            {"status": "completed", "outcome": "ok", "failed_attempts": 2},
            "Finished (OpenClaw reported success) after 2 failed attempt(s)",
        ),
    ],
)
def test_one_status_line_per_situation(manifest: dict[str, Any], expected: str) -> None:
    assert display_status(manifest, _report(str(manifest["status"]))) == expected


def test_a_stop_says_whether_openclaw_was_told_to_enforce_it() -> None:
    stopped = {"status": "completed", "outcome": "stopped", "stopped": True}
    enforced = ToolCallGate(queries=3, enforced=True)
    passive = ToolCallGate(queries=3, enforced=False)

    report = _report(stopped=True)
    report.tool_call_gate = enforced
    assert display_status(stopped, report) == (
        "Stop decided by Pathlayer (enforced, but no later tool call was attempted)"
    )
    report.tool_call_gate = ToolCallGate(
        queries=3, enforced=True, blocked=(BlockedToolCall("exec", "c3", None, "loop: 5x"),)
    )
    assert display_status(stopped, report) == "Stopped by Pathlayer"
    report.tool_call_gate = passive
    assert display_status(stopped, report) == (
        "Stop decided by Pathlayer (not enforced: recorded only)"
    )
    report.tool_call_gate = None
    assert display_status(stopped, report) == (
        "Stop decided by Pathlayer (enforcement unknown: no gate log)"
    )


def test_a_stopped_run_that_never_ended_says_its_record_is_incomplete() -> None:
    manifest = {"status": "interrupted", "stopped": True}
    assert display_status(manifest, _report("interrupted", stopped=True)) == (
        "Stop decided by Pathlayer (enforcement unknown: no gate log) -- record incomplete: "
        "the run never reached its end"
    )


def test_a_long_error_is_clipped_on_the_status_line() -> None:
    manifest = {"status": "completed", "outcome": "error", "error": "x" * 500}
    line = display_status(manifest, _report())
    assert line.endswith("...")
    assert len(line) < 200


# --------------------------------------------------------------------------- #
# Clock times, never event positions
# --------------------------------------------------------------------------- #


def test_actions_and_blocked_calls_show_the_time_they_were_received(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    _pin_times(run)
    report = build_run_report(run)
    stop = next(action for action in report.actions if action.action is Action.STOP)
    times = event_times(run)

    text = render_run_text(run, report)
    stop_clock = times[stop.index][11:19]
    assert f"  {stop_clock} UTC  Pathlayer decided to stop the run: " in text
    assert "the agent kept repeating the same tool calls (loop:" in text
    assert f"event {stop.index}" not in text
    assert "Started: 2026-11-20 17:00:00 UTC   Ended: 17:01:01 UTC   (1m01s)" in text


def test_a_time_on_another_day_than_the_start_carries_its_date(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    _pin_times(run)
    run.update_manifest(ended_at="2026-11-21T01:02:03+00:00")

    text = render_run_text(run, build_run_report(run))
    assert "Ended: 2026-11-21 01:02:03 UTC" in text


def test_report_json_gains_the_status_and_the_times_and_keeps_every_field(
    tmp_path: Path,
) -> None:
    run = _stopped_run(tmp_path)
    report = build_run_report(run)
    payload = json.loads(run.report_path.read_text(encoding="utf-8"))

    assert set(report.to_dict()) <= set(payload)
    assert payload["display_status"] == "Stopped by Pathlayer"
    times = event_times(run)
    for action in payload["actions"]:
        assert action["at"] == times[action["index"]]
    for finding in payload["findings"]:
        assert finding["first_at"] == times[finding["first_index"]]
        assert finding["last_at"] == times[finding["last_index"]]
    assert payload == report_payload(run, report)


def test_a_corrupt_decision_journal_costs_the_times_only(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    report = build_run_report(run)
    run.decisions_path.write_text("not json\n{}\n", encoding="utf-8")

    assert event_times(run) == {}
    assert "time unknown  Pathlayer decided to stop the run" in render_run_text(run, report)


# --------------------------------------------------------------------------- #
# What Pathlayer did, said explicitly
# --------------------------------------------------------------------------- #


def test_blocked_calls_are_named_with_their_tool_and_time(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    _pin_times(run)
    report = build_run_report(run)
    gate = report.tool_call_gate
    assert gate is not None and gate.blocked

    text = render_run_text(run, report)
    assert (
        f"Calls blocked by Pathlayer: {len(gate.blocked)} of {gate.queries} tool call(s) "
        "OpenClaw asked about"
    ) in text
    first = gate.blocked[0]
    assert first.at is not None
    assert f"  {first.at[11:19]} UTC  exec -- the run was stopped (loop:" in text


def test_a_passive_recording_says_nothing_was_blocked(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path, enforce=False)
    text = render_run_text(run, build_run_report(run))

    assert "Status: Stop decided by Pathlayer (not enforced: recorded only)" in text
    assert "not enforced: recorded only, OpenClaw ran every call" in text
    assert "call(s) ran that enforcement would have blocked" in text


def test_a_call_refused_by_the_in_flight_cap_says_why(tmp_path: Path) -> None:
    report = _report()
    report.tool_call_gate = ToolCallGate(
        queries=5,
        enforced=True,
        blocked=(
            BlockedToolCall(
                tool_name="exec",
                tool_call_id="c5",
                at="2026-11-20T17:00:05+00:00",
                reason="identical calls in flight: 4 exec call(s) with the same arguments",
            ),
        ),
    )
    run = _stopped_run(tmp_path)
    text = render_run_text(run, report)
    assert "exec -- the same call with the same arguments was already running" in text


def test_lighter_actions_are_said_to_be_recorded_only(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    ticks = [
        TickRecord(index=0, ts=0.0, fused=0.6, action=Action.WARN, reason="loop: 3x repeat"),
        TickRecord(
            index=1, ts=1.0, fused=0.95, action=Action.STOP, reason="loop: 5x", dry_run=True
        ),
    ]
    report = build_session_report(run_id="r", status="completed", ticks=ticks, events_recorded=2)
    text = render_run_text(run, report)

    assert "Pathlayer raised a warning (recorded only: OpenClaw has no channel for it)" in text
    assert "Pathlayer would have stopped the run (observe-only mode, never applied)" in text


def test_a_quiet_run_says_pathlayer_did_nothing(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-q"))
    recorder.process_payload(_read("run-q", 0))
    recorder.process_payload(_end("run-q"))
    run = _run(tmp_path, "run-q")

    text = render_run_text(run, build_run_report(run))
    assert "Status: Finished (OpenClaw reported success)" in text
    assert "  nothing: it analyzed 1 event(s) (tool calls and model replies) and never" in text


# --------------------------------------------------------------------------- #
# Internal vocabulary stays in the appendix
# --------------------------------------------------------------------------- #


def test_internal_vocabulary_only_appears_in_the_technical_appendix(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    report = build_run_report(run)

    plain = render_run_text(run, report).lower()
    for word in INTERNAL_WORDS:
        assert word not in plain, word
    technical = render_run_text(run, report, technical=True)
    head, _, appendix = technical.partition("\nTechnical details\n")
    assert head.lower().rstrip() == plain.rsplit("\n\ndetector readings:", 1)[0]
    assert appendix == report.format()
    assert "fused score" in appendix


def test_reference_rendering_of_a_stopped_run(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    _pin_times(run)
    text = render_run_text(run, build_run_report(run))
    expected = (FIXTURES / "stop_loop.txt").read_text(encoding="utf-8")
    assert text + "\n" == expected


def _result(run_id: str, call: str, result: Any) -> dict[str, Any]:
    event = {"toolName": "exec", "params": {"command": "ls"}, "toolCallId": call}
    return {
        "hook": "after_tool_call",
        "event": {**event, "result": result, "runId": run_id},
        "ctx": _ctx(run_id),
    }


def test_calls_that_ran_after_the_stop_are_counted_never_the_blocked_ones(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    recorder.process_payload(_start("run-a"))
    for index in range(5):
        recorder.process_payload(_read("run-a", index))
    # Blocked by the gate: its failed result never ran.
    assert recorder.tool_call_decision(_query("run-a", "gated"))["block"] is True
    recorder.process_payload(_result("run-a", "gated", {"error": "blocked"}))
    # Denied by OpenClaw itself.
    denied = {"details": {"status": "blocked", "deniedReason": "approval"}}
    recorder.process_payload(_result("run-a", "denied", denied))
    # Already running when Pathlayer decided: it ran.
    recorder.process_payload(_result("run-a", "running", "ok"))
    recorder.process_payload(_end("run-a"))
    run = _run(tmp_path, "run-a")

    sequence, _ = first_stop(run)
    assert tool_results_after(run, sequence) == 1
    text = render_run_text(run, build_run_report(run))
    assert "after the stop decision, 1 more tool call(s) ran (already running" in text


def test_no_stop_means_no_count_of_calls_after_it(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.startup()
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_read("run-a", 0))
    recorder.process_payload(_end("run-a"))
    run = _run(tmp_path, "run-a")

    assert first_stop(run) == (None, None)
    assert tool_results_after(run, None) is None
    assert "after the stop decision" not in render_run_text(run, build_run_report(run))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_prints_the_plain_report_and_the_appendix_on_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stopped_run(tmp_path)

    assert main(["run-a", "--runs-root", str(tmp_path)]) == 0
    plain = capsys.readouterr().out
    assert plain.startswith("Run run-a (session sess-1, agent agent-1)\nStatus: Stopped by")
    assert "Technical details" not in plain
    assert "Detector readings: pathlayer-report run-a --technical" in plain

    assert main(["run-a", "--technical", "--runs-root", str(tmp_path)]) == 0
    technical = capsys.readouterr().out
    assert "Technical details\nrun run-a" in technical
    assert "outcome: stopped" in technical


def test_cli_json_carries_the_status_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stopped_run(tmp_path)
    assert main(["run-a", "--json", "--runs-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["display_status"] == "Stopped by Pathlayer"
    assert payload["outcome"] == "stopped"


def test_cli_listing_ends_each_line_with_the_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stopped_run(tmp_path, run_id="run-a")
    _stopped_run(tmp_path, run_id="run-b", enforce=False)
    assert main(["--all", "--runs-root", str(tmp_path)]) == 0
    lines = sorted(capsys.readouterr().out.strip().splitlines(), key=lambda x: x.split()[1])
    assert re.match(r"\S+  run-a  \d+ event\(s\)  cost n/a  Stopped by Pathlayer$", lines[0])
    assert lines[1].endswith("  Stop decided by Pathlayer (not enforced: recorded only)")


def test_notes_about_detector_internals_stay_in_the_appendix(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    plain_note = "trace.json is absent: this run never reached agent_end"
    technical_note = "they carry no fused score and no detector confidence except on a STOP"
    report = _report(notes=[plain_note, technical_note])

    text = render_run_text(run, report)
    assert f"  - {plain_note}" in text
    assert technical_note not in text
    assert f"note: {technical_note}" in render_run_text(run, report, technical=True)


def test_a_long_reason_is_clipped_in_plain_words(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    long_reason = "retry: 3x identical failure of 'exec' (error " + "e" * 400 + ")"
    ticks = [TickRecord(index=0, ts=0.0, fused=1.0, action=Action.STOP, reason=long_reason)]
    report = build_session_report(
        run_id="r", status="completed", ticks=ticks, events_recorded=1, stopped=True
    )
    (line,) = [x for x in render_run_text(run, report).splitlines() if "decided to stop" in x]
    assert "the same tool call kept failing the same way (retry: 3x identical" in line
    assert line.endswith("...)")
    assert len(line) < 250
    assert long_reason in render_run_text(run, report, technical=True)


def test_a_stop_while_waiting_for_a_retry_keeps_its_wording() -> None:
    manifest = {"status": "awaiting_retry", "failed_attempts": 1, "stopped": True}
    report = _report("awaiting_retry", stopped=True)
    report.tool_call_gate = ToolCallGate(queries=1, enforced=False)
    assert display_status(manifest, report) == (
        "Waiting for a retry: attempt 1 did not succeed; "
        "stop decided by Pathlayer (not enforced: recorded only)"
    )


def test_calls_refused_by_the_in_flight_cap_are_an_intervention(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    recorder.process_payload(_start("run-cap"))
    recorder.process_payload(_read("run-cap", 0))
    for index in range(6):
        query = _query("run-cap", f"same-{index}")
        query["event"]["paramsDigest"] = "d" * 64
        recorder.tool_call_decision(query)
    recorder.process_payload(_end("run-cap"))
    run = _run(tmp_path, "run-cap")
    report = build_run_report(run)
    assert report.actions == ()
    assert report.tool_call_gate is not None and report.tool_call_gate.blocked

    text = render_run_text(run, report)
    assert "nothing:" not in text
    assert "no detector asked for an action; the tool-call gate blocked calls (below)" in text
    assert "exec -- the same call with the same arguments was already running" in text


def test_a_non_finite_event_index_is_skipped(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    with run.decisions_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_index": Infinity, "received_at": "x", "analyzed": false}\n')
        handle.write('{"event_index": NaN, "received_at": "x", "analyzed": false}\n')
    assert "x" not in event_times(run).values()


def test_a_time_with_another_offset_is_shown_in_utc(tmp_path: Path) -> None:
    run = _stopped_run(tmp_path)
    _pin_times(run)
    run.update_manifest(ended_at="2026-11-20T18:01:01+01:00")
    assert "Ended: 17:01:01 UTC" in render_run_text(run, build_run_report(run))


def test_cli_refuses_technical_outside_a_single_text_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stopped_run(tmp_path)
    for extra in (["--all"], ["--today"], ["--last", "--json"]):
        with pytest.raises(SystemExit):
            main([*extra, "--technical", "--runs-root", str(tmp_path)])
    assert "--technical applies to the text report of one run" in capsys.readouterr().err
