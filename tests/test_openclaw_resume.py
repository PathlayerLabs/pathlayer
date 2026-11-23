"""The OpenClaw resume dossier: ``resume.json`` / ``resume.md`` after a bad run."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import pytest

from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_report import llm_cost_split, main
from pathlayer.adapters.openclaw_resume import (
    DECIDED,
    OBSERVED,
    PROVEN,
    SCHEMA,
    UNKNOWN,
    _merged_entries,
    _template_key,
    _why_stopped,
    build_resume_dossier,
    mask_text,
    render_resume_markdown,
    template_texts,
)
from pathlayer.adapters.openclaw_run_store import RunDirectory, RunStore
from pathlayer.policies import Action
from pathlayer.postmortem import BlockedToolCall, TickRecord, ToolCallGate, build_session_report

FIXTURES = Path(__file__).parent / "fixtures"
SECTIONS = (
    "identity",
    "why_stopped",
    "attempts",
    "last_clean_call",
    "proof_ledger",
    "pathlayer_actions",
    "cost",
    "options",
    "unknowns",
)
BANNED = ("impossible", "good state", "guaranteed", "safe to")
DIGEST = "d" * 64


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
            "result": {"content": [{"type": "text", "text": "print(1)"}]},
            "toolCallId": f"call-{index}",
            "runId": run_id,
        },
        "ctx": _ctx(run_id),
    }


def _write(run_id: str, call: str, content: str) -> dict[str, Any]:
    return {
        "hook": "after_tool_call",
        "event": {
            "toolName": "write",
            "params": {"path": "notes.md", "content": content},
            "result": {
                "content": [{"type": "text", "text": "Successfully wrote 9 bytes to notes.md"}]
            },
            "toolCallId": call,
            "runId": run_id,
        },
        "ctx": _ctx(run_id),
    }


def _query(run_id: str, call: str, tool: str = "read") -> dict[str, Any]:
    return {
        "hook": "before_tool_call",
        "event": {"toolName": tool, "toolCallId": call, "paramsDigest": DIGEST},
        "ctx": _ctx(run_id),
    }


def _end(run_id: str) -> dict[str, Any]:
    return {
        "hook": "agent_end",
        "event": {"success": True, "durationMs": 1000},
        "ctx": _ctx(run_id),
    }


def _error_end(run_id: str, message: str = "Context overflow") -> dict[str, Any]:
    last = {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": message}
    return {
        "hook": "agent_end",
        "event": {"success": True, "durationMs": 1000, "messages": [last]},
        "ctx": _ctx(run_id),
    }


def _run(root: Path, run_id: str) -> RunDirectory:
    run = RunStore(root).resolve_run(run_id)
    assert run is not None
    return run


def _loop(recorder: OpenClawRunRecorder, run_id: str, *, end: bool = True) -> RunDirectory:
    """Identical reads until the loop detector stops the run; each asked about first."""
    recorder.process_payload(_start(run_id))
    for index in range(8):
        recorder.tool_call_decision(_query(run_id, f"call-{index}"))
        recorder.process_payload(_read(run_id, index))
    if end:
        recorder.process_payload(_end(run_id))
    return _run(recorder.runs_root, run_id)


def _stored(run: RunDirectory) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(run.resume_json_path.read_text(encoding="utf-8"))
    return payload


# --------------------------------------------------------------------------- #
# When a dossier exists
# --------------------------------------------------------------------------- #


def test_a_healthy_run_gets_no_dossier(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.process_payload(_start("ok"))
    recorder.process_payload(_read("ok", 0))
    recorder.process_payload(_end("ok"))
    run = _run(tmp_path, "ok")

    assert build_resume_dossier(run) is None
    assert not run.resume_json_path.exists()
    assert not run.resume_md_path.exists()


def test_a_loop_stop_writes_a_dossier_whose_stop_is_proven(tmp_path: Path) -> None:
    run = _loop(OpenClawRunRecorder(tmp_path, enforce=True), "run-a")
    dossier = _stored(run)

    assert dossier["schema"] == SCHEMA
    assert dossier["provisional"] is False
    assert all(dossier[name]["status"] in {OBSERVED, DECIDED, PROVEN, UNKNOWN} for name in SECTIONS)
    why = dossier["why_stopped"]
    assert why["status"] == PROVEN
    assert why["detector"] == "loop"
    assert why["headline"] == "Pathlayer stopped the run"
    assert why["enforcement"].startswith("enforced: ")
    assert why["at"] is not None
    (group,) = dossier["attempts"]["groups"]
    assert group == {
        "tool": "read",
        "params_digest": DIGEST[:12],
        "calls": 8,
        "blocked": group["blocked"],
        "failed": 0,
        "first_at": group["first_at"],
        "last_at": group["last_at"],
    }
    assert group["blocked"] >= 1
    assert dossier["pathlayer_actions"]["gate"]["blocked"] == group["blocked"]
    assert dossier["options"]["decision"].startswith("change the approach")
    assert dossier == build_resume_dossier(run)
    assert run.resume_md_path.read_text(encoding="utf-8") == render_resume_markdown(dossier)


def test_the_summary_is_three_lines_of_template_text(tmp_path: Path) -> None:
    run = _loop(OpenClawRunRecorder(tmp_path, enforce=True), "run-a")
    summary = _stored(run)["summary"]

    assert len(summary) == 3
    assert re.match(r"Pathlayer stopped the run at \d\d:\d\d:\d\d UTC: ", summary[0])
    assert summary[0].endswith(": the agent kept repeating the same tool calls.")
    assert summary[1].startswith("8 proven, 0 failed, 0 unproven fact(s); ")
    assert summary[2].startswith("Decision needed: change the approach")
    assert all("a.py" not in line for line in summary)


def test_a_stop_writes_a_provisional_dossier_before_the_run_ends(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    run = _loop(recorder, "run-a", end=False)

    dossier = _stored(run)
    assert dossier["provisional"] is True
    assert dossier["why_stopped"]["detector"] == "loop"
    assert "how the run ends: this dossier is provisional" in dossier["unknowns"]["items"]

    recorder.process_payload(_end("run-a"))
    assert _stored(run)["provisional"] is False


def test_a_provider_overflow_leads_to_a_new_session_option(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_read("run-a", 0))
    recorder.process_payload(_error_end("run-a"))
    run = _run(tmp_path, "run-a")

    waiting = _stored(run)
    assert waiting["provisional"] is True
    assert waiting["why_stopped"]["cause"] == "awaiting_retry"
    assert waiting["options"]["decision"].startswith("start a new session or compact")

    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_error_end("run-a"))
    recorder.process_payload(_start("run-b"))
    final = _stored(run)
    assert final["provisional"] is False
    why = final["why_stopped"]
    assert (why["status"], why["cause"], why["detail"]) == (
        OBSERVED,
        "openclaw_error",
        "a context overflow",
    )
    assert why["reason"] == "Context overflow"
    assert final["options"]["decision"].startswith("start a new session or compact")


def test_a_retry_that_succeeds_removes_the_provisional_dossier(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_read("run-a", 0))
    recorder.process_payload(_error_end("run-a"))
    run = _run(tmp_path, "run-a")
    assert run.resume_json_path.exists()

    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_read("run-a", 1))
    recorder.process_payload(_end("run-a"))
    assert run.read_manifest()["outcome"] == "ok"
    assert not run.resume_json_path.exists()
    assert not run.resume_md_path.exists()


def test_the_startup_sweep_writes_a_dossier_for_an_interrupted_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_read("run-a", 0))
    run = _run(tmp_path, "run-a")
    assert not run.resume_json_path.exists()

    OpenClawRunRecorder(tmp_path).startup()
    dossier = _stored(run)
    why = dossier["why_stopped"]
    assert (why["status"], why["cause"], why["at"]) == (UNKNOWN, "interrupted", None)
    assert dossier["provisional"] is False
    assert dossier["identity"]["ended_at"] is None
    assert "what the run did after the listener's last record" in dossier["unknowns"]["items"]
    assert dossier["last_clean_call"]["status"] == OBSERVED
    assert dossier["last_clean_call"]["event_id"] == "openclaw:2:0"
    raw = [json.loads(line) for line in run.raw_events_path.read_text().splitlines()]
    assert why["last_record_at"] == raw[-1]["received_at"]
    assert why["noticed_at"] == run.read_manifest()["ended_at"]
    assert "- Last record from the run: " in run.resume_md_path.read_text()
    assert "the interruption was noticed at " in run.resume_md_path.read_text()


def test_calls_refused_by_the_in_flight_cap_alone_get_a_dossier(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    recorder.process_payload(_start("run-cap"))
    for index in range(6):
        recorder.tool_call_decision(_query("run-cap", f"same-{index}", tool="exec"))
    recorder.process_payload(_end("run-cap"))

    dossier = _stored(_run(tmp_path, "run-cap"))
    assert dossier["why_stopped"]["cause"] == "blocked_calls"
    assert dossier["why_stopped"]["status"] == DECIDED
    blocked = dossier["pathlayer_actions"]["blocked_calls"]
    assert blocked and all(call["why"] == "the same call was already running" for call in blocked)


# --------------------------------------------------------------------------- #
# How a stop is known
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("reason", "status", "template"),
    [
        ("loop: 5x repeat of 1-action block", PROVEN, "loop"),
        ("retry: 3x identical failure of 'exec' (error 'x')", PROVEN, "retry"),
        # The same detectors' inferred components are not proven.
        ("loop: 3/5 overlapping 2-action windows are the same recurring pattern", DECIDED, "stop"),
        ("retry: 'exec' dominates the last 8 calls regardless of result", DECIDED, "stop"),
        # A progress contract surfaces as a no_progress stall: an inference.
        ("no_progress: 6 steps without new progress (cusum)", DECIDED, "no_progress"),
        ("budget: $5.00 over the $4.00 limit", DECIDED, "budget"),
        ("resource: run exceeded 600s", DECIDED, "resource"),
        ("custom: something", DECIDED, "stop"),
    ],
)
def test_only_identical_calls_or_failures_make_a_stop_proven(
    reason: str, status: str, template: str
) -> None:
    ticks = [TickRecord(index=0, ts=0.0, fused=1.0, action=Action.STOP, reason=reason)]
    report = build_session_report(
        run_id="r", status="completed", ticks=ticks, events_recorded=1, stopped=True
    )
    why = _why_stopped("pathlayer_stop", {"stopped": True}, report, None, None)
    assert why["status"] == status
    proven = why["status"] == PROVEN
    assert _template_key("pathlayer_stop", why["detector"], None, proven=proven) == template


# --------------------------------------------------------------------------- #
# Cost before and after the stop
# --------------------------------------------------------------------------- #


def test_the_cost_is_split_at_the_stop_by_arrival(tmp_path: Path) -> None:
    payloads = json.loads((FIXTURES / "openclaw_2026_7" / "in_run.json").read_text())
    recorder = OpenClawRunRecorder(tmp_path)
    for payload in payloads:
        recorder.process_payload(payload)
    run = _run(tmp_path, "fixture-run")
    raw = [json.loads(line) for line in run.raw_events_path.read_text().splitlines()]
    llm = [r["sequence"] for r in raw if r["payload"].get("hook") == "llm_output"]
    assert llm

    whole, nothing = llm_cost_split(run, None)
    assert whole.total_usd == pytest.approx(0.021)
    assert nothing.costed_events == 0
    before, after = llm_cost_split(run, llm[0] - 1)
    assert before.costed_events == 0
    assert after.total_usd == pytest.approx(0.021)


# --------------------------------------------------------------------------- #
# Privacy and wording
# --------------------------------------------------------------------------- #


def test_no_raw_argument_and_no_secret_reaches_the_dossier(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.process_payload(_start("run-p"))
    recorder.process_payload(_write("run-p", "w1", "TOP-SECRET-CONTENT-4242"))
    secret = "sk-" + "A1b2C3d4" * 4
    recorder.process_payload(_error_end("run-p", f"provider said: key {secret} rejected " * 10))
    recorder.process_payload(_start("run-q"))
    run = _run(tmp_path, "run-p")

    text = run.resume_json_path.read_text() + run.resume_md_path.read_text()
    assert "TOP-SECRET-CONTENT-4242" not in text
    assert secret not in text
    dossier = _stored(run)
    assert "[masked]" in dossier["why_stopped"]["reason"]
    assert len(dossier["why_stopped"]["reason"]) <= 120
    assert "provider said" not in " ".join(dossier["summary"])
    if os.name != "nt":
        for path in (run.resume_json_path, run.resume_md_path):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_masking_keeps_paths_and_hides_tokens() -> None:
    assert mask_text("ls /tmp/pathlayer-test-file-inexistant-98765/a/b") == (
        "ls /tmp/pathlayer-test-file-inexistant-98765/a/b"
    )
    assert mask_text("Authorization: Bearer abc.def-ghi") == "Authorization: [masked]"
    assert mask_text("AKIAABCDEFGHIJKLMNOP and ghp_" + "x" * 30) == "[masked] and [masked]"
    assert mask_text("mail me at a.b@example.com") == "mail me at [masked]"
    assert mask_text("x\ny\tz") == "x y z"
    assert mask_text("PGPASSWORD=hunter2 psql") == "PGPASSWORD=[masked] psql"
    assert mask_text("X-Api-Key: abcdef123456") == "X-Api-Key: [masked]"
    assert mask_text("postgres://bob:pw@db/x") == "postgres://[masked]@db/x"
    assert mask_text("sk_live_" + "a" * 24) == "[masked]"
    assert mask_text("npm i react@18.3.1") == "npm i react@18.3.1"
    assert len(mask_text("a " * 200)) == 120


def test_no_template_or_dossier_promises_anything(tmp_path: Path) -> None:
    texts = template_texts()
    run = _loop(OpenClawRunRecorder(tmp_path, enforce=True), "run-a")
    texts.append(run.resume_md_path.read_text(encoding="utf-8"))
    for text in texts:
        lowered = text.lower()
        for word in BANNED:
            assert word not in lowered, (word, text)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_prints_the_dossier_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _loop(OpenClawRunRecorder(tmp_path, enforce=True), "run-a")
    expected = run.resume_md_path.read_text(encoding="utf-8")
    run.resume_json_path.unlink()
    run.resume_md_path.unlink()

    assert main(["run-a", "--resume", "--runs-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == expected
    assert main(["--last", "--resume", "--json", "--runs-root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == SCHEMA
    assert not run.resume_json_path.exists()


def test_cli_says_when_a_run_needs_no_dossier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("ok"))
    recorder.process_payload(_end("ok"))
    assert main(["ok", "--resume", "--runs-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("no resume dossier: this run ended normally")
    assert main(["nope", "--resume", "--runs-root", str(tmp_path)]) == 1


def test_cli_refuses_resume_over_several_runs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--all", "--resume", "--runs-root", str(tmp_path)])


def test_a_passive_stop_is_not_said_to_be_enforced(tmp_path: Path) -> None:
    run = _loop(OpenClawRunRecorder(tmp_path), "run-a")
    dossier = _stored(run)
    (stop,) = [a for a in dossier["pathlayer_actions"]["actions"] if a["action"] == "STOP"]
    assert stop["emitted"] is True
    assert stop["delivery"] == "recorded only: not enforced"
    assert dossier["why_stopped"]["headline"] == "Pathlayer decided to stop the run"
    assert dossier["why_stopped"]["enforcement"] == "not enforced: recorded only"
    text = run.resume_md_path.read_text()
    assert "Pathlayer stopped" not in text
    assert "- Enforcement: not enforced: recorded only." in text


def test_late_usage_counts_in_the_whole_run_cost_when_nothing_stopped(tmp_path: Path) -> None:
    payloads = json.loads((FIXTURES / "openclaw_2026_7" / "post_run.json").read_text())
    recorder = OpenClawRunRecorder(tmp_path)
    for payload in payloads:
        recorder.process_payload(payload)
    run = _run(tmp_path, "fixture-run")
    assert run.late_events_path.exists()

    whole, after = llm_cost_split(run, None)
    assert whole.total_usd == pytest.approx(0.021)
    assert after.costed_events == 0
    before, late = llm_cost_split(run, 10**9)
    assert before.costed_events == 0
    assert late.total_usd == pytest.approx(0.021)


def test_a_secret_in_a_command_is_masked_in_the_ledger(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-s"))
    command = "curl -H 'Authorization: Bearer abc.def' https://u:pw@example.org/x"
    recorder.process_payload(
        {
            "hook": "after_tool_call",
            "event": {
                "toolName": "exec",
                "params": {"command": command},
                "result": {"details": {"status": "failed", "exitCode": 7}},
                "toolCallId": "e1",
                "runId": "run-s",
            },
            "ctx": _ctx("run-s"),
        }
    )
    recorder.process_payload(_error_end("run-s", "boom"))
    recorder.process_payload(_start("run-t"))
    run = _run(tmp_path, "run-s")
    text = run.resume_json_path.read_text() + run.resume_md_path.read_text()
    assert "abc.def" not in text
    assert "u:pw@" not in text
    (failed,) = _stored(run)["proof_ledger"]["failed"]
    assert failed["statement"].startswith("`curl -H 'Authorization: [masked]")


def test_a_dossier_that_cannot_be_rebuilt_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlayer.adapters import openclaw_recorder

    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_read("run-a", 0))
    recorder.process_payload(_error_end("run-a"))
    run = _run(tmp_path, "run-a")
    assert run.resume_json_path.exists()

    def broken(*args: Any, **kwargs: Any) -> None:
        raise ValueError("unreadable journal")

    monkeypatch.setattr(openclaw_recorder, "write_run_report", broken)
    monkeypatch.setattr(openclaw_recorder, "write_resume_dossier", broken)
    recorder.process_payload(_start("run-a"))
    recorder.process_payload(_end("run-a"))
    assert not run.resume_json_path.exists()
    assert not run.resume_md_path.exists()


def test_an_old_manifest_with_an_error_and_no_outcome_gets_a_dossier(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("old"))
    recorder.process_payload(_read("old", 0))
    recorder.process_payload(_end("old"))
    run = _run(tmp_path, "old")
    manifest = run.read_manifest()
    del manifest["outcome"]
    manifest["error"] = "provider failed"
    run.update_manifest(**manifest)
    run.update_manifest(outcome=None)

    dossier = build_resume_dossier(run)
    assert dossier is not None
    assert dossier["why_stopped"]["cause"] == "openclaw_error"


def _every_dossier(root: Path) -> list[str]:
    """resume.md of each kind of run this module covers."""
    _loop(OpenClawRunRecorder(root / "a", enforce=True), "loop")
    _loop(OpenClawRunRecorder(root / "b"), "passive")
    overflow = OpenClawRunRecorder(root / "c")
    overflow.process_payload(_start("err"))
    overflow.process_payload(_error_end("err"))
    waiting = OpenClawRunRecorder(root / "d")
    waiting.process_payload(_start("wait"))
    waiting.process_payload(_error_end("wait"))
    overflow.process_payload(_start("next"))
    cut = OpenClawRunRecorder(root / "e")
    cut.process_payload(_start("cut"))
    cut.process_payload(_read("cut", 0))
    OpenClawRunRecorder(root / "e").startup()
    return [path.read_text() for path in sorted(root.rglob("resume.md"))]


def test_no_dossier_of_any_kind_promises_anything(tmp_path: Path) -> None:
    texts = _every_dossier(tmp_path)
    assert len(texts) == 5
    for text in texts:
        lowered = text.lower()
        for word in BANNED:
            assert word not in lowered, (word, text)


# --------------------------------------------------------------------------- #
# resume.md: decision first, readable, nothing empty
# --------------------------------------------------------------------------- #


def test_the_markdown_puts_the_decision_first_and_the_run_details_last(tmp_path: Path) -> None:
    run = _loop(OpenClawRunRecorder(tmp_path, enforce=True), "run-a")
    text = run.resume_md_path.read_text(encoding="utf-8")

    headings = [line for line in text.splitlines() if line.startswith(("**Decision", "## "))]
    assert headings[0].startswith("**Decision needed:** change the approach")
    assert headings[1] == "## How to restart"
    assert headings[2] == "## What happened"
    assert headings[-2:] == ["## Not known", "## Run details"]
    assert "How we know: proven by the records" in text
    # Clock times, never raw ISO stamps, and no status tag in a heading.
    assert not re.search(r"\d{4}-\d\d-\d\dT\d\d:", text)
    assert "_observed_" not in text and "_proven_" not in text
    # The stop reason is said once.
    assert text.count("loop: 5x repeat of 1-action block") == 1


def test_an_unmeasured_cost_and_an_empty_ledger_column_are_left_out(tmp_path: Path) -> None:
    run = _loop(OpenClawRunRecorder(tmp_path, enforce=True), "run-a")
    text = run.resume_md_path.read_text(encoding="utf-8")

    assert "## Cost" not in text
    assert "the cost: no model usage report could be priced" in text
    assert "**failed**" not in text and "- none" not in text


def test_a_passive_stop_counts_the_calls_that_still_ran(tmp_path: Path) -> None:
    run = _loop(OpenClawRunRecorder(tmp_path), "run-a")
    dossier = _stored(run)

    assert dossier["why_stopped"]["tool_calls_after_stop"] == 3
    assert dossier["summary"][1].endswith("; 3 call(s) ran after the stop decision.")
    text = run.resume_md_path.read_text(encoding="utf-8")
    assert "After the stop decision, 3 more tool call(s) ran" in text


def _blocked_result(run_id: str, call: str) -> dict[str, Any]:
    payload = _read(run_id, 0)
    details = {
        "status": "blocked",
        "deniedReason": "plugin-before-tool-call",
        "reason": "Pathlayer stopped this run",
    }
    payload["event"] = {**payload["event"], "toolCallId": call, "result": {"details": details}}
    return payload


def test_an_enforced_stop_does_not_count_the_calls_it_blocked(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.process_payload(_start("run-a"))
    for index in range(5):
        recorder.tool_call_decision(_query("run-a", f"call-{index}"))
        recorder.process_payload(_read("run-a", index))
    for index in range(5, 8):
        assert recorder.tool_call_decision(_query("run-a", f"call-{index}"))["block"] is True
        recorder.process_payload(_blocked_result("run-a", f"call-{index}"))
    recorder.process_payload(_end("run-a"))
    dossier = _stored(_run(tmp_path, "run-a"))

    assert dossier["why_stopped"]["tool_calls_after_stop"] == 0
    assert dossier["why_stopped"]["headline"] == "Pathlayer stopped the run"
    assert dossier["why_stopped"]["enforcement"] == "enforced: 3 call(s) blocked after the stop"
    assert "ran after the stop decision" not in dossier["summary"][1]


def test_a_cap_block_alone_does_not_make_the_stop_enforced() -> None:
    ticks = [TickRecord(index=0, ts=0.0, fused=1.0, action=Action.STOP, reason="loop: 5x")]
    report = build_session_report(
        run_id="r", status="completed", ticks=ticks, events_recorded=1, stopped=True
    )
    cap = BlockedToolCall("read", "c", None, "identical calls in flight: 4 of 4")
    report.tool_call_gate = ToolCallGate(queries=5, enforced=True, blocked=(cap,))
    why = _why_stopped("pathlayer_stop", {"stopped": True}, report, None, None)

    assert why["headline"] == "Pathlayer decided to stop the run"
    assert why["enforcement"] == "enforced, but no later tool call was attempted"


def test_failures_about_different_files_stay_separate_lines() -> None:
    entries = [
        {"statement": "read failed: ENOENT (file_1.txt)", "count": 1},
        {"statement": "read failed: ENOENT (file_2.txt)", "count": 1},
        {"statement": "wrote a.md (reported size 10)", "count": 1},
        {"statement": "wrote a.md (reported size 12)", "count": 1},
    ]
    assert _merged_entries(entries) == [
        "read failed: ENOENT (file_1.txt)",
        "read failed: ENOENT (file_2.txt)",
        "wrote a.md (reported size 12) (x2; the last one shown)",
    ]


def _read_text(run_id: str, call: str, text: str) -> dict[str, Any]:
    payload = _read(run_id, 0)
    payload["event"] = {
        **payload["event"],
        "toolCallId": call,
        "result": {"content": [{"type": "text", "text": text}]},
    }
    return payload


def test_reads_of_one_file_at_several_sizes_are_one_markdown_line(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.process_payload(_start("run-a"))
    for index, text in enumerate(("a", "bb", "ccc")):
        recorder.process_payload(_read_text("run-a", f"call-{index}", text))
    recorder.process_payload(_error_end("run-a"))
    run = _run(tmp_path, "run-a")

    assert len(_stored(run)["proof_ledger"]["proven"]) == 3
    lines = [
        line
        for line in run.resume_md_path.read_text().splitlines()
        if "a.py" in line and "read" in line
    ]
    assert lines == ["- read a.py back (3 characters) (x3; the last one shown)"]
