"""Facts read from an OpenClaw run's raw journal (pathlayer.adapters.openclaw_facts).

The payloads reproduce the field names and shapes of real OpenClaw 2026.7.1
``after_tool_call`` results (exec ``result.details``, the write tool's
confirmation, a call blocked by a plugin); their contents are invented.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pathlayer.adapters.openclaw_facts import (
    MAX_FACT_TEXT_CHARS,
    FactKind,
    build_proof_ledger,
    extract_facts,
    facts_to_json,
)
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_report import build_run_report
from pathlayer.adapters.openclaw_run_store import RunStore
from pathlayer.postmortem import LEDGER_FAILED, LEDGER_PROVEN, LEDGER_UNPROVEN


def _record(sequence: int, event: dict[str, Any], hook: str = "after_tool_call") -> dict[str, Any]:
    return {"sequence": sequence, "payload": {"hook": hook, "event": event, "ctx": {"runId": "r"}}}


def _exec(command: str, **details: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "toolName": "exec",
        "toolCallId": f"call-{command}",
        "params": {"command": command},
        "result": {
            "content": [{"type": "text", "text": "output"}],
            "details": {"status": "completed", "cwd": "/work", **details},
        },
    }
    return event


def test_exec_exit_codes_become_command_facts() -> None:
    failed = _exec("node --check app.js", exitCode=127, failureKind="shell-command-not-found")
    failed["result"]["details"]["status"] = "failed"
    failed["error"] = "/bin/sh: 1: node: not found"
    facts = extract_facts([_record(2, _exec("mkdir -p game", exitCode=0)), _record(3, failed)])

    assert [fact.kind for fact in facts] == [FactKind.COMMAND_SUCCEEDED, FactKind.COMMAND_FAILED]
    ok, ko = facts
    assert ok.sequence == 2 and ok.event_id == "openclaw:2:0"
    assert ok.tool == "exec" and ok.tool_call_id == "call-mkdir -p game"
    assert ok.detail["exit_code"] == 0
    assert ok.detail["command"] == "mkdir -p game"
    assert ok.detail["command_sha256"] == hashlib.sha256(b"mkdir -p game").hexdigest()
    assert ok.detail["cwd"] == "/work"
    assert ko.detail["exit_code"] == 127
    assert ko.detail["failure_kind"] == "shell-command-not-found"


@pytest.mark.parametrize("failure_kind", ["overall-timeout", "no-output-timeout"])
def test_a_timed_out_command_is_its_own_fact(failure_kind: str) -> None:
    event = _exec("sleep 999", timedOut=True, failureKind=failure_kind)
    event["result"]["details"]["status"] = "failed"
    (fact,) = extract_facts([_record(1, event)])
    assert fact.kind is FactKind.COMMAND_TIMED_OUT


def test_a_denied_exec_approval_is_a_block_not_a_timeout() -> None:
    text = "Exec denied (gateway id=ap-1, approval-timeout: allowlist-miss): rm -rf build"
    event = {
        "toolName": "exec",
        "params": {"command": "rm -rf build"},
        "result": {
            "content": [{"type": "text", "text": text}],
            "details": {"status": "failed", "exitCode": None, "durationMs": 0, "timedOut": True},
        },
    }
    (fact,) = extract_facts([_record(1, event)])
    assert fact.kind is FactKind.CALL_BLOCKED
    assert fact.detail["denied_reason"] == "exec-approval: approval-timeout: allowlist-miss"
    assert fact.detail["by_pathlayer"] is False


def test_a_shell_backgrounded_command_is_only_started() -> None:
    # The real shape: the shell exits 0 at once, status completed.
    command = "cd game && python3 -m http.server 8080 > /dev/null 2>&1 &"
    (fact,) = extract_facts([_record(7, _exec(command, exitCode=0))])
    assert fact.kind is FactKind.COMMAND_BACKGROUNDED
    (chained,) = extract_facts([_record(8, _exec("make &&", exitCode=0))])
    assert chained.kind is FactKind.COMMAND_SUCCEEDED


@pytest.mark.parametrize("status", ["failed", "running"])
def test_exit_zero_with_another_status_proves_nothing(status: str) -> None:
    event = _exec("x", exitCode=0)
    event["result"]["details"]["status"] = status
    assert extract_facts([_record(1, event)]) == ()


def test_exit_zero_with_an_error_proves_nothing() -> None:
    event = _exec("x", exitCode=0)
    event["error"] = "boom"
    assert extract_facts([_record(1, event)]) == ()


@pytest.mark.parametrize("exit_code", [None, "0", True, 1.0])
def test_exec_without_a_usable_exit_code_proves_nothing(exit_code: Any) -> None:
    # Still running (background=true / yieldMs), or killed by a signal.
    event = _exec("serve")
    event["result"]["details"]["status"] = "running"
    if exit_code is not None:
        event["result"]["details"]["exitCode"] = exit_code
    assert extract_facts([_record(1, event)]) == ()


def test_exec_without_exit_code_but_with_an_error_is_a_failed_call() -> None:
    event = _exec("x")
    event["error"] = "spawn failed"
    (fact,) = extract_facts([_record(1, event)])
    assert fact.kind is FactKind.CALL_FAILED
    assert fact.detail["error"] == "spawn failed"


def test_a_write_confirmation_becomes_a_file_fact() -> None:
    content = "<!DOCTYPE html>"
    event = {
        "toolName": "write",
        "toolCallId": "w1",
        "params": {"path": "game/index.html", "content": content},
        "result": {
            "content": [{"type": "text", "text": "Successfully wrote 15 bytes to game/index.html"}]
        },
    }
    (fact,) = extract_facts([_record(4, event)])
    assert fact.kind is FactKind.FILE_WRITTEN
    assert fact.detail == {
        "path": "game/index.html",
        "reported_size": 15,
        "content_utf8_bytes": 15,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"content": []},
        {"content": [{"type": "image"}]},
        {"content": [{"type": "text", "text": "Wrote something"}]},
    ],
)
def test_a_write_without_its_confirmation_proves_nothing(result: Any) -> None:
    event = {"toolName": "write", "params": {"path": "a"}, "result": result}
    assert extract_facts([_record(1, event)]) == ()


def test_a_failed_write_is_a_failed_call() -> None:
    event = {"toolName": "write", "params": {"path": "/root/x"}, "error": "EACCES"}
    (fact,) = extract_facts([_record(1, event)])
    assert fact.kind is FactKind.CALL_FAILED


def _edit(path: str = "doc.md", text: str | None = None, **extra: Any) -> dict[str, Any]:
    text = text if text is not None else f"Successfully replaced 1 block(s) in {path}."
    return {
        "toolName": "edit",
        "params": {"path": path, "edits": [{"oldText": "a", "newText": "b"}]},
        "result": {"content": [{"type": "text", "text": text}], "details": {"diff": "..."}},
        **extra,
    }


def _read(path: str = "doc.md", text: str = "# Title\nbody") -> dict[str, Any]:
    return {
        "toolName": "read",
        "params": {"path": path, "offset": -20},
        "result": {"content": [{"type": "text", "text": text}]},
    }


def test_a_confirmed_edit_becomes_a_file_edited_fact() -> None:
    event = _edit(text="Successfully replaced 3 block(s) in notes/doc.md.")
    (fact,) = extract_facts([_record(1, event)])
    assert fact.kind is FactKind.FILE_EDITED
    assert fact.detail == {"path": "notes/doc.md", "blocks": 3}


@pytest.mark.parametrize(
    "text",
    [
        "Edited it",
        "Successfully replaced x block(s) in a.",
        "Successfully replaced 1 block(s) in a",
    ],
)
def test_an_edit_without_its_confirmation_proves_nothing(text: str) -> None:
    assert extract_facts([_record(1, _edit(text=text))]) == ()
    no_content = {"toolName": "edit", "params": {"path": "a"}, "result": {"content": []}}
    assert extract_facts([_record(1, no_content)]) == ()


def test_a_failed_edit_stays_a_failed_call() -> None:
    error = "Found 2 occurrences of the text in doc.md. The text must be unique."
    event = _edit(text='{"status":"error"}', error=error)
    event["result"]["details"] = {"status": "error", "tool": "edit"}
    (fact,) = extract_facts([_record(1, event)])
    assert fact.kind is FactKind.CALL_FAILED
    assert fact.detail["error"] == error


def test_a_read_keeps_its_path_and_length_never_its_text() -> None:
    (fact,) = extract_facts([_record(1, _read(text="# Title\nsecret body"))])
    assert fact.kind is FactKind.FILE_READ
    assert fact.detail == {"path": "doc.md", "chars": len("# Title\nsecret body")}
    assert "secret" not in facts_to_json([fact])


@pytest.mark.parametrize(
    "event",
    [
        {"toolName": "read", "params": {"path": "a"}, "result": "text"},
        {"toolName": "read", "params": {"path": "a"}, "result": {"content": []}},
        {"toolName": "read", "params": {"path": "a"}, "result": {"content": [{"type": "image"}]}},
        {"toolName": "read", "params": {}, "result": {"content": [{"type": "text", "text": "x"}]}},
        {
            "toolName": "read",
            "params": {"path": 3},
            "result": {"content": [{"type": "text", "text": "x"}]},
        },
        {"toolName": "read", "params": {"path": "a"}, "error": "ENOENT"},
    ],
)
def test_a_read_without_text_or_path_is_no_proof(event: dict[str, Any]) -> None:
    kinds = [fact.kind for fact in extract_facts([_record(1, event)])]
    assert FactKind.FILE_READ not in kinds


def test_edits_and_reads_are_proven_and_identical_reads_group() -> None:
    records = [
        _record(1, _edit()),
        _record(2, _read()),
        _record(3, _read()),
        _record(4, _read(text="longer text")),
        _record(5, _read(path="other.md")),
    ]
    proven = build_proof_ledger(records).column(LEDGER_PROVEN)
    assert [(entry.statement, entry.count) for entry in proven] == [
        ("edited doc.md (1 block(s) replaced)", 1),
        ("read doc.md back (12 characters)", 2),
        ("read doc.md back (11 characters)", 1),
        ("read other.md back (12 characters)", 1),
    ]


def test_a_path_in_an_edit_or_read_statement_is_escaped() -> None:
    records = [_record(1, _edit(path="a\tb.md")), _record(2, _read(path="c\nd.md"))]
    statements = [entry.statement for entry in build_proof_ledger(records).column(LEDGER_PROVEN)]
    assert statements == [
        "edited a\\tb.md (1 block(s) replaced)",
        "read c\\nd.md back (12 characters)",
    ]


def test_a_blocked_call_says_who_blocked_it() -> None:
    reason = "Pathlayer stopped this run (retry: 3x identical failure of 'exec'). Do not ..."
    blocked = _exec("cat missing")
    blocked["result"]["details"] = {
        "status": "blocked",
        "deniedReason": "plugin-before-tool-call",
        "reason": reason,
    }
    other = _exec("rm -rf /")
    other["result"]["details"] = {"status": "blocked", "deniedReason": "plugin-approval"}
    by_pathlayer, by_other = extract_facts([_record(5, blocked), _record(6, other)])

    assert by_pathlayer.kind is FactKind.CALL_BLOCKED
    assert by_pathlayer.detail["by_pathlayer"] is True
    assert by_pathlayer.detail["denied_reason"] == "plugin-before-tool-call"
    assert by_other.detail["by_pathlayer"] is False
    assert by_other.detail["reason"] is None


def test_other_tools_and_hooks_yield_no_fact() -> None:
    status = {"toolName": "session_status", "result": {"details": {"ok": True}}}
    records = [
        _record(1, {"prompt": "go"}, hook="before_agent_run"),
        _record(2, status),
        _record(3, {"toolName": "read", "params": {"path": "a"}, "result": "text"}),
    ]
    assert extract_facts(records) == ()


@pytest.mark.parametrize(
    "record",
    [
        {"sequence": "1", "payload": {"hook": "after_tool_call", "event": {"toolName": "exec"}}},
        {"sequence": True, "payload": {}},
        {"sequence": 1, "payload": "not an object"},
        {"sequence": 1, "payload": {"hook": "after_tool_call", "event": []}},
        {"sequence": 1, "payload": {"hook": "after_tool_call", "event": {"toolName": ""}}},
        {"sequence": 1, "payload": {"hook": "after_tool_call", "event": {"result": "x"}}},
    ],
)
def test_malformed_records_are_skipped(record: dict[str, Any]) -> None:
    assert extract_facts([record]) == ()


def test_long_texts_are_clipped() -> None:
    event = _exec("x" * 500, exitCode=0)
    (fact,) = extract_facts([_record(1, event)])
    assert len(fact.detail["command"]) == MAX_FACT_TEXT_CHARS
    assert fact.detail["command"].endswith("...")
    # The digest is of the full command.
    assert fact.detail["command_sha256"] == hashlib.sha256(("x" * 500).encode()).hexdigest()


def test_facts_serialize_deterministically() -> None:
    facts = extract_facts([_record(1, _exec("ls", exitCode=0))])
    assert json.loads(facts_to_json(facts))[0]["kind"] == "command_succeeded"
    assert facts_to_json(facts) == facts_to_json(
        extract_facts([_record(1, _exec("ls", exitCode=0))])
    )


def test_facts_point_at_the_analyzed_events(tmp_path: Path) -> None:
    """A fact's sequence is the raw record decisions.jsonl names for that event."""
    recorder = OpenClawRunRecorder(tmp_path)
    ctx = {"runId": "run-a", "sessionKey": "s", "agentId": "a"}
    recorder.process_payload({"hook": "before_agent_run", "event": {}, "ctx": ctx})
    for command, code in (("make", 2), ("make test", 0)):
        event = _exec(command, exitCode=code)
        event["runId"] = "run-a"
        recorder.process_payload({"hook": "after_tool_call", "event": event, "ctx": ctx})
    recorder.process_payload({"hook": "agent_end", "event": {"success": True}, "ctx": ctx})

    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None
    records = [json.loads(line) for line in run.raw_events_path.read_text().splitlines()]
    facts = extract_facts(records)
    decisions = [json.loads(line) for line in run.decisions_path.read_text().splitlines()]
    # decisions.jsonl names the raw record each analyzed event came from.
    assert [fact.sequence for fact in facts] == [line["sequence"] for line in decisions]
    assert [fact.tool for fact in facts] == [line["tool"] for line in decisions]
    assert [fact.kind for fact in facts] == [FactKind.COMMAND_FAILED, FactKind.COMMAND_SUCCEEDED]


# --------------------------------------------------------------------------- #
# Proof ledger
# --------------------------------------------------------------------------- #


def _write(path: str, size: int) -> dict[str, Any]:
    return {
        "toolName": "write",
        "params": {"path": path, "content": "x" * size},
        "result": {
            "content": [{"type": "text", "text": f"Successfully wrote {size} bytes to {path}"}]
        },
    }


def _blocked() -> dict[str, Any]:
    event = _exec("cat missing")
    event["result"]["details"] = {
        "status": "blocked",
        "deniedReason": "plugin-before-tool-call",
        "reason": "Pathlayer stopped this run (retry). Do not call more tools.",
    }
    return event


def test_the_ledger_sorts_facts_into_three_columns() -> None:
    failed = _exec("node --check app.js", exitCode=127, failureKind="shell-command-not-found")
    records = [
        _record(1, {"prompt": "go"}, hook="before_agent_run"),
        _record(2, _exec("mkdir -p game", exitCode=0)),
        _record(3, _write("game/app.js", 10)),
        _record(4, _exec("python3 -m http.server 8080 &", exitCode=0)),
        _record(5, failed),
        _record(6, {"toolName": "search", "params": {"q": "a"}, "result": "text"}),
        _record(7, {"toolName": "search", "params": {"q": "b"}, "result": "text"}),
    ]
    ledger = build_proof_ledger(records)

    assert [entry.statement for entry in ledger.column(LEDGER_PROVEN)] == [
        "`mkdir -p game` exited 0",
        "wrote game/app.js (reported size 10)",
    ]
    assert [entry.statement for entry in ledger.column(LEDGER_FAILED)] == [
        "`node --check app.js` exited 127, shell-command-not-found"
    ]
    unproven = ledger.column(LEDGER_UNPROVEN)
    assert [entry.statement for entry in unproven] == [
        "`python3 -m http.server 8080 &` started in the background: whether it ran is not observed",
        "search: no outcome fact in its result",
    ]
    assert unproven[1].count == 2 and unproven[1].sequences == (6, 7)


def test_identical_facts_are_one_entry_with_a_count() -> None:
    records = [_record(i, _exec("cat missing", exitCode=1)) for i in (2, 3, 4)]
    records.append(_record(5, _blocked()))
    ledger = build_proof_ledger(records)

    failed = ledger.column(LEDGER_FAILED)
    assert [(entry.statement, entry.count) for entry in failed] == [
        ("`cat missing` exited 1", 3),
        ("`cat missing` blocked by Pathlayer", 1),
    ]
    assert failed[0].sequences == (2, 3, 4)
    text = "\n".join(ledger.describe_lines())
    assert "`cat missing` exited 1 (x3)" in text
    assert "proven: none" in text
    assert text.startswith("proof ledger (observed facts, not a verdict on the task):")


def test_a_command_chain_ending_in_and_is_not_backgrounded() -> None:
    ledger = build_proof_ledger([_record(1, _exec("make &&", exitCode=0))])
    assert [entry.status for entry in ledger.entries] == [LEDGER_PROVEN]


def test_other_failures_and_blocks_are_worded_for_what_they_are() -> None:
    other = _exec("rm -rf /")
    other["result"]["details"] = {"status": "blocked"}
    timed_out = _exec("sleep 99", timedOut=True, failureKind="overall-timeout")
    failed_write = {"toolName": "write", "params": {"path": "/x"}, "error": "EACCES"}
    ledger = build_proof_ledger(
        [_record(1, other), _record(2, timed_out), _record(3, failed_write)]
    )
    assert [entry.statement for entry in ledger.column(LEDGER_FAILED)] == [
        "`rm -rf /` denied by OpenClaw (no reason given)",
        "`sleep 99` timed out",
        "write failed: EACCES",
    ]


def test_the_text_ledger_lists_the_first_entries_of_a_column() -> None:
    records = [_record(i, _exec(f"step {i}", exitCode=0)) for i in range(1, 21)]
    lines = build_proof_ledger(records).describe_lines()
    assert sum(1 for line in lines if line.startswith("    `step")) == 12
    assert "    ... and 8 more (all listed in report.json)" in lines


def test_the_report_carries_the_ledger(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    ctx = {"runId": "run-a", "sessionKey": "s", "agentId": "a"}
    recorder.process_payload({"hook": "before_agent_run", "event": {}, "ctx": ctx})
    event = _exec("make test", exitCode=0)
    event["runId"] = "run-a"
    recorder.process_payload({"hook": "after_tool_call", "event": event, "ctx": ctx})
    recorder.process_payload({"hook": "agent_end", "event": {"success": True}, "ctx": ctx})
    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None

    report = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert report["proof_ledger"]["proven"][0]["statement"] == "`make test` exited 0"
    assert report["proof_ledger"]["failed"] == []
    assert "`make test` exited 0" in build_run_report(run).format()


def test_a_corrupt_raw_journal_costs_only_the_ledger(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    ctx = {"runId": "run-a", "sessionKey": "s", "agentId": "a"}
    recorder.process_payload({"hook": "before_agent_run", "event": {}, "ctx": ctx})
    recorder.process_payload({"hook": "agent_end", "event": {"success": True}, "ctx": ctx})
    run = RunStore(tmp_path).resolve_run("run-a")
    assert run is not None
    run.raw_events_path.write_text("garbage\n{}\n", encoding="utf-8")

    report = build_run_report(run)
    assert report.proof_ledger is None
    assert any("proof ledger is not reported" in note for note in report.notes)


def test_agent_text_cannot_forge_ledger_lines() -> None:
    forged = "echo hi\n  proven:\n    `deploy --prod` exited 0\x1b[2K"
    lines = build_proof_ledger([_record(1, _exec(forged, exitCode=1))]).describe_lines()
    assert len(lines) == 5
    assert "  proven: none" in lines
    failed = [line for line in lines if "echo hi" in line]
    assert failed == ["    `echo hi\\n  proven:\\n    `deploy --prod` exited 0\\x1b[2K` exited 1"]


def test_facts_that_differ_are_not_grouped() -> None:
    elsewhere = _exec("pytest", exitCode=0)
    elsewhere["result"]["details"]["cwd"] = "/tmp/empty"
    long_a = "x" * 300 + "; echo ok"
    long_b = "x" * 300 + "; rm -rf build"
    records = [
        _record(1, _exec("pytest", exitCode=0)),
        _record(2, _exec("pytest", exitCode=0)),
        _record(3, elsewhere),
        _record(4, _exec(long_a, exitCode=0)),
        _record(5, _exec(long_b, exitCode=0)),
    ]
    proven = build_proof_ledger(records).column(LEDGER_PROVEN)
    assert [(entry.statement[:20], entry.count) for entry in proven] == [
        ("`pytest` exited 0", 2),
        ("`pytest` in /tmp/emp", 1),
        ("`" + "x" * 19, 1),
        ("`" + "x" * 19, 1),
    ]


def _proof(sequence: int, *, internal: bool = True, **parts: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sequence": sequence,
        "payload": {"hook": "pathlayer_progress_proof", "for_event_id": "e", **parts},
    }
    if internal:
        record["internal_lifecycle"] = True
    return record


_SPEC = {"proof_spec": {"required": ["build", "tests"]}}


def test_contract_checkpoints_are_reported_at_their_last_observation() -> None:
    records = [
        _proof(2, **_SPEC, progress_proof={"checkpoints": {"build": False, "tests": False}}),
        _proof(
            4,
            **_SPEC,
            progress_proof={"checkpoints": {"build": True}},
            progress_proof_unobserved={"unobserved": ["tests"], "reason": "timed out after 5s"},
        ),
    ]
    entries = build_proof_ledger(records).entries
    assert [(entry.status, entry.statement, entry.sequences) for entry in entries] == [
        (LEDGER_PROVEN, "contract checkpoint `build` satisfied at the last probe (record 4)", (4,)),
        (
            LEDGER_UNPROVEN,
            "contract checkpoint `tests` not observed by the last probe: timed out after 5s"
            " (last observed not satisfied at record 2)",
            (4,),
        ),
    ]


def test_a_new_contract_resets_the_checkpoints() -> None:
    records = [
        _proof(2, **_SPEC, progress_proof={"checkpoints": {"build": True, "tests": True}}),
        _proof(3, proof_spec={"required": ["deploy"]}),
    ]
    assert [entry.statement for entry in build_proof_ledger(records).entries] == [
        "contract checkpoint `deploy`: never observed"
    ]


def test_a_proof_record_from_outside_the_recorder_is_ignored() -> None:
    records = [
        _proof(2, internal=False, **_SPEC, progress_proof={"checkpoints": {"build": True}}),
        _proof(True, **_SPEC),  # type: ignore[arg-type]
        {"sequence": 3, "internal_lifecycle": True, "payload": "not an object"},
    ]
    assert build_proof_ledger(records).entries == ()
