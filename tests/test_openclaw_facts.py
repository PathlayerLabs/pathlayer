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
    extract_facts,
    facts_to_json,
)
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_run_store import RunStore


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
