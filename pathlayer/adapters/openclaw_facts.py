"""Facts an OpenClaw run's tool calls establish, read from its raw journal.

OpenClaw's runtime reports what a tool call did in the ``after_tool_call``
result: an ``exec`` carries its exit code, failure kind, time-out flag and
working directory in ``result.details``; a ``write`` answers with the number of
bytes it wrote; a call the runtime refused carries ``details.status:
"blocked"`` and the refusal reason. The agent chooses the command, never these
fields, so they are observed facts, not claims.

This module extracts them, and nothing else:

- It is pure: it reads payloads already journaled and returns facts. It never
  feeds a detector, and adds no field to the events the pipeline analyzes.
  Mapping ``details.exitCode`` onto the ``exit_code`` progress anchor would
  make ``no_progress`` able to STOP on OpenClaw, and silence the loop detector
  on a run of identical successful calls; that needs its own labeled evidence.
- It abstains on anything it cannot read: an absent, mistyped or malformed
  field yields no fact rather than a guess.
- A command sent to the background with a trailing ``&`` is only known to
  have been started. Edits, patches and writes that change nothing yield no
  fact.
- A fact says what a call did, never that the task was done. An exit code of 0
  proves that a command exited 0; a write proves that bytes were written at
  that moment, not what the file holds now, and its size is as the tool
  reported it (characters in 2026.7.1, UTF-8 bytes later).

Each fact points back to the raw journal record (``sequence``, which
``decisions.jsonl`` also names) and to the normalized event the recorder
derived from it (``event_id``, its id in the durable state), so a reader can
check it against the source.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pathlayer.adapters.openclaw_lifecycle import AFTER_TOOL_CALL
from pathlayer.postmortem import (
    LEDGER_FAILED,
    LEDGER_PROVEN,
    LEDGER_UNPROVEN,
    LedgerEntry,
    ProofLedger,
)

# Longest command or error text kept in a fact.
MAX_FACT_TEXT_CHARS = 200

# "Successfully wrote 1113 bytes to chess/index.html": OpenClaw's write tool.
# The number is a character count in 2026.7.1 and a UTF-8 byte count in later
# versions, so it is kept as reported, not as bytes.
_WROTE = re.compile(r"^Successfully wrote (\d+) bytes to (.+)$")


class FactKind(str, Enum):
    """What one tool call established."""

    COMMAND_SUCCEEDED = "command_succeeded"
    COMMAND_FAILED = "command_failed"
    COMMAND_TIMED_OUT = "command_timed_out"
    COMMAND_BACKGROUNDED = "command_backgrounded"
    FILE_WRITTEN = "file_written"
    CALL_BLOCKED = "call_blocked"
    CALL_FAILED = "call_failed"


@dataclass(frozen=True, slots=True)
class OpenClawFact:
    """One observed fact about one tool call, traceable to its journal record."""

    kind: FactKind
    sequence: int
    event_id: str
    tool: str
    tool_call_id: str | None
    detail: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "sequence": self.sequence,
            "event_id": self.event_id,
            "tool": self.tool,
            "tool_call_id": self.tool_call_id,
            "detail": dict(self.detail),
        }


_Make = Callable[["FactKind", dict[str, Any]], "OpenClawFact"]


def extract_facts(records: Iterable[Mapping[str, Any]]) -> tuple[OpenClawFact, ...]:
    """Every fact the ``after_tool_call`` records of a raw journal establish, in order.

    ``records`` are raw journal records (``{"sequence": ..., "payload": ...}``,
    as in ``events.raw.jsonl``). Other hooks, and records without a usable
    sequence or tool name, are skipped.
    """
    facts: list[OpenClawFact] = []
    for record in records:
        sequence = record.get("sequence")
        payload = record.get("payload")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            continue
        if not isinstance(payload, Mapping) or payload.get("hook") != AFTER_TOOL_CALL:
            continue
        event = payload.get("event")
        if not isinstance(event, Mapping):
            continue
        fact = _fact(event, sequence)
        if fact is not None:
            facts.append(fact)
    return tuple(facts)


def _fact(event: Mapping[str, Any], sequence: int) -> OpenClawFact | None:
    tool = event.get("toolName")
    if not isinstance(tool, str) or not tool:
        return None
    call_id = event.get("toolCallId")
    result = event.get("result")
    details = result.get("details") if isinstance(result, Mapping) else None
    details = details if isinstance(details, Mapping) else {}

    def make(kind: FactKind, detail: dict[str, Any]) -> OpenClawFact:
        return OpenClawFact(
            kind=kind,
            sequence=sequence,
            # Same id as the recorder gives the normalized event (offset 0:
            # one after_tool_call is one event).
            event_id=f"openclaw:{sequence}:0",
            tool=tool,
            tool_call_id=call_id if isinstance(call_id, str) else None,
            detail=detail,
        )

    exec_denied = _exec_approval_denied(tool, result)
    if details.get("status") == "blocked" or exec_denied is not None:
        reason = details.get("reason")
        denied = exec_denied if exec_denied is not None else details.get("deniedReason")
        return make(
            FactKind.CALL_BLOCKED,
            {
                "denied_reason": denied if isinstance(denied, str) else None,
                "reason": _clip(reason) if isinstance(reason, str) else None,
                "by_pathlayer": details.get("deniedReason") == "plugin-before-tool-call"
                and isinstance(reason, str)
                and reason.startswith("Pathlayer "),
            },
        )
    if tool == "exec":
        return _exec_fact(event, details, make)
    error = event.get("error")
    if isinstance(error, str) and error.strip():
        return make(FactKind.CALL_FAILED, {"error": _clip(error.strip())})
    if tool == "write":
        return _write_fact(event, result, make)
    return None


def _exec_fact(
    event: Mapping[str, Any], details: Mapping[str, Any], make: _Make
) -> OpenClawFact | None:
    params = event.get("params")
    command = params.get("command") if isinstance(params, Mapping) else None
    base: dict[str, Any] = {
        "command": _clip(command) if isinstance(command, str) else None,
        "command_sha256": _sha256(command) if isinstance(command, str) else None,
        "cwd": details.get("cwd") if isinstance(details.get("cwd"), str) else None,
    }
    failure_kind = details.get("failureKind")
    if failure_kind in _TIMEOUT_FAILURE_KINDS:
        return make(FactKind.COMMAND_TIMED_OUT, base)
    exit_code = details.get("exitCode")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        # No exit code: still running (``background`` / ``yieldMs``), killed
        # by a signal, or an unknown runtime. Nothing observed about its end.
        error = event.get("error")
        if isinstance(error, str) and error.strip():
            return make(FactKind.CALL_FAILED, {**base, "error": _clip(error.strip())})
        return None
    if exit_code == 0 and details.get("status") == "completed" and not _has_error(event):
        if isinstance(command, str) and _shell_backgrounded(command):
            # The shell exited 0 after sending the command to the background:
            # nothing says the command itself ran.
            return make(FactKind.COMMAND_BACKGROUNDED, {**base, "exit_code": 0})
        return make(FactKind.COMMAND_SUCCEEDED, {**base, "exit_code": 0})
    if exit_code == 0:
        # Exit 0 with a failed status or an error: contradictory, no fact.
        return None
    return make(
        FactKind.COMMAND_FAILED,
        {
            **base,
            "exit_code": exit_code,
            "failure_kind": failure_kind if isinstance(failure_kind, str) else None,
        },
    )


def _write_fact(event: Mapping[str, Any], result: Any, make: _Make) -> OpenClawFact | None:
    text = _first_text(result)
    if text is None:
        return None
    match = _WROTE.match(text.strip())
    if match is None:
        return None
    params = event.get("params")
    content = params.get("content") if isinstance(params, Mapping) else None
    return make(
        FactKind.FILE_WRITTEN,
        {
            "path": match.group(2),
            # As the write tool reported it: characters in 2026.7.1, UTF-8
            # bytes later.
            "reported_size": int(match.group(1)),
            "content_utf8_bytes": (
                len(content.encode("utf-8")) if isinstance(content, str) else None
            ),
            # The content the agent asked to write: what was written at that
            # moment, not what the file holds now.
            "content_sha256": _sha256(content) if isinstance(content, str) else None,
        },
    )


# exec failure kinds that mean the command ran out of time. A denied approval
# also sets ``timedOut`` when its reason mentions a timeout, without running.
_TIMEOUT_FAILURE_KINDS = frozenset({"overall-timeout", "no-output-timeout"})


def _has_error(event: Mapping[str, Any]) -> bool:
    error = event.get("error")
    return isinstance(error, str) and bool(error.strip())


def _shell_backgrounded(command: str) -> bool:
    """A command ending with a shell ``&`` (not ``&&``): sent to the background."""
    stripped = command.rstrip()
    return stripped.endswith("&") and not stripped.endswith("&&")


def _exec_approval_denied(tool: str, result: Any) -> str | None:
    """The reason of an ``exec`` the gateway's approval denied, or ``None``.

    OpenClaw answers such a call with ``status: failed``, no exit code and the
    text ``Exec denied (gateway id=..., <reason>): <command>``: the command
    never ran.
    """
    if tool != "exec":
        return None
    text = _first_text(result)
    if text is None or not text.startswith("Exec denied ("):
        return None
    head = text[len("Exec denied (") :].split(")", 1)[0]
    reason = head.split(", ", 1)[1] if ", " in head else ""
    return f"exec-approval: {reason}" if reason else "exec-approval"


def _first_text(result: Any) -> str | None:
    if not isinstance(result, Mapping):
        return None
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return None
    first = content[0]
    if not isinstance(first, Mapping) or first.get("type") != "text":
        return None
    text = first.get("text")
    return text if isinstance(text, str) else None


def _clip(text: str) -> str:
    if len(text) <= MAX_FACT_TEXT_CHARS:
        return text
    return text[: MAX_FACT_TEXT_CHARS - 3] + "..."


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def facts_to_json(facts: Iterable[OpenClawFact]) -> str:
    """Deterministic JSON of a fact list (sorted keys), for snapshots and exports."""
    return json.dumps([fact.to_dict() for fact in facts], sort_keys=True, ensure_ascii=False)


def build_proof_ledger(records: Iterable[Mapping[str, Any]]) -> ProofLedger:
    """The proof ledger of a run: its facts in three columns, identical ones grouped.

    - **proven**: commands that exited 0, files written (what the runtime
      confirmed, at that moment);
    - **failed**: commands that exited non-zero or timed out, failed calls,
      blocked calls;
    - **unproven**: commands sent to the background (their exit 0 only says
      they started), and tool calls whose result says nothing observable about
      how they ended (a read, a search), counted per tool.

    Identical facts (same kind, same command, path or error) are one entry with
    a count, in the order of their first occurrence.
    """
    materialized = list(records)
    facts = extract_facts(materialized)
    with_fact = {fact.sequence for fact in facts}
    groups: dict[tuple[str, str], list[OpenClawFact]] = {}
    for fact in facts:
        groups.setdefault((_status(fact), _statement(fact)), []).append(fact)
    entries = [
        LedgerEntry(
            status=status,
            statement=statement,
            count=len(group),
            sequences=tuple(fact.sequence for fact in group),
        )
        for (status, statement), group in groups.items()
    ]
    silent: dict[str, list[int]] = {}
    for record in materialized:
        sequence = record.get("sequence")
        payload = record.get("payload")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence in with_fact:
            continue
        if not isinstance(payload, Mapping) or payload.get("hook") != AFTER_TOOL_CALL:
            continue
        event = payload.get("event")
        tool = event.get("toolName") if isinstance(event, Mapping) else None
        if isinstance(tool, str) and tool:
            silent.setdefault(tool, []).append(sequence)
    entries.extend(
        LedgerEntry(
            status=LEDGER_UNPROVEN,
            statement=f"{tool}: no observable outcome",
            count=len(sequences),
            sequences=tuple(sequences),
        )
        for tool, sequences in silent.items()
    )
    return ProofLedger(entries=tuple(entries))


def _backgrounded(fact: OpenClawFact) -> bool:
    """A shell command sent to the background: its exit 0 only says it started."""
    command = fact.detail.get("command")
    if not isinstance(command, str):
        return False
    stripped = command.rstrip()
    return stripped.endswith("&") and not stripped.endswith("&&")


def _status(fact: OpenClawFact) -> str:
    if fact.kind is FactKind.COMMAND_SUCCEEDED and _backgrounded(fact):
        return LEDGER_UNPROVEN
    if fact.kind in {FactKind.COMMAND_SUCCEEDED, FactKind.FILE_WRITTEN}:
        return LEDGER_PROVEN
    return LEDGER_FAILED


def _statement(fact: OpenClawFact) -> str:
    detail = fact.detail
    command = detail.get("command") or "a command"
    if fact.kind is FactKind.COMMAND_SUCCEEDED and _backgrounded(fact):
        return f"`{command}` started in the background: whether it ran is not observed"
    if fact.kind is FactKind.COMMAND_SUCCEEDED:
        return f"`{command}` exited 0"
    if fact.kind is FactKind.COMMAND_FAILED:
        kind = f", {detail['failure_kind']}" if detail.get("failure_kind") else ""
        return f"`{command}` exited {detail.get('exit_code')}{kind}"
    if fact.kind is FactKind.COMMAND_TIMED_OUT:
        return f"`{command}` timed out"
    if fact.kind is FactKind.FILE_WRITTEN:
        return f"wrote {detail.get('bytes')} bytes to {detail.get('path')}"
    if fact.kind is FactKind.CALL_BLOCKED:
        if detail.get("by_pathlayer"):
            return f"{fact.tool} call blocked by Pathlayer"
        denied = detail.get("denied_reason") or "no reason given"
        return f"{fact.tool} call blocked by OpenClaw ({denied})"
    return f"{fact.tool} failed: {detail.get('error')}"
