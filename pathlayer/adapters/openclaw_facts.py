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
- A fact says what a call did, never that the task was done. An exit code of 0
  proves that a command exited 0; a write proves that bytes were written at
  that moment, not what the file holds now.

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

# Longest command or error text kept in a fact.
MAX_FACT_TEXT_CHARS = 200

# "Successfully wrote 1113 bytes to chess/index.html": OpenClaw's write tool.
_WROTE = re.compile(r"^Successfully wrote (\d+) bytes to (.+)$")


class FactKind(str, Enum):
    """What one tool call established."""

    COMMAND_SUCCEEDED = "command_succeeded"
    COMMAND_FAILED = "command_failed"
    COMMAND_TIMED_OUT = "command_timed_out"
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

    if details.get("status") == "blocked":
        reason = details.get("reason")
        denied = details.get("deniedReason")
        return make(
            FactKind.CALL_BLOCKED,
            {
                "denied_reason": denied if isinstance(denied, str) else None,
                "reason": _clip(reason) if isinstance(reason, str) else None,
                "by_pathlayer": isinstance(reason, str) and reason.startswith("Pathlayer "),
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
    if details.get("timedOut") is True:
        return make(FactKind.COMMAND_TIMED_OUT, base)
    exit_code = details.get("exitCode")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        # No exit code: a background command, a signal, an unknown runtime.
        # Nothing observed about how it ended.
        error = event.get("error")
        if isinstance(error, str) and error.strip():
            return make(FactKind.CALL_FAILED, {**base, "error": _clip(error.strip())})
        return None
    if exit_code == 0:
        return make(FactKind.COMMAND_SUCCEEDED, {**base, "exit_code": 0})
    failure_kind = details.get("failureKind")
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
            "bytes": int(match.group(1)),
            # The content the agent asked to write: what was written at that
            # moment, not what the file holds now.
            "content_sha256": _sha256(content) if isinstance(content, str) else None,
        },
    )


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
