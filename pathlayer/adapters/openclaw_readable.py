"""A plain-language view of one recorded OpenClaw run, for ``pathlayer-report``.

The session report (:class:`~pathlayer.postmortem.SessionReport`) is written
for whoever tunes Pathlayer: detector readings per tick, the fused score, event
positions. This module renders the same run for whoever *ran the agent*:

- **one status**, built from the manifest's execution ``status``, the run's own
  ``outcome`` and its failed attempts, instead of two overlapping fields;
- **clock times** instead of event positions. An event's ``ts`` is its position
  in the run, never a time; the time the listener *received* each analyzed
  event is in ``decisions.jsonl``, and that is the time shown (UTC);
- **what Pathlayer did**, said explicitly: a stop it decided, whether OpenClaw
  was told to enforce it, each tool call it blocked;
- internal vocabulary (fused score, ticks, peaks, detectors wired) only in the
  "technical details" appendix, which is the session report's own text,
  unchanged.

Everything here is rendering over the run directory: no detector, no
threshold, no clock. It never says the task succeeded -- OpenClaw's ``success``
is OpenClaw's claim, and is reported as such.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pathlayer.adapters.openclaw_facts import FactKind, extract_facts
from pathlayer.adapters.openclaw_lifecycle import AFTER_TOOL_CALL
from pathlayer.adapters.openclaw_run_store import (
    INTERRUPTED_WHILE_AWAITING_RETRY,
    RunDirectory,
    parse_iso,
    read_journal,
)
from pathlayer.policies import Action
from pathlayer.postmortem import (
    AWAITING_RETRY_STATUS,
    COMPLETED_STATUS,
    ActionTaken,
    BlockedToolCall,
    SessionOutcome,
    SessionReport,
    ToolCallGate,
)

# Run outcomes the recorder writes to the manifest (``run_outcome``).
_OUTCOME_OK = "ok"
_OUTCOME_ERROR = "error"
_OUTCOME_UNKNOWN = "unknown"
INTERRUPTED_STATUS = "interrupted"

# An error text from OpenClaw is shown on the status line up to this length;
# the full (already capped) text stays in the manifest and the appendix.
_MAX_STATUS_ERROR_CHARS = 120
# Report notes about detector internals (scores, per-tick readings) are only
# printed in the technical appendix, which carries every note.
_TECHNICAL_NOTE_MARKERS = ("fused score", "tick", "detector confidence")
# A detector or gate reason is shown up to this length.
_MAX_REASON_CHARS = 120
# Blocked calls listed one by one; report.json lists all of them.
_MAX_BLOCKED_LINES = 10
# The gate's reason for a call refused by the cap on identical calls in flight.
_IN_FLIGHT_REASON_PREFIX = "identical calls in flight"

# What each detector that can drive an action means, in plain words. The
# detector's own reason follows in parentheses: it is the evidence.
_DETECTOR_MEANING: dict[str, str] = {
    "loop": "the agent kept repeating the same tool calls",
    "retry": "the same tool call kept failing the same way",
    "no_progress": "the agent made no observable progress",
    "budget": "the run's cost reached its configured limit",
    "resource": "the run reached its configured time or size limit",
    "progress_proof": "a declared progress checkpoint was not met",
    "coding_tests": "the tests showed no progress",
    "coding_exit_code": "the commands showed no progress",
    "coding_git": "the repository showed no progress",
    "coding_state": "the workspace showed no progress",
    "state_evidence": "the workspace showed no progress",
}

# On OpenClaw only a STOP has an actuator (the tool-call gate); every lighter
# action is recorded, never delivered to the agent, and the wording says so.
_ACTION_VERB: dict[Action, str] = {
    Action.WARN: "raised a warning (recorded only: OpenClaw has no channel for it)",
    Action.SUMMARIZE: "chose to ask for a summary (recorded only: not applied on OpenClaw)",
    Action.FALLBACK: "chose a fallback (recorded only: not applied on OpenClaw)",
    Action.ESCALATE: "chose to call a human (recorded only: not applied on OpenClaw)",
    Action.STOP: "decided to stop the run",
}

_DRY_RUN_VERB: dict[Action, str] = {
    Action.WARN: "would have warned",
    Action.SUMMARIZE: "would have asked the agent to summarize",
    Action.FALLBACK: "would have asked for a fallback",
    Action.ESCALATE: "would have asked for a human",
    Action.STOP: "would have stopped the run",
}


def event_times(run: RunDirectory) -> dict[int, str]:
    """When the listener received each analyzed event, by event position.

    From ``decisions.jsonl`` (``event_index`` -> ``received_at``). A corrupt or
    absent journal yields no time at all rather than a guessed one.
    """
    try:
        records, _ = read_journal(run.decisions_path)
    except (OSError, ValueError):
        return {}
    times: dict[int, str] = {}
    for record in records:
        index = record.get("event_index")
        at = record.get("received_at")
        if isinstance(index, bool) or not isinstance(index, (int, float)):
            continue
        if not math.isfinite(index):
            continue
        if isinstance(at, str):
            times.setdefault(int(index), at)
    return times


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def first_stop(run: RunDirectory) -> tuple[int | None, int | None]:
    """The raw sequence and event index of the run's first applied STOP decision.

    From ``decisions.jsonl``: the first analyzed, non-dry-run line whose
    arbitrated action is STOP. ``(None, None)`` when there is none, or when the
    journal cannot be read.
    """
    try:
        records, _ = read_journal(run.decisions_path)
    except (OSError, ValueError):
        return None, None
    for record in records:
        if not record.get("analyzed", True) or record.get("dry_run") is True:
            continue
        if record.get("arbitrated_action", record.get("action")) == Action.STOP.name:
            return _int_or_none(record.get("sequence")), _int_or_none(record.get("event_index"))
    return None, None


def tool_results_after(run: RunDirectory, sequence: int | None) -> int | None:
    """How many tool calls OpenClaw reported as run after raw record ``sequence``.

    Counted from ``events.raw.jsonl`` (``after_tool_call`` records). A blocked
    call comes back as a result too: one whose own result says it was blocked
    (by Pathlayer's gate) or denied (by OpenClaw) is not counted, since it
    never ran. A ``block`` in ``gate.jsonl`` is not enough: when the plugin's
    decision times out it fails open and the call runs, which is exactly what
    this count must show. A call delivered twice counts once. A result that arrives after a
    STOP decision comes from a call already running when Pathlayer decided, or
    one the stop did not block: the count says that calls went on, never which
    of the two. ``None`` without a stop or a readable journal.
    """
    if sequence is None:
        return None
    try:
        records, _ = read_journal(run.raw_events_path)
    except (OSError, ValueError):
        return None
    later = [
        record
        for record in records
        if (_int_or_none(record.get("sequence")) or 0) > sequence
        and isinstance(record.get("payload"), Mapping)
        and record["payload"].get("hook") == AFTER_TOOL_CALL
    ]
    not_run = {fact.sequence for fact in extract_facts(later) if fact.kind is FactKind.CALL_BLOCKED}
    ran: set[Any] = set()
    for record in later:
        if record.get("sequence") in not_run:
            continue
        event = record["payload"].get("event")
        call_id = event.get("toolCallId") if isinstance(event, Mapping) else None
        ran.add(call_id if isinstance(call_id, str) else ("sequence", record.get("sequence")))
    return len(ran)


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def stop_blocks(gate: ToolCallGate | None) -> int:
    """Calls the gate blocked because the run was stopped.

    The cap on identical calls in flight blocks calls too, before any stop:
    those are not the stop's doing.
    """
    if gate is None:
        return 0
    return sum(
        1 for call in gate.blocked if not (call.reason or "").startswith(_IN_FLIGHT_REASON_PREFIX)
    )


def _stop_wording(gate: ToolCallGate | None) -> str:
    """ "Stopped" only when OpenClaw was told to block, and something was blocked.

    An enforced STOP that no later tool call ran into blocked nothing: the
    agent ended on its own, and saying Pathlayer stopped it would be a claim
    the gate log does not support.
    """
    if gate is None:
        return "Stop decided by Pathlayer (enforcement unknown: no gate log)"
    if not gate.enforced:
        return "Stop decided by Pathlayer (not enforced: recorded only)"
    if stop_blocks(gate):
        return "Stopped by Pathlayer"
    return "Stop decided by Pathlayer (enforced, but no later tool call was attempted)"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def display_status(manifest: Mapping[str, Any], report: SessionReport) -> str:
    """The run's one status line, from its execution status and its outcome.

    The manifest says how far the run got (``status``) and how it ended
    (``outcome``); a reader needs one answer. A stop is named first because it
    is what Pathlayer did; an incomplete record is never hidden behind it.
    """
    status = str(manifest.get("status") or "")
    outcome = manifest.get("outcome")
    failed = manifest.get("failed_attempts")
    failed_attempts = failed if isinstance(failed, int) and not isinstance(failed, bool) else 0
    interrupted_waiting = manifest.get(INTERRUPTED_WHILE_AWAITING_RETRY) is True

    if status == AWAITING_RETRY_STATUS:
        attempts = f"attempt {failed_attempts}" if failed_attempts else "the last attempt"
        line = f"Waiting for a retry: {attempts} did not succeed"
        if manifest.get("stopped") is True:
            line += f"; {_lower_first(_stop_wording(report.tool_call_gate))}"
        return line
    if manifest.get("stopped") is True or report.outcome is SessionOutcome.STOPPED:
        line = _stop_wording(report.tool_call_gate)
        if status != COMPLETED_STATUS:
            line += " -- record incomplete: the run never reached its end"
        return line
    if status == COMPLETED_STATUS:
        if outcome == _OUTCOME_ERROR:
            error = manifest.get("error")
            detail = f": {_clip(error, _MAX_STATUS_ERROR_CHARS)}" if isinstance(error, str) else ""
            return f"Ended with an error reported by OpenClaw{detail}"
        if outcome == _OUTCOME_OK:
            after = f" after {failed_attempts} failed attempt(s)" if failed_attempts else ""
            return f"Finished (OpenClaw reported success){after}"
        if outcome == _OUTCOME_UNKNOWN:
            return "Ended without success, and without an error (for example, aborted)"
        return "Finished"
    if status == INTERRUPTED_STATUS:
        if interrupted_waiting:
            return "Interrupted while waiting for a retry: the listener restarted"
        return "Interrupted: the run never reached its end (the listener stopped or restarted)"
    return "No end recorded: the run may still be running, or its listener stopped"


def _clock(at: str | None, start_day: str | None) -> str | None:
    """``HH:MM:SS UTC``, with the date when it is not the run's start day."""
    moment = parse_iso(at)
    if moment is None:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    day = moment.date().isoformat()
    clock = moment.strftime("%H:%M:%S")
    return f"{clock} UTC" if day == start_day else f"{day} {clock} UTC"


def _start_day(started_at: Any) -> str | None:
    moment = parse_iso(started_at)
    return moment.date().isoformat() if moment is not None else None


def _plain_reason(reason: str) -> str:
    """A detector's reason in plain words, keeping the reason itself as evidence.

    The reason is clipped: a retry reason quotes the tool's error, which can be
    long and is kept whole in ``report.json`` and the appendix.
    """
    name, sep, _ = reason.partition(":")
    reason = _clip(reason, _MAX_REASON_CHARS)
    meaning = _DETECTOR_MEANING.get(name.strip()) if sep or reason in _DETECTOR_MEANING else None
    if meaning is None:
        return reason or "no reason recorded"
    return f"{meaning} ({reason})"


def _action_line(action: ActionTaken, times: Mapping[int, str], start_day: str | None) -> str:
    when = _clock(times.get(action.index), start_day) or "time unknown"
    if action.emitted:
        verb = _ACTION_VERB.get(action.action, action.action.name.lower())
    else:
        verb = _DRY_RUN_VERB.get(action.action, f"would have chosen {action.action.name}")
        verb += " (observe-only mode, never applied)"
    reason = f": {_plain_reason(action.reason)}" if action.reason else ""
    return f"  {when}  Pathlayer {verb}{reason}"


def _blocked_line(call: BlockedToolCall, start_day: str | None) -> str:
    when = _clock(call.at, start_day) or "time unknown"
    tool = call.tool_name or "unknown tool"
    reason = call.reason or ""
    if reason.startswith(_IN_FLIGHT_REASON_PREFIX):
        why = "the same call with the same arguments was already running"
    elif reason:
        why = f"the run was stopped ({_clip(reason, _MAX_REASON_CHARS)})"
    else:
        why = "the run was stopped"
    return f"  {when}  {tool} -- {why}"


def _gate_lines(gate: ToolCallGate | None, start_day: str | None) -> list[str]:
    if gate is None:
        return []
    blocked = len(gate.blocked)
    lines = [
        "",
        f"Calls blocked by Pathlayer: {blocked} of {gate.queries} tool call(s) "
        "OpenClaw asked about",
    ]
    if not gate.enforced:
        lines.append("  not enforced: recorded only, OpenClaw ran every call")
    for call in gate.blocked[:_MAX_BLOCKED_LINES]:
        lines.append(_blocked_line(call, start_day))
    hidden = blocked - _MAX_BLOCKED_LINES
    if hidden > 0:
        lines.append(f"  ... and {hidden} more blocked call(s) (all listed in report.json)")
    if gate.would_have_blocked:
        lines.append(f"  {gate.would_have_blocked} call(s) ran that enforcement would have blocked")
    return lines


def _cost_line(report: SessionReport) -> str:
    total = report.cost.total_usd
    if total is None:
        return "Cost: not measured"
    return f"Cost: ${total:.4f} (an estimate at local prices; its sources with --technical)"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "duration unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{rest:02d}s" if hours else f"{minutes}m{rest:02d}s"


def readable_fields(run: RunDirectory, report: SessionReport) -> dict[str, Any]:
    """The plain-language additions to ``report.json``: status and clock times.

    Additive only: ``display_status`` plus an ``at`` (UTC, as received) on each
    action and the first and last ``at`` of each finding. ``None`` where no
    time was journaled.
    """
    manifest = run.read_manifest()
    times = event_times(run)
    return {
        "display_status": display_status(manifest, report),
        "action_times": [times.get(action.index) for action in report.actions],
        "finding_times": [
            [times.get(finding.first_index), times.get(finding.last_index)]
            for finding in report.findings
        ],
    }


def report_payload(run: RunDirectory, report: SessionReport) -> dict[str, Any]:
    """``report.json``: the session report, plus its plain-language fields."""
    payload = report.to_dict()
    extra = readable_fields(run, report)
    payload["display_status"] = extra["display_status"]
    for entry, at in zip(payload["actions"], extra["action_times"], strict=True):
        entry["at"] = at
    for entry, (first, last) in zip(payload["findings"], extra["finding_times"], strict=True):
        entry["first_at"] = first
        entry["last_at"] = last
    return payload


def _is_technical(note: str) -> bool:
    """A report note about detector internals, kept for the appendix."""
    lowered = note.lower()
    return any(marker in lowered for marker in _TECHNICAL_NOTE_MARKERS)


def _model(manifest: Mapping[str, Any]) -> str | None:
    model = manifest.get("openclaw_model")
    provider = manifest.get("openclaw_provider")
    if isinstance(model, str) and isinstance(provider, str):
        return f"{model} ({provider})"
    return model if isinstance(model, str) else None


def render_run_text(run: RunDirectory, report: SessionReport, *, technical: bool = False) -> str:
    """The text ``pathlayer-report <run>`` prints: plain words first.

    ``technical`` appends the session report's own text, with every detector
    reading, as an appendix.
    """
    manifest = run.read_manifest()
    times = event_times(run)
    start_day = _start_day(report.started_at)
    identity = [f"session {report.session_key}"] if report.session_key else []
    if report.agent_id:
        identity.append(f"agent {report.agent_id}")
    header = f"Run {report.run_id}" + (f" ({', '.join(identity)})" if identity else "")

    started = parse_iso(report.started_at)
    start_text = (
        started.strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(started, datetime) else "unknown"
    )
    ended = _clock(report.ended_at, start_day)
    if report.status == COMPLETED_STATUS:
        end_text = ended or "unknown"
    elif report.status == INTERRUPTED_STATUS and ended:
        end_text = f"not recorded (interruption noticed at {ended})"
    else:
        end_text = "not yet"
    lines = [
        header,
        f"Status: {display_status(manifest, report)}",
        f"Started: {start_text}   Ended: {end_text}   ({_duration(report.duration_seconds)})",
    ]
    model = _model(manifest)
    if model:
        lines.append(f"Model: {model}")

    lines.extend(["", "What Pathlayer did"])
    gate = report.tool_call_gate
    if report.actions:
        lines.extend(_action_line(action, times, start_day) for action in report.actions)
        after = tool_results_after(run, first_stop(run)[0])
        if after:
            lines.append(
                f"  after the stop decision, {after} more tool call(s) ran (already running "
                "when Pathlayer decided, or not blocked)"
            )
    elif gate is not None and gate.blocked:
        # A call refused by the in-flight cap is an intervention without any
        # detector action: never say "nothing" above the list of blocked calls.
        lines.append("  no detector asked for an action; the tool-call gate blocked calls (below)")
    elif report.events_analyzed == 0:
        lines.append("  nothing: no event of this run was analyzed")
    else:
        lines.append(
            f"  nothing: it analyzed {report.events_analyzed} event(s) (tool calls and model "
            "replies) and never intervened"
        )
    lines.extend(_gate_lines(report.tool_call_gate, start_day))

    if report.proof_ledger is not None:
        ledger = report.proof_ledger.describe_lines()
        lines.extend(["", "Proof ledger (observed facts, not a verdict on the task)", *ledger[1:]])

    lines.extend(["", _cost_line(report)])
    plain_notes = [note for note in report.notes if not _is_technical(note)]
    if plain_notes:
        lines.extend(["", "Notes"])
        lines.extend(f"  - {note}" for note in plain_notes)
    if technical:
        lines.extend(["", "Technical details", report.format()])
    else:
        lines.extend(
            [
                "",
                f"Detector readings: pathlayer-report {report.run_id} --technical",
            ]
        )
    return "\n".join(lines)


def summary_line(run: RunDirectory, report: SessionReport) -> str:
    """One line per run for a listing: start, run id, status, events, cost."""
    try:
        manifest = run.read_manifest()
    except (OSError, ValueError):
        manifest = {}
    when = report.started_at or "unknown start"
    total = report.cost.total_usd
    cost = f"${total:.4f}" if total is not None else "cost n/a"
    return (
        f"{when}  {report.run_id}  {report.events_recorded} event(s)  {cost}  "
        f"{display_status(manifest, report)}"
    )


__all__ = [
    "display_status",
    "event_times",
    "first_stop",
    "readable_fields",
    "render_run_text",
    "report_payload",
    "stop_blocks",
    "summary_line",
    "tool_results_after",
]
