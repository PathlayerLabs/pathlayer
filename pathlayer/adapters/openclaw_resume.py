"""The resume dossier of an OpenClaw run that did not end well.

After a stop, an error or an interruption, whoever ran the agent needs one
document that answers: why did it stop, what is proven done, what failed, what
was tried, what did Pathlayer do, and how to restart without repeating the
same failure. This module assembles that document from the run directory
alone -- the same files the report reads -- and writes it as ``resume.json``
(schema :data:`SCHEMA`) and ``resume.md``.

Three rules shape it:

- **Every section says how it is known.** ``observed`` (the runtime reported
  it), ``decided`` (Pathlayer chose it), ``proven`` (the stop rests on facts a
  reader can check: identical calls for ``loop``, identical failures for
  ``retry``, an unmet checkpoint for a contract) or ``unknown``. A
  ``no_progress`` stop is ``decided``: it rests on the absence of progress,
  which is an inference, not a fact.
- **It never says the task succeeded, and never promises a restart works.**
  The ledger proves what tool calls did, not that the work is right; the
  options are suggestions.
- **Nothing raw leaves the tool calls.** Arguments appear only as a digest; an
  error or reason is clipped to :data:`MAX_TEXT_CHARS` with token-like strings
  masked; the ledger keeps the report's own statements (commands and paths,
  clipped), masked the same way. The 3-line summary carries no argument and no
  error text at all, since lot 3 sends it back into the agent's session.

:func:`build_resume_dossier` is pure over the files: no clock, no network, no
randomness. :func:`write_resume_dossier` is the only writer, and removes a
stale dossier when the run no longer needs one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pathlayer.adapters.openclaw_facts import FactKind, OpenClawFact, extract_facts
from pathlayer.adapters.openclaw_lifecycle import AGENT_END, attempt_end_error
from pathlayer.adapters.openclaw_readable import (
    clock_text,
    detector_meaning,
    display_status,
    event_times,
    first_stop,
    start_day_of,
    tool_results_after,
)
from pathlayer.adapters.openclaw_report import build_run_report, llm_cost_split
from pathlayer.adapters.openclaw_run_store import (
    RunDirectory,
    atomic_write_json,
    atomic_write_text,
    parse_iso,
    read_journal,
)
from pathlayer.policies import Action
from pathlayer.postmortem import (
    AWAITING_RETRY_STATUS,
    COMPLETED_STATUS,
    LEDGER_FAILED,
    LEDGER_PROVEN,
    LEDGER_UNPROVEN,
    CostShare,
    SessionReport,
)

SCHEMA = "pathlayer.resume/1"

OBSERVED = "observed"
DECIDED = "decided"
PROVEN = "proven"
UNKNOWN = "unknown"

# Longest error, reason or statement text kept in a dossier.
MAX_TEXT_CHARS = 120
# Longest ledger statement kept (commands and paths are what make it useful).
MAX_STATEMENT_CHARS = 200
# Groups of attempted calls, and blocked calls, listed one by one.
MAX_LISTED = 20
# Shown length of a params digest: enough to tell groups apart.
DIGEST_CHARS = 12

INTERRUPTED_STATUS = "interrupted"

# Causes, in the order they are looked for: the first that applies is the one
# the dossier is about.
CAUSE_STOP = "pathlayer_stop"
CAUSE_ERROR = "openclaw_error"
CAUSE_NO_SUCCESS = "ended_without_success"
CAUSE_AWAITING_RETRY = "awaiting_retry"
CAUSE_INTERRUPTED = "interrupted"
CAUSE_BLOCKED = "blocked_calls"

# Stop reasons that rest on facts a reader can check: the loop detector's exact
# repeat component (``Nx repeat of a K-action block``) and the retry detector's
# identical-failure component. The same detectors' other components (a
# recurring pattern over overlapping windows, one tool dominating the recent
# calls) are inferences, and stay ``decided``. A progress contract surfaces as a
# ``no_progress`` stall, which is an inference too.
_PROVEN_REASONS = (
    re.compile(r"^loop: \d+x repeat of "),
    re.compile(r"^retry: \d+x identical failure of "),
)

_MASKED = "[masked]"
_SECRET_PATTERNS = (
    # ``NAME=value`` / ``Name: value`` where the name says it is a secret.
    re.compile(
        r"(?i)\b([A-Za-z0-9_-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)"
        r"[A-Za-z0-9_-]*)(\s*[=:]\s*)(?!\[masked\])\S+"
    ),
    # user:password@ in a URL.
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+@"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/]{8,}=*"),
    re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
    # A long base64-like run with at least one digit; ``/`` and ``-`` are left
    # out so that file paths and slugs stay readable.
    re.compile(r"(?<![A-Za-z0-9+_])(?=[A-Za-z0-9+_]*\d)[A-Za-z0-9+_]{40,}={0,2}"),
)


def mask_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    """One line of text, token-like strings masked, clipped to ``limit``.

    Masking runs before clipping, so a secret cut in half by the clip is never
    shown half-masked. Control characters become spaces: the text cannot forge
    a line of the Markdown dossier.
    """
    text = " ".join(str(value).split())
    text = _SECRET_PATTERNS[0].sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASKED}", text)
    text = _SECRET_PATTERNS[1].sub(f"{_MASKED}@", text)
    for pattern in _SECRET_PATTERNS[2:]:
        text = pattern.sub(_MASKED, text)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# --------------------------------------------------------------------------- #
# Templates: how to restart, per cause
# --------------------------------------------------------------------------- #

# Each cause: (the decision a person has to make, in one line; the options).
# Wording rule, enforced by a test: suggest, never promise. No "impossible",
# no "guaranteed", no "good state", no "safe to".
_TEMPLATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "loop": (
        "change the approach before resuming: the agent was repeating itself",
        (
            "Do not resume with the same calls: the ones grouped under 'What was tried' "
            "repeated without moving on.",
            "Change the approach or the input (a different command, a narrower task), then "
            "resume with a small number of calls.",
            "Use the proof ledger to tell the agent what is already done, so it is not redone.",
        ),
    ),
    "retry": (
        "fix the failing call before resuming: it failed the same way several times",
        (
            "The same call kept failing with the same error: fix its cause (missing file, "
            "wrong path, missing tool, permissions) before resuming.",
            "If the call is not needed, tell the agent to skip it and continue from the last "
            "call without an error.",
        ),
    ),
    "no_progress": (
        "check the last outputs: Pathlayer saw no observable progress",
        (
            "Pathlayer saw no observable progress. Check whether the agent's last outputs "
            "moved the task forward.",
            "If they did, the detector may be too strict for this task: resume with a small "
            "number of calls, or tune no_progress.",
        ),
    ),
    "budget": (
        "raise the cost limit or narrow the task",
        (
            "The run reached its configured cost limit. Raise the limit, or narrow the task, "
            "before resuming.",
        ),
    ),
    "resource": (
        "raise the time or size limit, or split the task",
        (
            "The run reached its configured time or size limit. Split the task into smaller "
            "runs, or raise the limit.",
        ),
    ),
    "stop": (
        "read why Pathlayer stopped the run before resuming",
        ("Read the reason above, then resume with a small number of calls.",),
    ),
    "overflow": (
        "start a new session or compact the context: the model ran out of context",
        (
            "OpenClaw reported a context overflow. Resuming in the same session may hit the "
            "same limit: start a new session (/new) or compact first.",
            "Give the new session the proof ledger, so it does not redo what is done.",
        ),
    ),
    CAUSE_ERROR: (
        "read the provider error before retrying",
        (
            "OpenClaw reported an error on the last attempt. Retrying unchanged may fail the "
            "same way: read the error first.",
        ),
    ),
    CAUSE_NO_SUCCESS: (
        "decide whether the run should continue: it ended without success or error",
        (
            "The last attempt ended without success and without an error, which is how "
            "OpenClaw reports an abort. Resume only if the run should go on.",
        ),
    ),
    CAUSE_AWAITING_RETRY: (
        "wait: OpenClaw may start another attempt by itself",
        (
            "The last attempt did not succeed and OpenClaw may start another one by itself. "
            "This dossier is provisional and is rewritten when the run ends.",
        ),
    ),
    CAUSE_INTERRUPTED: (
        "check the workspace: what happened after the last record is unknown",
        (
            "The listener stopped while the run was going on: whatever happened after its "
            "last record is not observed. Check the workspace before resuming.",
        ),
    ),
    CAUSE_BLOCKED: (
        "let the calls already running finish instead of repeating them",
        (
            "Pathlayer blocked calls identical to ones already running. Use their results "
            "rather than calling them again.",
        ),
    ),
}

_ALWAYS = (
    "Anything under 'unproven' in the proof ledger is not observed: check it by hand "
    "before relying on it.",
)


def template_texts() -> list[str]:
    """Every sentence a dossier can carry from a template, for wording tests."""
    texts = list(_ALWAYS)
    for decision, options in _TEMPLATES.values():
        texts.append(decision)
        texts.extend(options)
    return texts


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def needs_dossier(manifest: Mapping[str, Any], report: SessionReport) -> bool:
    """True for a run that did not end well; a healthy run gets no dossier.

    Healthy means: it ended (``completed``), OpenClaw reported success, and
    Pathlayer neither stopped it nor blocked a call. A run still in flight and
    not stopped has nothing to resume yet. Failed attempts followed by a
    successful one are not enough: the run recovered on its own.
    """
    status = str(manifest.get("status") or "")
    gate = report.tool_call_gate
    if manifest.get("stopped") is True or report.stop_reason is not None:
        return True
    if gate is not None and gate.blocked:
        return True
    if status in {AWAITING_RETRY_STATUS, INTERRUPTED_STATUS}:
        return True
    if status == COMPLETED_STATUS:
        outcome = manifest.get("outcome")
        if outcome is None:
            # A manifest older than ``outcome``: only an error says it went wrong.
            return isinstance(manifest.get("error"), str)
        return outcome not in {"ok", "stopped"}
    return False


def _detector(reason: str | None) -> str | None:
    if not reason:
        return None
    name, sep, _ = reason.partition(":")
    return name.strip() if sep else None


def _is_overflow(error: str | None) -> bool:
    lowered = (error or "").lower()
    return "overflow" in lowered or "context length" in lowered or "too many tokens" in lowered


def _cause(manifest: Mapping[str, Any], report: SessionReport) -> str:
    status = str(manifest.get("status") or "")
    if manifest.get("stopped") is True or report.stop_reason is not None:
        return CAUSE_STOP
    if status == COMPLETED_STATUS and (
        manifest.get("outcome") == "error"
        or (manifest.get("outcome") is None and isinstance(manifest.get("error"), str))
    ):
        return CAUSE_ERROR
    if status == COMPLETED_STATUS and manifest.get("outcome") == "unknown":
        return CAUSE_NO_SUCCESS
    if status == AWAITING_RETRY_STATUS:
        return CAUSE_AWAITING_RETRY
    if status == INTERRUPTED_STATUS:
        return CAUSE_INTERRUPTED
    return CAUSE_BLOCKED


def _is_proven(reason: str | None) -> bool:
    return reason is not None and any(pattern.match(reason) for pattern in _PROVEN_REASONS)


def _template_key(
    cause: str, detector: str | None, error: str | None, *, proven: bool = True
) -> str:
    if cause == CAUSE_STOP:
        if detector in {"loop", "retry"} and not proven:
            return "stop"
        return detector if detector in _TEMPLATES else "stop"
    if cause in {CAUSE_ERROR, CAUSE_AWAITING_RETRY} and _is_overflow(error):
        return "overflow"
    return cause


def _section(
    status: str,
    cause: str,
    headline: str,
    detail: str | None,
    *,
    detector: str | None = None,
    reason: str | None = None,
    at: Any = None,
    enforcement: str | None = None,
    tool_calls_after_stop: int | None = None,
    last_record_at: str | None = None,
    noticed_at: Any = None,
) -> dict[str, Any]:
    """``headline`` and ``detail`` are template text only: the summary reuses them."""
    return {
        "status": status,
        "cause": cause,
        "detector": detector,
        "headline": headline,
        "detail": detail,
        "explanation": f"{headline}: {detail}" if detail else headline,
        "reason": reason,
        "at": at if isinstance(at, str) else None,
        "enforcement": enforcement,
        "tool_calls_after_stop": tool_calls_after_stop,
        "last_record_at": last_record_at,
        "noticed_at": noticed_at if isinstance(noticed_at, str) else None,
    }


def _enforcement(gate: Any) -> str:
    """Whether OpenClaw was told to block the run's calls after the stop."""
    if gate is None:
        return "unknown: no gate log"
    if not gate.enforced:
        return "not enforced: recorded only"
    if gate.blocked:
        return f"enforced: {len(gate.blocked)} call(s) blocked"
    return "enforced, but no later tool call was attempted"


def _why_stopped(
    cause: str,
    manifest: Mapping[str, Any],
    report: SessionReport,
    stop_at: str | None,
    after_stop: int | None,
    last_record_at: str | None = None,
) -> dict[str, Any]:
    error = manifest.get("error") if isinstance(manifest.get("error"), str) else None
    if cause == CAUSE_STOP:
        gate = report.tool_call_gate
        # "Stopped" only when the gate blocked a call; otherwise only the
        # decision is on record.
        enforced = gate is not None and gate.enforced and bool(gate.blocked)
        return _section(
            PROVEN if _is_proven(report.stop_reason) else DECIDED,
            cause,
            "Pathlayer stopped the run" if enforced else "Pathlayer decided to stop the run",
            detector_meaning(report.stop_reason or ""),
            detector=_detector(report.stop_reason),
            reason=mask_text(report.stop_reason) if report.stop_reason else None,
            at=stop_at,
            enforcement=_enforcement(gate),
            tool_calls_after_stop=after_stop,
        )
    if cause == CAUSE_ERROR:
        return _section(
            OBSERVED,
            cause,
            "Ended with an error reported by OpenClaw",
            "a context overflow" if _is_overflow(error) else None,
            reason=mask_text(error) if error else None,
            at=manifest.get("ended_at"),
        )
    if cause == CAUSE_NO_SUCCESS:
        return _section(
            OBSERVED,
            cause,
            "Ended without success",
            "the last attempt reported neither success nor an error",
            at=manifest.get("ended_at"),
        )
    if cause == CAUSE_AWAITING_RETRY:
        return _section(
            OBSERVED,
            cause,
            "Waiting for a retry",
            "the last attempt did not succeed and OpenClaw may start another one",
            reason=mask_text(error) if error else None,
        )
    if cause == CAUSE_INTERRUPTED:
        return _section(
            UNKNOWN,
            cause,
            "Interrupted",
            "the listener stopped or restarted before the run ended, so how it ended is "
            "not observed",
            last_record_at=last_record_at,
            # The recorder stamps an interrupted run's end when it notices it.
            noticed_at=manifest.get("ended_at"),
        )
    return _section(
        DECIDED,
        cause,
        "Calls blocked by Pathlayer",
        "tool calls identical to ones already running were blocked",
    )


def _received_at(raw: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    times: dict[int, str] = {}
    for record in raw:
        sequence = record.get("sequence")
        at = record.get("received_at")
        if isinstance(sequence, int) and not isinstance(sequence, bool) and isinstance(at, str):
            times[sequence] = at
    return times


def _fact_status(fact: OpenClawFact) -> str:
    if fact.kind is FactKind.COMMAND_BACKGROUNDED:
        return LEDGER_UNPROVEN
    if fact.kind in {
        FactKind.COMMAND_SUCCEEDED,
        FactKind.FILE_WRITTEN,
        FactKind.FILE_EDITED,
        FactKind.FILE_READ,
    }:
        return LEDGER_PROVEN
    return LEDGER_FAILED


def _attempts(
    gate_records: Sequence[Mapping[str, Any]],
    facts: Sequence[OpenClawFact],
    received: Mapping[int, str],
) -> dict[str, Any]:
    """The tool calls the run tried, grouped by tool and params digest.

    From ``gate.jsonl``, where the plugin sent a digest of each call's
    arguments, never the arguments. Without a gate log the calls are grouped
    by tool only, from the facts.
    """
    failed_ids = {
        fact.tool_call_id
        for fact in facts
        if fact.tool_call_id is not None and _fact_status(fact) == LEDGER_FAILED
    }
    groups: dict[tuple[str, str | None], dict[str, Any]] = {}
    if gate_records:
        for record in gate_records:
            name = record.get("tool_name")
            tool = name if isinstance(name, str) else "?"
            digest = record.get("params_digest")
            digest = digest if isinstance(digest, str) and digest else None
            group = groups.setdefault(
                (tool, digest),
                {
                    "tool": tool,
                    "params_digest": digest[:DIGEST_CHARS] if digest else None,
                    "calls": 0,
                    "blocked": 0,
                    "failed": 0,
                    "first_at": None,
                    "last_at": None,
                },
            )
            group["calls"] += 1
            group["blocked"] += int(record.get("decision") == "block")
            group["failed"] += int(record.get("tool_call_id") in failed_ids)
            at = record.get("answered_at") if isinstance(record.get("answered_at"), str) else None
            group["first_at"] = group["first_at"] or at
            group["last_at"] = at or group["last_at"]
        source = "gate.jsonl"
    else:
        for fact in facts:
            group = groups.setdefault(
                (fact.tool, None),
                {
                    "tool": fact.tool,
                    "params_digest": None,
                    "calls": 0,
                    "blocked": 0,
                    "failed": 0,
                    "first_at": None,
                    "last_at": None,
                },
            )
            group["calls"] += 1
            group["failed"] += int(_fact_status(fact) == LEDGER_FAILED)
            at = received.get(fact.sequence)
            group["first_at"] = group["first_at"] or at
            group["last_at"] = at or group["last_at"]
        source = "events.raw.jsonl"
    ordered = sorted(groups.values(), key=lambda g: (-g["calls"], g["first_at"] or "", g["tool"]))
    return {
        "status": OBSERVED if groups else UNKNOWN,
        "source": source,
        "digest_scope": (
            "a digest groups calls with the same tool and arguments within one OpenClaw plugin "
            "process; after a gateway restart the same call gets another digest"
        ),
        "groups": ordered[:MAX_LISTED],
        "more_groups": max(0, len(ordered) - MAX_LISTED),
    }


def _last_clean_call(facts: Sequence[OpenClawFact], times: Mapping[int, str]) -> dict[str, Any]:
    for fact in reversed(facts):
        if _fact_status(fact) == LEDGER_PROVEN:
            return {
                "status": OBSERVED,
                "event_id": fact.event_id,
                "tool": fact.tool,
                "kind": fact.kind.value,
                "at": times.get(fact.sequence),
            }
    return {"status": UNKNOWN, "event_id": None, "tool": None, "kind": None, "at": None}


def _ledger(report: SessionReport) -> dict[str, Any]:
    ledger = report.proof_ledger
    if ledger is None:
        return {"status": UNKNOWN, LEDGER_PROVEN: [], LEDGER_FAILED: [], LEDGER_UNPROVEN: []}
    section: dict[str, Any] = {"status": OBSERVED}
    for column in (LEDGER_PROVEN, LEDGER_FAILED, LEDGER_UNPROVEN):
        section[column] = [
            {
                "statement": mask_text(entry.statement, MAX_STATEMENT_CHARS),
                "count": entry.count,
                "event_ids": [f"openclaw:{sequence}:0" for sequence in entry.sequences],
            }
            for entry in ledger.column(column)
        ]
    return section


def _delivery(action: Action, emitted: bool, gate: Any) -> str:
    """Whether OpenClaw was told: only a STOP has an actuator there, the gate."""
    if not emitted:
        return "observe-only: never applied"
    if action is not Action.STOP:
        return "recorded only: OpenClaw has no channel for it"
    if gate is None:
        return "enforcement unknown: no gate log"
    if gate.enforced:
        return "enforced by the tool-call gate"
    return "recorded only: not enforced"


def _actions(report: SessionReport, times: Mapping[int, str]) -> dict[str, Any]:
    gate = report.tool_call_gate
    blocked = list(gate.blocked) if gate is not None else []
    return {
        "status": DECIDED,
        "actions": [
            {
                "action": action.action.name,
                "at": times.get(action.index),
                "reason": mask_text(action.reason) if action.reason else None,
                "emitted": action.emitted,
                "delivery": _delivery(action.action, action.emitted, gate),
            }
            for action in report.actions
        ],
        "gate": None
        if gate is None
        else {
            "enforced": gate.enforced,
            "tool_calls_asked_about": gate.queries,
            "blocked": len(gate.blocked),
            "would_have_blocked": gate.would_have_blocked,
        },
        "blocked_calls": [
            {
                "tool": call.tool_name,
                "at": call.at,
                "why": "the same call was already running"
                if (call.reason or "").startswith("identical calls in flight")
                else "the run was stopped",
            }
            for call in blocked[:MAX_LISTED]
        ],
        "more_blocked_calls": max(0, len(blocked) - MAX_LISTED),
    }


def _share(share: CostShare) -> dict[str, Any]:
    return {
        "usd": share.total_usd,
        "costed_events": share.costed_events,
        "unmeasured_events": share.unmeasured_events,
    }


def _cost(run: RunDirectory, stop_sequence: int | None, stopped: bool) -> dict[str, Any]:
    try:
        before, after = llm_cost_split(run, stop_sequence if stopped else None)
    except (OSError, ValueError):
        return {"status": UNKNOWN, "estimated": True, "before_stop": None, "after_stop": None}
    measured = bool(before.costed_events or after.costed_events)
    return {
        "status": OBSERVED if measured else UNKNOWN,
        "estimated": True,
        "split_by": "when each model usage report reached Pathlayer"
        if stopped
        else "no stop: everything is before",
        "before_stop": _share(before),
        "after_stop": _share(after) if stopped else None,
    }


def _unknowns(
    manifest: Mapping[str, Any],
    report: SessionReport,
    cause: str,
    cost: Mapping[str, Any],
) -> list[str]:
    unknowns = [
        "whether the task itself is done: the ledger says what tool calls did, not whether "
        "the work is right"
    ]
    if report.tool_call_gate is None:
        unknowns.append(
            "whether OpenClaw was told to block calls: this run has no gate log (recorded by "
            "an older listener, or no tool call was asked about)"
        )
    if cause == CAUSE_INTERRUPTED or manifest.get("interrupted_while_awaiting_retry") is True:
        unknowns.append("what the run did after the listener's last record")
    if cost.get("status") == UNKNOWN:
        unknowns.append("the cost: no model usage report could be priced")
    if str(manifest.get("status") or "") not in {COMPLETED_STATUS, INTERRUPTED_STATUS}:
        unknowns.append("how the run ends: this dossier is provisional")
    return unknowns


def _summary(dossier: Mapping[str, Any]) -> list[str]:
    """Three lines, no argument and no error text: what, facts, decision.

    Built from template text and counts only: lot 3 sends these lines back into
    the agent's own session.
    """
    why = dossier["why_stopped"]
    clock = clock_text(why.get("at"), start_day_of(dossier["identity"]["started_at"]))
    when = f" at {clock}" if clock else ""
    detail = f": {why['detail']}" if why.get("detail") else ""
    ledger = dossier["proof_ledger"]
    blocked = (dossier["pathlayer_actions"]["gate"] or {}).get("blocked", 0)
    counts = {
        column: sum(entry["count"] for entry in ledger[column])
        for column in (LEDGER_PROVEN, LEDGER_FAILED, LEDGER_UNPROVEN)
    }
    after = why.get("tool_calls_after_stop")
    ran_after = f"; {after} call(s) ran after the stop decision" if after else ""
    return [
        f"{why['headline']}{when}{detail}.",
        f"{counts[LEDGER_PROVEN]} proven, {counts[LEDGER_FAILED]} failed, "
        f"{counts[LEDGER_UNPROVEN]} unproven fact(s); {blocked} call(s) blocked by "
        f"Pathlayer{ran_after}.",
        f"Decision needed: {dossier['options']['decision']}.",
    ]


def _last_attempt_error(raw: Sequence[Mapping[str, Any]]) -> str | None:
    for record in reversed(raw):
        payload = record.get("payload")
        if isinstance(payload, Mapping) and payload.get("hook") == AGENT_END:
            event = payload.get("event")
            return attempt_end_error(dict(event)) if isinstance(event, Mapping) else None
    return None


def build_resume_dossier(
    run: RunDirectory, report: SessionReport | None = None
) -> dict[str, Any] | None:
    """The resume dossier of one run, rebuilt from its directory; ``None`` if healthy.

    ``report`` is the run's report when the caller has just built it from the
    same files, to avoid rebuilding it.
    """
    if report is None:
        report = build_run_report(run)
    manifest = run.read_manifest()
    if not needs_dossier(manifest, report):
        return None
    try:
        raw, _ = read_journal(run.raw_events_path)
    except ValueError:
        raw = []
    try:
        gate_records, _ = read_journal(run.gate_path)
    except ValueError:
        gate_records = []
    facts = extract_facts(raw)
    received = _received_at(raw)
    times = event_times(run)
    stop_sequence, stop_index = first_stop(run)
    stop_action = next((a for a in report.actions if a.action is Action.STOP and a.emitted), None)
    if stop_action is not None:
        stop_at = times.get(stop_action.index)
    else:
        stop_at = times.get(stop_index) if stop_index is not None else None

    cause = _cause(manifest, report)
    error = manifest.get("error") if isinstance(manifest.get("error"), str) else None
    if error is None:
        # Between two attempts the manifest has no error yet: the last
        # agent_end carries it.
        error = _last_attempt_error(raw)
    after_stop = tool_results_after(run, stop_sequence) if cause == CAUSE_STOP else None
    last_record_at = max(received.values(), default=None)
    why = _why_stopped(
        cause, {**manifest, "error": error}, report, stop_at, after_stop, last_record_at
    )
    decision, options = _TEMPLATES[
        _template_key(cause, why["detector"], error, proven=why["status"] == PROVEN)
    ]
    stopped = cause == CAUSE_STOP
    cost = _cost(run, stop_sequence, stopped)
    status = str(manifest.get("status") or "")
    model = manifest.get("openclaw_model")
    dossier: dict[str, Any] = {
        "schema": SCHEMA,
        "provisional": status != COMPLETED_STATUS and status != INTERRUPTED_STATUS,
        "identity": {
            "status": OBSERVED,
            "run_id": report.run_id,
            "session_key": mask_text(report.session_key, 200) if report.session_key else None,
            "agent_id": report.agent_id,
            "model": model if isinstance(model, str) else None,
            "provider": manifest.get("openclaw_provider"),
            "started_at": report.started_at,
            "ended_at": report.ended_at if status == COMPLETED_STATUS else None,
            "run_status": status,
            "outcome": manifest.get("outcome"),
            # The status line quotes OpenClaw's error: masked like any other text.
            "display_status": mask_text(display_status(manifest, report), 200),
            "pathlayer_version": manifest.get("pathlayer_version"),
        },
        "why_stopped": why,
        "attempts": _attempts(gate_records, facts, received),
        "last_clean_call": _last_clean_call(facts, received),
        "proof_ledger": _ledger(report),
        "pathlayer_actions": _actions(report, times),
        "cost": cost,
        "options": {"status": DECIDED, "decision": decision, "steps": [*options, *_ALWAYS]},
        "unknowns": {"status": UNKNOWN, "items": _unknowns(manifest, report, cause, cost)},
    }
    dossier["summary"] = _summary(dossier)
    return dossier


# --------------------------------------------------------------------------- #
# Rendering and writing
# --------------------------------------------------------------------------- #


def _md(text: Any) -> str:
    """Text safe inside a table cell or a code span: no backtick, no pipe."""
    return str(text).replace("`", "'").replace("|", "\\|")


# How each status reads in resume.md. resume.json keeps the bare status.
_HOW_WE_KNOW = {
    PROVEN: "proven by the records: the stop reason counts the identical calls it saw",
    DECIDED: "decided by Pathlayer: a judgement on the records, not a proof",
    OBSERVED: "observed: OpenClaw reported it",
    UNKNOWN: "not known: the records end before it",
}

# Ledger entries that differ only by the number in their last parenthesis
# (a file read back 11 times at 11 sizes) are one line in resume.md.
_TRAILING_NUMBER = re.compile(r"\d+(?=[^()]*\)\s*$)")


def _bullets(lines: Iterable[str]) -> list[str]:
    return [f"- {line}" for line in lines]


def _merged_entries(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(_TRAILING_NUMBER.sub("#", entry["statement"]), []).append(entry)
    lines = []
    for group in groups.values():
        total = sum(entry["count"] for entry in group)
        last = group[-1]["statement"]
        if len(group) > 1:
            lines.append(f"{last} (x{total}; the last one shown)")
        else:
            lines.append(f"{last} (x{total})" if total > 1 else last)
    return lines


def render_resume_markdown(dossier: Mapping[str, Any]) -> str:
    """``resume.md``: what to decide first, then what happened, then the evidence.

    Times are clock times (UTC, with the date when not the start day); an empty
    section is left out, and what is not known is said once, at the end.
    ``resume.json`` keeps every field, every ISO time and every status.
    """
    identity = dossier["identity"]
    why = dossier["why_stopped"]
    day = start_day_of(identity["started_at"])

    def clock(at: Any) -> str:
        return clock_text(at, day) or "time unknown"

    lines = [f"# Resume dossier: run {identity['run_id']}", ""]
    if dossier["provisional"]:
        lines.extend(["> Provisional: the run has not ended; this dossier is rewritten.", ""])
    options = dossier["options"]
    lines.extend(
        [
            f"**Decision needed:** {options['decision']}.",
            "",
            "## How to restart",
            "",
            *_bullets(options["steps"]),
        ]
    )

    # What happened: one statement, then how it is known and its evidence.
    when = f" at {clock_text(why.get('at'), day)}" if clock_text(why.get("at"), day) else ""
    detail = f": {why['detail']}" if why.get("detail") else ""
    lines.extend(["", "## What happened", "", f"{why['headline']}{when}{detail}.", ""])
    facts = []
    if why.get("enforcement"):
        facts.append(f"Enforcement: {why['enforcement']}.")
    after = why.get("tool_calls_after_stop")
    if after:
        facts.append(
            f"After the stop decision, {after} more tool call(s) ran (already running when "
            "Pathlayer decided, or not blocked)."
        )
    if why.get("last_record_at"):
        noticed = clock_text(why.get("noticed_at"), day)
        facts.append(
            f"Last record from the run: {clock(why['last_record_at'])}"
            + (f"; the interruption was noticed at {noticed}." if noticed else ".")
        )
    if why.get("reason"):
        facts.append(f"Recorded reason: `{_md(why['reason'])}`")
    facts.append(f"How we know: {_HOW_WE_KNOW[why['status']]}.")
    lines.extend(_bullets(facts))

    actions = dossier["pathlayer_actions"]
    others = [a for a in actions["actions"] if a["reason"] != why.get("reason")]
    if others or actions["blocked_calls"]:
        lines.extend(["", "## What Pathlayer did", ""])
        for action in actions["actions"]:
            same = action["reason"] == why.get("reason")
            reason = "" if same or not action["reason"] else f": `{_md(action['reason'])}`"
            lines.append(
                f"- {clock(action['at'])}: {action['action']} ({action['delivery']}){reason}"
            )
        for call in actions["blocked_calls"]:
            lines.append(f"- {clock(call['at'])}: {call['tool']} blocked ({call['why']})")
        if actions["more_blocked_calls"]:
            lines.append(f"- ... and {actions['more_blocked_calls']} more blocked call(s)")

    ledger = dossier["proof_ledger"]
    columns = [c for c in (LEDGER_PROVEN, LEDGER_FAILED, LEDGER_UNPROVEN) if ledger[c]]
    if columns:
        lines.extend(
            [
                "",
                "## Proof ledger",
                "",
                "Observed facts, not a verdict on the task. Proven: the call's own result shows "
                "it was done (a command exited 0, a file was written or read). Failed: its "
                "result shows it was not. Unproven: its result says neither.",
            ]
        )
        for column in columns:
            lines.extend(["", f"**{column}**", *_bullets(_merged_entries(ledger[column]))])

    attempts = dossier["attempts"]
    if attempts["groups"]:
        digests = any(group["params_digest"] for group in attempts["groups"])
        head = "| Tool | Calls | Failed | Blocked | First | Last |"
        rule = "|---|---|---|---|---|---|"
        if digests:
            head = "| Tool | Same arguments (digest) | Calls | Failed | Blocked | First | Last |"
            rule = "|---|---|---|---|---|---|---|"
        lines.extend(["", "## What was tried", "", head, rule])
        for group in attempts["groups"]:
            digest = f" {group['params_digest'] or '-'} |" if digests else ""
            lines.append(
                f"| {_md(mask_text(group['tool'], 40))} |{digest} {group['calls']} | "
                f"{group['failed']} | {group['blocked']} | {clock(group['first_at'])} | "
                f"{clock(group['last_at'])} |"
            )
        if attempts["more_groups"]:
            lines.append(f"\n... and {attempts['more_groups']} more group(s) in resume.json")
        if digests:
            lines.extend(["", f"Note: {attempts['digest_scope']}."])
        else:
            lines.extend(["", "Grouped by tool only: this run has no gate log with digests."])

    clean = dossier["last_clean_call"]
    if clean["event_id"]:
        lines.extend(
            [
                "",
                "## Last call without an error",
                "",
                f"{clean['tool']} ({clean['kind'].replace('_', ' ')}) at {clock(clean['at'])}.",
            ]
        )

    cost = dossier["cost"]
    if cost["status"] != UNKNOWN:
        before = cost["before_stop"]
        after_cost = cost["after_stop"]
        lines.extend(["", "## Cost (estimated)", ""])
        label = "Before the stop decision" if after_cost is not None else "Whole run"
        lines.append(f"- {label}: {_usd(before)}")
        if after_cost is not None:
            lines.append(f"- After it: {_usd(after_cost)}")
            lines.append(f"- Split by {cost['split_by']}.")

    lines.extend(["", "## Not known", "", *_bullets(dossier["unknowns"]["items"])])

    started = parse_iso(identity["started_at"])
    lines.extend(
        [
            "",
            "## Run details",
            "",
            *_bullets(
                [
                    f"Status: {identity['display_status']}",
                    f"Session: {identity['session_key'] or 'unknown'}; "
                    f"agent: {identity['agent_id'] or 'unknown'}",
                    f"Model: {identity['model'] or 'unknown'}",
                    "Started: "
                    + (started.strftime("%Y-%m-%d %H:%M:%S UTC") if started else "unknown")
                    + f"; ended: {clock_text(identity['ended_at'], day) or 'not recorded'}",
                    f"Pathlayer: {identity['pathlayer_version'] or 'unknown version'}",
                ]
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _usd(share: Mapping[str, Any] | None) -> str:
    if share is None or share["usd"] is None:
        return "not measured"
    return f"${share['usd']:.4f} over {share['costed_events']} model call(s)"


def write_resume_dossier(
    run: RunDirectory, report: SessionReport | None = None
) -> dict[str, Any] | None:
    """Write ``resume.json`` and ``resume.md``, or remove them for a healthy run.

    A provisional dossier written at a STOP or a failed attempt must not
    outlive a run that then ended well.
    """
    dossier = build_resume_dossier(run, report)
    if dossier is None:
        run.resume_json_path.unlink(missing_ok=True)
        run.resume_md_path.unlink(missing_ok=True)
        return None
    atomic_write_json(run.resume_json_path, dossier)
    atomic_write_text(run.resume_md_path, render_resume_markdown(dossier))
    return dossier


__all__ = [
    "SCHEMA",
    "build_resume_dossier",
    "mask_text",
    "needs_dossier",
    "render_resume_markdown",
    "template_texts",
    "write_resume_dossier",
]
