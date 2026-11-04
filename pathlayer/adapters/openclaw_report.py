"""Post-session reports for recorded OpenClaw runs, plus ``pathlayer-report``.

After a run ends the user needs one view of what happened: what it cost, which
detectors reacted and how strongly, what Pathlayer did, and how the run ended.
This module builds that view -- a :class:`~pathlayer.postmortem.SessionReport` --
and the CLI that reads it back.

Two properties shape the design:

- **A report is rebuilt from the run directory, never from memory.** Everything
  comes from ``manifest.json``, ``decisions.jsonl``, ``trace.json`` and
  ``final.json``. That is what makes an ``interrupted`` run reportable at all:
  its process is gone, so there is no in-memory state left to ask. The run
  directory is the adapter's own capture and predates OBS-2; this module reads
  it, and adds no state of its own. The engine's cross-run state belongs to
  INF-1's store and is read through its own interface; the run directory cannot
  see it, so ``CostBreakdown.carried_over_usd`` stays empty here and would not
  enter the effective total anyway.
- **A report never fills a gap with a zero, and never hides a computed zero.**
  A journal line written before OBS-2 has no fused score. An OpenClaw LLM event
  can carry a total-token estimate but no provider input/output split; its
  estimated status remains explicit. Missing token/model evidence is reported as
  *unmeasured*, never as "nothing happened" or "$0.00". Conversely a computed
  free-model value is reported even when it is zero. The report is aggregation
  only -- no detector, no threshold, no inference beyond what the files say.

The report data model itself lives in :mod:`pathlayer.postmortem` next to the
ACT-5 post-mortem, as OBS-2 asks. Only the filesystem reconstruction and the CLI
live here, in the adapter layer, because reading an OpenClaw run directory is
adapter knowledge -- exactly like :mod:`pathlayer.adapters.openclaw_export`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pathlayer.adapters.openclaw import translate_openclaw_llm_event
from pathlayer.adapters.openclaw_lifecycle import (
    BEFORE_AGENT_RUN,
    LLM_OUTPUT,
    extract_openclaw_run_id,
)
from pathlayer.adapters.openclaw_run_store import (
    RunDirectory,
    RunStore,
    atomic_write_json,
    runs_root_from_env,
)
from pathlayer.detectors.budget import BudgetDetector, BudgetState, budget_delta
from pathlayer.events import ToolEvent
from pathlayer.policies import Action
from pathlayer.policies.actions import action_from_export, action_name
from pathlayer.postmortem import (
    COMPLETED_STATUS,
    TRIGGER_ABSENT,
    UNATTRIBUTED_MISSING_RUN_ID,
    CostAttribution,
    CostBreakdown,
    CostShare,
    SessionReport,
    SignalObservation,
    TickRecord,
    UnattributedCost,
    build_session_report,
)
from pathlayer.pricing import ModelPrice
from pathlayer.validation.traces import load_trace

# Manifest keys carrying the two cost components a report cannot recompute from
# the run directory, because they depend on the config the run was recorded with
# (see :func:`cost_manifest_fields`). Three different situations leave them
# absent, and the report must not conflate them: a completed run recorded without
# the budget detector (no estimate was ever computed), a run cut before its first
# tool call, and a run journaled by a listener that predates these keys. Only the
# first supports a conclusion; see the ``estimated`` branches in
# :func:`_build_cost`.
ESTIMATED_COST_KEY = "cost_estimated_usd"
BASELINE_COST_KEY = "cost_baseline_usd"
COST_BUDGET_STATUS_KEY = "cost_budget_status"
COST_BUDGET_ENABLED = "enabled"
COST_BUDGET_DISABLED = "disabled"
COST_BUDGET_UNKNOWN = "unknown"

# Columns the recorder appends to every *analyzed* line of ``decisions.jsonl``,
# carrying one :class:`~pathlayer.postmortem.TickRecord`. The event's position and
# timestamp already travel as ``event_index`` / ``event_ts`` and are not
# duplicated; ``action`` stays the *emitted* action, so the arbitrated one gets
# its own column (they differ in dry-run).
SIGNALS_COLUMN = "signals"
FUSED_COLUMN = "fused"
ARBITRATED_COLUMN = "arbitrated_action"
DRY_RUN_COLUMN = "dry_run"
REASON_COLUMN = "reason"

# ``fused`` and ``arbitrated_action`` are also the schema discriminator: OBS-2
# introduced both, so a line carrying neither was written by an earlier listener.
# ``signals`` and ``reason`` are *not* usable for that test -- the previous
# recorder already wrote them, copied from the post-mortem, on the STOP line.
_OBS2_COLUMNS = (FUSED_COLUMN, ARBITRATED_COLUMN)


def tick_columns(tick: TickRecord) -> dict[str, Any]:
    """The durable per-tick columns of one ``decisions.jsonl`` line."""
    return {
        SIGNALS_COLUMN: [signal.to_dict() for signal in tick.signals],
        FUSED_COLUMN: tick.fused,
        ARBITRATED_COLUMN: action_name(tick.action),
        DRY_RUN_COLUMN: tick.dry_run,
        REASON_COLUMN: tick.reason,
    }


def cost_manifest_fields(detector: BudgetDetector, state: BudgetState) -> dict[str, Any]:
    """The cost components a later report could not recompute on its own.

    The captured cost is already in ``trace.json`` (every event carries its own
    ``cost_usd``), so only the total estimate and the configured cross-run
    baseline are persisted here: the BUD-3 fallback and baseline depend on the run
    config. A marker distinguishes this combined estimate from older manifests
    carrying only BUD-3 fallback costs, so trace estimates are never added twice.

    ``state`` is the accumulator folded over *every* recorded event, including
    those that arrived after a passive STOP and were never analyzed. A session
    report is about the session, and those calls really did happen.

    The recorder writes these fields on every tool call, not only at
    ``agent_end``: a run killed mid-flight must still carry the estimate for what
    it had spent, otherwise its report would have to guess whether the budget
    detector was even enabled.
    """
    return {
        ESTIMATED_COST_KEY: state.estimated_usd,
        "cost_estimate_includes_event_estimates": True,
        BASELINE_COST_KEY: detector.baseline_usd,
        COST_BUDGET_STATUS_KEY: COST_BUDGET_ENABLED,
    }


def _read_journal(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Parse one append-only JSONL journal, tolerating a torn final line.

    Returns the usable records and whether a truncated last line was dropped.

    The journal is appended one flushed-and-fsynced line at a time, newline
    included, so a complete record always ends with one. A last line without a
    terminating newline is therefore the signature of a process killed
    mid-write -- exactly the interrupted run this module exists to make
    reportable. It is dropped, and the caller says so in the report rather than
    refusing to report the run at all. An unparsable line anywhere else is
    corruption rather than a torn write, and is still refused loudly.
    """
    if not path.is_file():
        return [], False
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    torn_candidate = len(lines) if text and not text.endswith("\n") else None
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == torn_candidate:
                return records, True
            raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
        if not isinstance(record, dict):
            # Not softened for the torn line: every record starts with ``{``, so a
            # truncated prefix of one cannot parse into something else.
            raise ValueError(f"invalid record in {path} at line {line_number}")
        records.append(record)
    return records, False


def _is_legacy_decision(record: dict[str, Any]) -> bool:
    """True for an analyzed line written before OBS-2 added the tick columns."""
    return not any(column in record for column in _OBS2_COLUMNS)


def _tick_from_decision(record: dict[str, Any]) -> TickRecord | None:
    """Rebuild one tick from a decision line, or ``None`` if the line has none.

    ``None`` means the line was never analyzed (``analyzed: false``, recorded
    past a passive STOP): no detector ran for that event, so there is no tick to
    rebuild and inventing one would assert a reading nobody took.

    Every analyzed line yields a tick, including one written before OBS-2, and
    what such a line does not carry is reported as unknown rather than as zero:

    - the action comes from the pre-OBS-2 ``action`` column when
      ``arbitrated_action`` is absent. That column has always held the action
      Pathlayer *emitted*, so a legacy STOP is still reported as a STOP with its
      real reason instead of being flattened to ``NONE``.
    - the fused score stays ``None`` when the column is absent. A missing score
      is not a score of zero, and defaulting it to ``0.0`` used to hide the run's
      real peak behind a fabricated one.
    - the signals stay empty when the column is absent, which the report renders
      as "what the detectors saw is unknown", never as "the detectors were
      silent". A legacy journal carries them on the STOP line only, copied from
      the post-mortem.

    ``event_index`` and ``event_ts`` are required, on every journal format this
    module reads: a tick placed at index 0 and t=0.0 because the line carried
    neither would be a fabricated position, and the report would print it as a
    measurement.
    """
    if not record.get("analyzed", True):
        return None
    raw_signals = record.get(SIGNALS_COLUMN) or []
    if not isinstance(raw_signals, list):
        raise ValueError("signals in a decision record must be a list")
    raw_action = record.get(ARBITRATED_COLUMN)
    if raw_action is None:
        raw_action = record.get("action", Action.NONE.name)
    try:
        action = action_from_export(raw_action, field="arbitrated action")
    except ValueError as exc:
        raise ValueError(f"unknown arbitrated action {raw_action!r} in a decision record") from exc
    index = _optional_float(record.get("event_index"))
    ts = _optional_float(record.get("event_ts"))
    if index is None or ts is None:
        raise ValueError("an analyzed decision record must carry event_index and event_ts")
    try:
        signals = tuple(SignalObservation.from_dict(entry) for entry in raw_signals)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid signals in a decision record") from exc
    return TickRecord(
        index=int(index),
        ts=ts,
        signals=signals,
        fused=_optional_float(record.get(FUSED_COLUMN)),
        action=action,
        reason=str(record.get(REASON_COLUMN) or ""),
        dry_run=bool(record.get(DRY_RUN_COLUMN, False)),
    )


def _duration_seconds(
    final: dict[str, Any], started_at: Any, ended_at: Any, *, completed: bool
) -> float | None:
    """Run duration, from OpenClaw's own measurement or the recorded timestamps.

    The recorded timestamps are a fallback only for a run that reached its end:
    the ``ended_at`` of an interrupted run is stamped by the startup sweep that
    noticed the interruption, which can be hours later, so deriving a duration
    from it would report the listener's downtime as the run's own length.
    ``None`` when nothing usable measured it -- an unknown duration is reported
    as unknown rather than as ``0.0s``.
    """
    duration_ms = final.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
        return float(duration_ms) / 1000.0
    if not completed:
        return None
    start = _parse_iso(started_at)
    end = _parse_iso(ended_at)
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _build_cost(
    manifest: dict[str, Any],
    events: Sequence[ToolEvent],
    trace_available: bool,
    *,
    completed: bool,
    legacy_journal: bool,
    events_from_journal: bool = False,
) -> CostBreakdown:
    """Assemble the cost section, naming what nothing measured.

    The captured component is recomputed from the trace; the BUD-3 estimate and
    the configured baseline come from the manifest (see
    :func:`cost_manifest_fields`). ``carried_over_usd`` stays ``None``: the
    cumulative total of a multi-run scope lives in a state store this function
    does not read, and a store that restores by replay recounts those dollars
    itself.

    An absent estimate supports the conclusion "no budget detector was enabled"
    in exactly one case: a run that reached ``agent_end`` while recorded by a
    listener that would have persisted the estimate. A run cut before its end, or
    one journaled by a listener that never wrote those keys at all
    (``legacy_journal``), proves nothing either way, and the gap is reported as
    indeterminate rather than blamed on a detector that may well have been on.

    ``events_from_journal`` marks events read back from ``events.raw.jsonl``
    because the run never wrote a ``trace.json``: its model calls were
    captured all the same, and leaving them out would report a spend that
    happened as unmeasured.
    """
    estimated = _optional_float(manifest.get(ESTIMATED_COST_KEY))
    baseline = _optional_float(manifest.get(BASELINE_COST_KEY))
    budget_status = manifest.get(COST_BUDGET_STATUS_KEY)
    captured: float | None = None
    priced: int | None = None
    adapter_estimate: float | None = None
    if trace_available:
        deltas = [budget_delta(event) for event in events]
        priced = sum(delta.priced_events for delta in deltas)
        if priced:
            captured = sum(delta.captured_usd for delta in deltas)
        if any(
            event.metadata.get("cost_measurement") == "estimated_total_tokens"
            and _optional_float(event.cost_usd) is not None
            and 0.0 <= event.cost_usd < float("inf")
            for event in events
        ):
            adapter_estimate = sum(delta.estimated_usd for delta in deltas)
    manifest_estimated = estimated
    if adapter_estimate is not None:
        if estimated is None:
            estimated = adapter_estimate
        elif not manifest.get("cost_estimate_includes_event_estimates", False):
            # Older manifests contain only BUD-3 per-call fallback estimates.
            estimated += adapter_estimate
    reasons: list[str] = []
    if events_from_journal:
        reasons.append(
            "trace.json is absent: the LLM cost was read back from the raw hook journal "
            "at the prices pinned when the run was recorded"
        )
    if not trace_available:
        reasons.append("no trace.json, so no event cost could be read back")
    elif not priced:
        reasons.append(
            "no recorded event carried a positive measured price; LLM spend can be "
            "estimated when total-token usage and model pricing are available"
        )
    if manifest_estimated is None and budget_status == COST_BUDGET_DISABLED:
        reasons.append(
            "no configured cost estimate: the budget detector was not enabled for this run, "
            "so no per-tool estimate was ever computed"
        )
    elif manifest_estimated is None and budget_status == COST_BUDGET_UNKNOWN:
        reasons.append(
            "the configured cost estimate is indeterminate: this run used a custom pipeline "
            "factory, so the recorder cannot establish whether a budget detector was enabled"
        )
    elif manifest_estimated is None and legacy_journal:
        reasons.append(
            "the configured cost estimate is indeterminate: this run was recorded before Pathlayer "
            "persisted cost components, so whether the budget detector was enabled cannot "
            "be established from this directory"
        )
    elif manifest_estimated is None and not completed:
        reasons.append(
            "the configured cost estimate is indeterminate: this run never reached its end, and no "
            "estimate was journaled before the cut, so whether the budget detector was "
            "enabled cannot be established from this directory"
        )
    elif manifest_estimated is None:
        reasons.append(
            "the configured cost estimate is indeterminate: this run directory does not record "
            "whether the budget detector was enabled"
        )
    elif manifest_estimated == 0.0:
        reasons.append(
            "the budget estimate is a computed $0.00: no positive estimated cost "
            "applied to the recorded calls"
        )
    if adapter_estimate is not None:
        reasons.append(
            f"${adapter_estimate:.6f} estimated from total tokens and an input/output ratio"
        )
    return CostBreakdown(
        captured_usd=captured,
        priced_events=priced,
        estimated_usd=estimated,
        baseline_usd=baseline,
        carried_over_usd=None,
        source="; ".join(reasons),
    )


# --------------------------------------------------------------------------- #
# Cost attribution
# --------------------------------------------------------------------------- #


def _record_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _pinned_rate(value: Any) -> float | None:
    """A pinned rate or ratio component, or ``None`` unless it is a finite number >= 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rate = float(value)
    return rate if math.isfinite(rate) and rate >= 0.0 else None


def _pinned_pricing(manifest: dict[str, Any]) -> tuple[dict[str, ModelPrice], float] | None:
    """The price table this run was recorded with, or ``None`` if none is usable.

    Never the current configuration: repricing an old run with today's table
    would report a figure nobody computed when the call was made. A snapshot
    with any rate that is not a finite number ``>= 0`` (negative, ``NaN``,
    infinite, a string) is unusable as a whole: pricing with it would print a
    negative or ``NaN`` cost, so its calls stay counted but unmeasured.
    """
    snapshot = manifest.get("llm_pricing_snapshot")
    if not isinstance(snapshot, dict):
        return None
    ratio = _pinned_rate(snapshot.get("input_ratio"))
    models = snapshot.get("models")
    if ratio is None or ratio > 1.0 or not isinstance(models, dict):
        return None
    prices: dict[str, ModelPrice] = {}
    for name, raw in models.items():
        if not isinstance(raw, dict):
            return None
        input_rate = _pinned_rate(raw.get("input_usd_per_mtok"))
        output_rate = _pinned_rate(raw.get("output_usd_per_mtok"))
        if input_rate is None or output_rate is None:
            return None
        prices[str(name)] = ModelPrice(input_rate, output_rate)
    return prices, ratio


def _llm_events(
    payload: dict[str, Any], pricing: tuple[dict[str, ModelPrice], float] | None
) -> list[ToolEvent]:
    """The cost event(s) of one ``llm_output`` payload, exactly as recording made them.

    Empty when nothing can be computed -- no pinned price table, an invalid one,
    or no usable token counter -- which the callers report as unmeasured.
    """
    if pricing is None:
        return []
    prices, ratio = pricing
    try:
        return translate_openclaw_llm_event(payload, prices=prices, input_ratio=ratio)
    except ValueError:
        return []


def _share_of(events: Sequence[ToolEvent]) -> CostShare:
    """Cost one payload contributed, via the same fold as the budget detector."""
    if not events:
        return CostShare(unmeasured_events=1)
    share = CostShare()
    for event in events:
        delta = budget_delta(event)
        if delta.priced_events or event.metadata.get("cost_measurement") == (
            "estimated_total_tokens"
        ):
            share = share.plus(
                CostShare(
                    measured_usd=delta.captured_usd,
                    estimated_usd=delta.estimated_usd,
                    costed_events=1,
                )
            )
        else:
            share = share.plus(CostShare(unmeasured_events=1))
    return share


def _source_trigger(payload: dict[str, Any]) -> str:
    """``ctx.trigger`` verbatim (``cron``, ``heartbeat``, ...), never inferred."""
    ctx = payload.get("ctx")
    trigger = ctx.get("trigger") if isinstance(ctx, dict) else None
    if isinstance(trigger, str) and trigger.strip():
        return trigger
    return TRIGGER_ABSENT


def build_cost_attribution(run: RunDirectory) -> CostAttribution:
    """Attribute every cost hook this run directory recorded.

    Read from the hook journals -- ``events.raw.jsonl`` and the ``late.jsonl``
    annex -- because only the raw payload says whether the *source* named the
    run, and what it gave as a trigger. ``llm_output`` is the only OpenClaw hook
    that carries a cost; tool calls carry none and BUD-3 per-tool estimates are
    configuration, not captured cost, so neither appears here. Each payload is
    re-translated at the prices pinned in the manifest, the very translation the
    recorder applied live, so the figures match ``trace.json`` for a run that
    has one.
    """
    manifest = run.read_manifest()
    pricing = _pinned_pricing(manifest)
    raw, _ = _read_journal(run.raw_events_path)
    late, _ = _read_journal(run.late_events_path)
    turn_started = any(_record_payload(record).get("hook") == BEFORE_AGENT_RUN for record in raw)
    attributed = late_share = without_start = CostShare()
    unattributed: dict[str, CostShare] = {}
    entries = [(record, False) for record in raw] + [(record, True) for record in late]
    for record, is_late in entries:
        payload = _record_payload(record)
        if payload.get("hook") != LLM_OUTPUT:
            continue
        share = _share_of(_llm_events(payload, pricing))
        if extract_openclaw_run_id(payload) is None:
            trigger = _source_trigger(payload)
            unattributed[trigger] = unattributed.get(trigger, CostShare()).plus(share)
            continue
        attributed = attributed.plus(share)
        if is_late:
            late_share = late_share.plus(share)
        if not turn_started:
            without_start = without_start.plus(share)
    return CostAttribution(
        attributed=attributed,
        late=late_share,
        without_turn_start=without_start,
        unattributed=tuple(
            UnattributedCost(reason=UNATTRIBUTED_MISSING_RUN_ID, trigger=trigger, share=share)
            for trigger, share in sorted(unattributed.items())
        ),
        runs=1,
    )


def build_store_cost_attribution(runs: Sequence[RunDirectory]) -> CostAttribution:
    """One attribution over several runs: unattributed cost lives in its own directories."""
    total = CostAttribution()
    for run in runs:
        total = total.merged(build_cost_attribution(run))
    return total


def _journal_llm_events(run: RunDirectory, manifest: dict[str, Any]) -> list[ToolEvent]:
    """The LLM cost events of a run that never wrote ``trace.json``, from its raw journal."""
    pricing = _pinned_pricing(manifest)
    records, _ = _read_journal(run.raw_events_path)
    return [
        event
        for record in records
        if _record_payload(record).get("hook") == LLM_OUTPUT
        for event in _llm_events(_record_payload(record), pricing)
    ]


def build_run_report(run: RunDirectory) -> SessionReport:
    """Rebuild the session report of one recorded run from its directory alone."""
    manifest = run.read_manifest()
    decisions, truncated_journal = _read_journal(run.decisions_path)
    ticks = [tick for tick in (_tick_from_decision(record) for record in decisions) if tick]
    final = _read_json_object(run.final_path)

    trace_available = run.trace_path.is_file()
    events: list[ToolEvent] = list(load_trace(run.trace_path).events) if trace_available else []
    events_recorded = len(events) if trace_available else len(decisions)

    status = str(manifest.get("status") or "")
    completed = status == COMPLETED_STATUS
    ended_at = manifest.get("ended_at")
    duration = _duration_seconds(final, manifest.get("started_at"), ended_at, completed=completed)
    notes: list[str] = []
    if truncated_journal:
        notes.append(
            "the last journal line was truncated and was dropped: this run's record "
            "stops one event short"
        )
    if duration is None and not completed:
        # The recorded end is only named when there is one: a run still in flight
        # has none, and claiming otherwise would invent the very stamp this note
        # exists to distrust.
        stamped = ", and its recorded end is when the interruption was noticed"
        notes.append(
            f"duration unknown: this run never reached its end{stamped if ended_at else ''}"
        )
    analyzed = [record for record in decisions if record.get("analyzed", True)]
    legacy = [record for record in analyzed if _is_legacy_decision(record)]
    # OBS-5 reads the hook journals as well. They only add to the report, so a
    # corrupt one costs the attribution, never the rest of the report.
    attribution: CostAttribution | None
    try:
        cost_events = events if trace_available else _journal_llm_events(run, manifest)
        attribution = build_cost_attribution(run)
    except ValueError:
        cost_events = events
        attribution = None
        notes.append(
            "a hook journal (events.raw.jsonl or late.jsonl) is corrupt: the cost "
            "attribution is unavailable for this run"
        )
    from_journal = not trace_available and bool(cost_events)
    cost = _build_cost(
        manifest,
        cost_events,
        trace_available or from_journal,
        completed=completed,
        legacy_journal=bool(legacy),
        events_from_journal=from_journal,
    )
    late = attribution.late if attribution is not None else CostShare()
    if late.costed_events or late.unmeasured_events:
        notes.append(
            f"{late.costed_events + late.unmeasured_events} model call(s) reached Pathlayer after "
            "this run had ended: they are attributed to it below, but were never analyzed and "
            "are not part of the cost figure above"
        )
    if legacy:
        notes.append(
            f"{len(legacy)} of {len(analyzed)} analyzed line(s) were journaled before "
            "end-of-run reports existed: "
            "they carry the action Pathlayer emitted, but no fused score and no detector "
            "confidence except on a STOP, so what the detectors saw there is unknown, not silent"
        )
    if cost.estimated_usd is not None and events_recorded > len(ticks):
        notes.append(
            f"the cost figures cover all {events_recorded} recorded event(s) while the "
            f"detectors only analyzed {len(ticks)}: a passive STOP ends analysis, not the "
            "run, so the per-tick budget signal stops short of the run's own total"
        )
    if not trace_available:
        notes.append("trace.json is absent: this run never reached agent_end")

    return build_session_report(
        run_id=str(manifest.get("run_id") or run.original_run_id),
        status=status,
        ticks=ticks,
        events_recorded=events_recorded,
        events=events if trace_available else None,
        cost=cost,
        cost_attribution=attribution,
        stopped=manifest.get("stopped") is True,
        session_key=manifest.get("session_key"),
        agent_id=manifest.get("agent_id"),
        started_at=manifest.get("started_at"),
        ended_at=ended_at,
        duration_seconds=duration,
        notes=notes,
    )


def write_run_report(run: RunDirectory) -> SessionReport:
    """Build the report and persist it as ``report.json`` (same atomic writer)."""
    report = build_run_report(run)
    atomic_write_json(run.report_path, report.to_dict())
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _sort_key(run: RunDirectory) -> tuple[str, str]:
    """Chronological, deterministic ordering: recorded start, then run id."""
    try:
        manifest = run.read_manifest()
    except (OSError, ValueError):
        return ("", run.original_run_id)
    started = manifest.get("started_at")
    return (started if isinstance(started, str) else "", run.original_run_id)


def _select_runs(
    store: RunStore, *, run_id: str | None, last: bool, today: bool
) -> list[RunDirectory]:
    """Resolve one selection mode to an ordered list of run directories.

    ``today`` is the one mode that reads the wall clock: it keeps the runs whose
    recorded start falls on the current UTC date, so the same command returns a
    different set tomorrow, and a run started just before midnight UTC is out.
    That is CLI selection, not detection -- nothing downstream of here reads a
    clock.
    """
    runs = sorted(store.list_runs(), key=_sort_key)
    if run_id is not None:
        matches = [run for run in runs if run.original_run_id == run_id]
        if not matches:
            raise LookupError(f"no recorded run found for run_id={run_id!r}")
        if len(matches) > 1:
            raise LookupError(
                f"ambiguous run_id={run_id!r}: {len(matches)} directories match; refine the id"
            )
        return matches
    if last:
        if not runs:
            raise LookupError("no recorded run found")
        return [runs[-1]]
    if today:
        stamp = datetime.now(UTC).date().isoformat()
        return [run for run in runs if _sort_key(run)[0].startswith(stamp)]
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pathlayer-report",
        description="Show the post-session report of one or more recorded OpenClaw runs.",
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Original OpenClaw runId (ctx.runId) of a single run",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--last", action="store_true", help="the most recently started run")
    selection.add_argument("--today", action="store_true", help="every run started today (UTC)")
    selection.add_argument("--all", action="store_true", help="every recorded run")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Recorded runs root (default: $PATHLAYER_RUNS_DIR or /var/lib/pathlayer/runs)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the structured report instead of the human-readable summary",
    )
    parser.add_argument(
        "--cost-attribution",
        action="store_true",
        help=(
            "show captured LLM cost over the selected runs split into attributed to a run "
            "and unattributed, by reason"
        ),
    )
    args = parser.parse_args(argv)

    modes = [args.run_id is not None, args.last, args.today, args.all]
    if sum(1 for mode in modes if mode) != 1:
        parser.error("choose exactly one of: run_id, --last, --today, --all")

    store = RunStore(args.runs_root or runs_root_from_env())
    try:
        runs = _select_runs(store, run_id=args.run_id, last=args.last, today=args.today)
        if args.cost_attribution:
            attribution = build_store_cost_attribution(runs)
        else:
            reports = [build_run_report(run) for run in runs]
    except (LookupError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.cost_attribution:
        if args.as_json:
            print(json.dumps(attribution.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"cost attribution over {attribution.runs} run(s), from llm_output hooks")
            print("\n".join(attribution.describe_lines()))
        return 0

    if args.as_json:
        payload: Any = reports[0].to_dict() if len(reports) == 1 else [r.to_dict() for r in reports]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if not reports:
        # An empty-but-valid query is not an error: say so and exit clean.
        print("no run matched the selection")
        return 0
    # One full report when a single run was asked for; one line per run when a
    # whole set was, so a long history stays readable.
    if len(reports) == 1:
        print(reports[0].format())
    else:
        for report in reports:
            print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
