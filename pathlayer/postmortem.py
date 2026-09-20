"""Structured run records: the STOP post-mortem (ACT-5) and the session report (OBS-2).

Two records, two moments, one module:

- :class:`PostMortem` (ACT-5) is written *at* a STOP: the triggering action, a
  human-readable reason, where in the stream it fired, the detector signals at
  that step, and a tail of recent events. A run that never stops has none.
- :class:`SessionReport` (OBS-2) is written *after* a run ends, whether it
  stopped or not: identity and duration, the final outcome, the cost with its
  provenance, which detectors reacted and how strongly, the actions taken, and
  the stop reason if there was one. :class:`TickRecord` is the per-event unit it
  aggregates, projected from the pipeline's canonical
  :class:`~pathlayer.pipeline.PipelineDecision` (see
  :meth:`TickRecord.from_decision`) and persisted by an adapter.

Both are plain data: serializable to JSON, rebuildable from JSON, and free of
any I/O or clock read of their own. A report is therefore reconstructible from a
durable journal long after the process that produced it is gone -- which is what
makes an ``interrupted`` run reportable at all.

The honesty rule this module exists to enforce: a report never presents the
*absence of a measurement* as a *result*, and never presents a *result* as an
absence. An unmeasured cost is reported as unmeasured rather than as ``$0.00``,
while a figure that was computed is reported even when it came out at zero; a
journal carrying no detector confidence is reported as carrying none, not as "no
detector reacted", and a fused score that was never recorded is unknown rather
than zero. See :class:`CostBreakdown` and :meth:`SessionReport.format`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pathlayer.detectors.base import Signal
from pathlayer.events import ToolEvent
from pathlayer.policies import Action

if TYPE_CHECKING:
    # Import-time only: the pipeline imports this module, so the projection
    # keeps the dependency pointing one way at runtime.
    from pathlayer.pipeline import PipelineDecision


@dataclass(frozen=True, slots=True)
class CodingDiagnostic:
    """Small, user-facing projection for one coding-agent session.

    The raw event metadata remains the durable forensic artefact.  This view
    deliberately exposes only the facts needed to understand a decision:
    contract state, the latest observable progress, the resource's latest
    state, calls since progress, and the detector that actually intervened.
    It never prints a fingerprint or treats a changed file as task success.
    """

    contract_state: str
    last_progress: str | None
    last_progress_event: int | None
    state_resources: tuple[str, ...]
    state_outcome: str | None
    state_event: int | None
    calls_since_progress: int | None
    intervention: str | None
    intervention_event: int | None
    intervention_action: str | None = None
    intervention_reason: str | None = None
    would_have_action: str | None = None
    would_have_reason: str | None = None
    would_have_event: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_state": self.contract_state,
            "last_progress": self.last_progress,
            "last_progress_event": self.last_progress_event,
            "state_resources": list(self.state_resources),
            "state_outcome": self.state_outcome,
            "state_event": self.state_event,
            "calls_since_progress": self.calls_since_progress,
            "intervention": self.intervention,
            "intervention_event": self.intervention_event,
            "intervention_action": self.intervention_action,
            "intervention_reason": self.intervention_reason,
            "would_have_action": self.would_have_action,
            "would_have_reason": self.would_have_reason,
            "would_have_event": self.would_have_event,
        }

    def format(self) -> str:
        lines = [f"contract: {self.contract_state}"]
        if self.last_progress is None:
            lines.append("last contract progress: none observed")
        else:
            lines.append(
                f"last contract progress: {self.last_progress} (event {self.last_progress_event})"
            )
        if not self.state_resources:
            lines.append("state evidence: none recorded")
        else:
            lines.append(
                f"state evidence: {', '.join(self.state_resources)} is {self.state_outcome} "
                f"(event {self.state_event})"
            )
        if self.calls_since_progress is None:
            lines.append("tool calls since contract progress: not applicable")
        else:
            lines.append(f"tool calls since contract progress: {self.calls_since_progress}")
        if self.intervention is None:
            lines.append("intervention: none")
        else:
            lines.append(
                f"intervention: {self.intervention_action} — {self.intervention_reason} "
                f"(event {self.intervention_event})"
            )
        if self.would_have_action is not None:
            lines.append(
                f"would have intervened: {self.would_have_action} — {self.would_have_reason} "
                f"(event {self.would_have_event})"
            )
        return "\n".join(lines)


def build_coding_diagnostic(
    events: Sequence[ToolEvent], ticks: Sequence[TickRecord]
) -> CodingDiagnostic:
    """Project recorded coding evidence and decisions without any I/O.

    ``ticks`` may be shorter than ``events`` after an enforced STOP.  Their
    indexes are the pipeline's zero-based event indexes; duplicate indexes are
    rejected because their intervention attribution would be ambiguous.
    """
    # These modules depend on the pipeline, which imports ``postmortem`` for
    # the STOP record.  Projection happens after capture, so keeping the
    # imports here preserves the one-way startup dependency.
    from pathlayer.detectors.proof import ProofProgressAnchor
    from pathlayer.state_evidence import state_evidence_from_event

    by_index: dict[int, TickRecord] = {}
    for tick in ticks:
        if tick.index in by_index:
            raise ValueError("coding diagnostic requires unique tick indexes")
        by_index[tick.index] = tick

    anchor = ProofProgressAnchor()
    proof_state = anchor.create_state()
    contract_state = "no contract observed"
    last_progress: str | None = None
    last_progress_event: int | None = None
    state_resources: tuple[str, ...] = ()
    state_outcome: str | None = None
    state_event: int | None = None
    intervention: str | None = None
    intervention_event: int | None = None
    calls_since_progress: int | None = None
    intervention_action: str | None = None
    intervention_reason: str | None = None
    would_have_action: str | None = None
    would_have_reason: str | None = None
    would_have_event: int | None = None

    for index, event in enumerate(events):
        prior = proof_state.active
        proof_state, snapshot = proof_state.advance(event)
        active = proof_state.active
        # Use the detector's accepted epoch, sequence, and trust-boundary semantics.
        previous = (
            dict(prior.states) if prior and active and prior.spec_hash == active.spec_hash else {}
        )
        newly_true = (
            [
                checkpoint
                for checkpoint, value in active.states
                if value and not previous.get(checkpoint, False)
            ]
            if active
            else []
        )
        if active is not None and (prior is None or prior.spec_hash != active.spec_hash):
            # A new contract epoch makes prior progress and call counts irrelevant.
            last_progress = None
            last_progress_event = None
            calls_since_progress = 0
        if active is not None:
            if newly_true:
                last_progress = newly_true[0]
                last_progress_event = index
                calls_since_progress = 0
            else:
                assert calls_since_progress is not None
                calls_since_progress += 1
        if active is None:
            contract_state = "unobserved" if proof_state.known_specs else "no contract observed"
        elif snapshot.level is None:
            contract_state = "unobserved"
        else:
            contract_state = "reached" if snapshot.satisfied else "incomplete"

        envelope = state_evidence_from_event(event)
        if envelope is not None:
            pairs = list(zip(envelope.records[::2], envelope.records[1::2], strict=True))
            matching = [
                (before, after)
                for before, after in pairs
                if (
                    any(r.status == envelope.outcome.value for r in (before, after))
                    if envelope.outcome.value in ("unavailable", "collection_failed")
                    else (before.canonical_fingerprint != after.canonical_fingerprint)
                    == (envelope.outcome.value == "changed")
                )
            ]
            # state_evidence_from_event has already verified that every envelope
            # outcome has at least one matching before/after pair.
            state_resources = tuple(after.resource_id for _, after in matching)
            state_outcome = envelope.outcome.value
            state_event = index

        decision_tick = by_index.get(index)
        if decision_tick is not None and decision_tick.action is not Action.NONE:
            action = decision_tick.action.name.lower()
            reason = decision_tick.reason or action
            if decision_tick.dry_run:
                if would_have_action is None:
                    would_have_action = action
                    would_have_reason = reason
                    would_have_event = index
            elif intervention is None:
                intervention = (
                    decision_tick.reason.split(":", 1)[0] if decision_tick.reason else action
                )
                intervention_event = index
                intervention_action = action
                intervention_reason = reason

    return CodingDiagnostic(
        contract_state=contract_state,
        last_progress=last_progress,
        last_progress_event=last_progress_event,
        state_resources=state_resources,
        state_outcome=state_outcome,
        state_event=state_event,
        calls_since_progress=calls_since_progress,
        intervention=intervention,
        intervention_event=intervention_event,
        intervention_action=intervention_action,
        intervention_reason=intervention_reason,
        would_have_action=would_have_action,
        would_have_reason=would_have_reason,
        would_have_event=would_have_event,
    )


@dataclass(slots=True)
class PostMortem:
    action: Action
    reason: str
    stopped_at_index: int
    events_processed: int
    signals: tuple[Signal, ...]
    recent_history: tuple[ToolEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.name,
            "reason": self.reason,
            "stopped_at_index": self.stopped_at_index,
            "events_processed": self.events_processed,
            "signals": [
                {"name": s.name, "confidence": s.confidence, "reason": s.reason}
                for s in self.signals
            ],
            "recent_history": [
                {
                    "tool": e.tool,
                    "args": e.args,
                    "kind": e.kind.value,
                    "cost_usd": e.cost_usd,
                    "ts": e.ts,
                    # The single most useful metadata field for a forensic read of
                    # a budget/retry stop; surfaced explicitly rather than dumping
                    # the whole (potentially large/sensitive) metadata blob.
                    "error": e.metadata.get("error"),
                }
                for e in self.recent_history
            ],
        }

    def write(self, path: str | Path) -> None:
        """Write the post-mortem as indented JSON to ``path``."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# OBS-2: the per-tick record a session report aggregates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SignalObservation:
    """One detector's reading on one tick, flattened for durable storage.

    A :class:`~pathlayer.detectors.base.Signal` minus ``diagnostics``: the three
    fields a report needs (who, how strongly, why). ``confidence`` is stored
    **raw**, exactly as the detector emitted it -- normalization into ``[0, 1]``
    belongs to the policy's use sites, and a report that silently clamped would
    hide a misbehaving custom detector instead of showing it.
    """

    name: str
    confidence: float
    reason: str = ""

    @classmethod
    def from_signal(cls, signal: Signal) -> SignalObservation:
        return cls(name=signal.name, confidence=signal.confidence, reason=signal.reason)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SignalObservation:
        if not isinstance(data, Mapping):
            raise ValueError("a signal observation must be an object")
        if "name" not in data or "confidence" not in data:
            raise ValueError("a signal observation requires name and confidence")
        confidence = data["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("a signal observation confidence must be numeric")
        return cls(
            name=str(data["name"]),
            confidence=float(confidence),
            reason=str(data.get("reason") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "confidence": self.confidence, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class TickRecord:
    """Everything one processed event produced, as a single immutable record.

    This is the OBS-2 persistence contract: an adapter writes :meth:`to_dict`
    (or its columns) to a durable journal, and a report is rebuilt from those
    rows alone. It is the shape the OpenClaw recorder persists today. A store
    that keeps its own decision row (INF-1) can project it onto this shape; the
    projection is the store's, and it needs the event's ``ts``, which a per-tick
    row does not have to carry.

    - ``index`` / ``ts``: the event's position in the run and its event-sourced
      timestamp (never a wall clock).
    - ``signals``: every detector that ran on this tick, at its raw confidence.
      Silent detectors are kept, not filtered: "wired and silent" and "not
      wired at all" are different facts and a report must be able to tell them
      apart.
    - ``fused``: the policy's weighted noisy-OR over the *actionable* signals --
      the number actually compared to the thresholds. A confidence peak that
      never reached a tier is only visible because this is recorded on every
      tick, not just on the ones that acted. ``None`` means *not recorded*, not
      ``0.0``: a journal line written before OBS-2 has no fused column, and
      reading it back as a zero would understate the run's real peak.
    - ``action``: the action arbitration *decided* for this tick. In dry-run
      that action was never emitted; ``dry_run`` says so, so a reader can never
      mistake an observation for an intervention.
    - ``reason``: why that action was taken, from the driving signal. Empty when
      ``action`` is ``NONE`` -- there is no reason for a non-decision, and
      inventing one would name a detector that drove nothing.
    """

    index: int
    ts: float
    signals: tuple[SignalObservation, ...] = ()
    fused: float | None = None
    action: Action = Action.NONE
    reason: str = ""
    dry_run: bool = False

    @property
    def reacting_signals(self) -> tuple[SignalObservation, ...]:
        """Signals that actually reported something (confidence > 0)."""
        return tuple(signal for signal in self.signals if signal.confidence > 0.0)

    @classmethod
    def from_decision(cls, decision: PipelineDecision) -> TickRecord:
        """Project the pipeline's canonical decision onto this reporting shape.

        The pipeline already arbitrated the tick, so nothing here is recomputed:
        a second fused score or a second reason could disagree with the one the
        run actually acted on. ``arbitrated_action`` becomes ``action`` -- in
        dry-run that is still what arbitration chose, and ``dry_run`` is what
        records that it was never emitted.
        """
        return cls(
            index=decision.index,
            ts=decision.ts,
            signals=tuple(SignalObservation.from_signal(s) for s in decision.signals),
            fused=decision.fused,
            action=decision.arbitrated_action,
            reason=decision.reason,
            dry_run=decision.dry_run,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TickRecord:
        """Rebuild a tick from :meth:`to_dict` output.

        ``index`` and ``ts`` are required -- a tick with no position in the run
        is not a tick. Every other field falls back to its default, so a row
        written by an older schema still loads; an absent ``fused`` stays
        ``None`` (unrecorded) rather than becoming a measured ``0.0``.
        """
        raw_signals = data.get("signals") or ()
        raw_fused = data.get("fused")
        return cls(
            index=int(data["index"]),
            ts=float(data["ts"]),
            signals=tuple(SignalObservation.from_dict(entry) for entry in raw_signals),
            fused=None if raw_fused is None else float(raw_fused),
            action=Action[str(data.get("action", Action.NONE.name))],
            reason=str(data.get("reason") or ""),
            dry_run=bool(data.get("dry_run", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "ts": self.ts,
            "signals": [signal.to_dict() for signal in self.signals],
            "fused": self.fused,
            "action": self.action.name,
            "reason": self.reason,
            "dry_run": self.dry_run,
        }


# --------------------------------------------------------------------------- #
# OBS-2: the session report
# --------------------------------------------------------------------------- #

# Manifest execution status of a run that reached its documented end.
COMPLETED_STATUS = "completed"


class SessionOutcome(str, Enum):
    """How a run ended, from Pathlayer's point of view.

    ``interrupted`` is not in the OBS-2 issue text but cannot be dropped:
    without it, a run whose listener died mid-flight would have to be filed as
    ``completed``, which is exactly the kind of claim this report refuses to
    make.
    """

    COMPLETED = "completed"
    STOPPED = "stopped"
    ESCALATED = "escalated"
    INTERRUPTED = "interrupted"


_OUTCOME_MEANING: dict[SessionOutcome, str] = {
    SessionOutcome.COMPLETED: "the run reached its end and Pathlayer never halted it",
    SessionOutcome.STOPPED: (
        "Pathlayer selected STOP and stopped analysis; runtime termination depends on the "
        "integration's actuator"
    ),
    SessionOutcome.ESCALATED: "arbitration reached the ESCALATE tier; no actuator applies it in V1",
    SessionOutcome.INTERRUPTED: "the run never reached its end, so its record is partial",
}


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """What a run cost, split by where each figure came from.

    ``None`` means *nothing computed this component*; a number means something
    did, zero included. That distinction is the whole point, and it cuts both
    ways: "the adapter wired no cost source" must not be reported as "this run
    was free", and a component that was computed and came out at zero must not
    be reported as unmeasured either.

    - ``captured_usd`` / ``priced_events``: cost the adapter actually attached to
      events (``ToolEvent.cost_usd``), and how many events carried one.
      ``captured_usd`` is ``None`` -- never ``0.0`` -- when ``priced_events`` is
      ``0``, because a zero sum over zero priced events measures nothing.
    - ``estimated_usd``: adapter estimates or the BUD-3 substitute -- a configured per-tool price
      times the number of eligible calls. ``None`` when no estimate was ever
      computed; ``0.0`` when the budget detector ran and no configured price
      applied to the recorded calls, which is a result rather than a gap.
    - ``baseline_usd``: the configured cross-run seed
      (``detectors.budget.baseline_usd``) that this run's detection actually
      started from, ``0.0`` being the ordinary single-run value.
    - ``carried_over_usd``: the cumulative spend of a *scope* shared by several
      runs, as a state store reports it through its own read interface. Always
      ``None`` here: this report reads the run directory, which is the adapter's
      own capture and knows nothing about other runs. It stays out of
      :attr:`total_usd` even once a store fills it in, because a store that
      restores a run by replaying its journal recomputes those same dollars into
      ``captured_usd`` / ``estimated_usd``. It is context for a reader, never an
      input to detection and never a seed for ``baseline_usd``.
    - ``source``: one honest sentence about provenance, written by whoever built
      the report (it is the adapter that knows what it can and cannot price).
    """

    captured_usd: float | None = None
    priced_events: int | None = None
    estimated_usd: float | None = None
    baseline_usd: float | None = None
    carried_over_usd: float | None = None
    source: str = ""

    @property
    def measured(self) -> bool:
        """True when at least one component was actually computed.

        The test is "was this computed?", not "is it non-zero". A budget detector
        that ran over a run whose tools carry no configured price legitimately
        totals ``$0.00``: that is a measured zero, and reporting it as an absence
        of measurement would be the mirror image of the lie this class exists to
        prevent -- both put a claim in the report that the evidence does not
        support.

        The captured component is keyed on ``priced_events`` rather than on
        ``captured_usd``, because the count *is* the number of measurements
        taken: a sum of zero over zero priced events measures nothing.
        """
        return (
            (self.priced_events is not None and self.priced_events > 0)
            or self.estimated_usd is not None
            or self.baseline_usd is not None
        )

    @property
    def total_usd(self) -> float | None:
        """Effective spend, or ``None`` when nothing measured it.

        ``None`` rather than ``0.0`` when unmeasured, on purpose: a zero total
        would read as a result ("this run was free") when the truth is an absence
        of measurement. A ``0.0`` returned here is the opposite -- a total that
        was computed and came out at zero.
        """
        if not self.measured:
            return None
        components = (self.captured_usd, self.estimated_usd, self.baseline_usd)
        return sum((value for value in components if value is not None), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured": self.measured,
            "total_usd": self.total_usd,
            "captured_usd": self.captured_usd,
            "priced_events": self.priced_events,
            "estimated_usd": self.estimated_usd,
            "baseline_usd": self.baseline_usd,
            "carried_over_usd": self.carried_over_usd,
            "source": self.source,
        }

    def describe(self) -> str:
        """One human-readable line, never claiming an unmeasured cost."""
        if not self.measured:
            detail = self.source or "no cost source is wired for this run"
            return f"not measured -- {detail}"
        # Each computed component is shown, including a computed zero; an absent
        # one is left out entirely rather than printed as $0.0000.
        parts: list[str] = []
        if self.captured_usd is not None:
            # Keyed on the figure, not on the count: a component left out of the
            # sum printed on the same line would not add up to the total.
            over = f" over {self.priced_events} priced event(s)" if self.priced_events else ""
            parts.append(f"${self.captured_usd:.4f} captured{over}")
        if self.estimated_usd is not None:
            parts.append(f"${self.estimated_usd:.4f} estimated")
        if self.baseline_usd is not None:
            parts.append(f"${self.baseline_usd:.4f} configured baseline")
        total = self.total_usd
        line = f"${total:.4f} effective" if total is not None else "unknown"
        if parts:
            line += " = " + " + ".join(parts)
        if self.source:
            line += f" [{self.source}]"
        return line


@dataclass(frozen=True, slots=True)
class DetectorActivity:
    """What one detector did across a whole run.

    ``peak_confidence`` and its position are the actionable part: a detector
    that climbed to 0.80 and never reached the stop threshold is the single most
    useful thing this report can tell a user, and it is invisible in a journal
    that only records the ticks that acted.

    ``invalid_ticks`` counts the readings that were not a usable confidence
    (``NaN``, infinite, or outside ``[0, 1]``). They are kept out of
    ``peak_confidence`` and out of ``active_ticks``: ``NaN`` loses every
    comparison, so a first unusable reading would both hide the run's real peak
    and let a misbehaving detector be reported as silent.
    """

    name: str
    observed_ticks: int
    active_ticks: int
    peak_confidence: float
    peak_index: int | None = None
    peak_ts: float | None = None
    peak_reason: str = ""
    invalid_ticks: int = 0

    @property
    def reacted(self) -> bool:
        """True if the detector ever reported a non-zero confidence."""
        return self.active_ticks > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observed_ticks": self.observed_ticks,
            "active_ticks": self.active_ticks,
            "reacted": self.reacted,
            "peak_confidence": self.peak_confidence,
            "peak_index": self.peak_index,
            "peak_ts": self.peak_ts,
            "peak_reason": self.peak_reason,
            "invalid_ticks": self.invalid_ticks,
        }


@dataclass(frozen=True, slots=True)
class ActionTaken:
    """One arbitration action, at the tick it was decided.

    Recorded on *transitions* only (the arbitrated action changed and is not
    ``NONE``), like :class:`~pathlayer.pipeline.WouldBeAction`, so a sustained
    STOP is one entry rather than one per remaining event. ``emitted`` is
    ``False`` for a dry-run decision: it was arbitrated, never applied.
    """

    action: Action
    index: int
    ts: float
    reason: str = ""
    emitted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.name,
            "index": self.index,
            "ts": self.ts,
            "reason": self.reason,
            "emitted": self.emitted,
        }


def _confidence(value: float) -> str:
    """Render a confidence at two decimals, never rounding a real reading to zero.

    A detector sitting at 0.004 is *not* silent, and printing ``0.00`` next to
    "non-zero on 5 of 5 ticks" would contradict itself. Small non-zero readings
    render as ``<0.01`` instead. An out-of-range or ``NaN`` value is never
    clamped into something plausible: it is printed as it is, and a *detector's*
    own unusable reading is reported as unusable rather than as a confidence
    (see :attr:`DetectorActivity.invalid_ticks`) -- a report exists to surface a
    misbehaving detector, not to tidy it away.
    """
    if 0.0 < value < 0.01:
        return "<0.01"
    return f"{value:.2f}"


def _is_usable(confidence: float) -> bool:
    """True for a reading a report can compare: finite and within ``[0, 1]``."""
    return math.isfinite(confidence) and 0.0 <= confidence <= 1.0


def _detector_activity(ticks: Sequence[TickRecord]) -> tuple[DetectorActivity, ...]:
    """Fold per-tick signals into one :class:`DetectorActivity` per detector.

    Ordered by descending peak confidence then by name, so the detector a user
    should look at first is printed first and the order stays deterministic.

    An unusable reading (``NaN``, infinite, out of range) is counted apart and
    kept out of both the peak and the active count: ``NaN`` compares false
    against everything, so folding it in would pin the peak on it and report the
    detector as silent.
    """
    order: list[str] = []
    observed: dict[str, int] = {}
    active: dict[str, int] = {}
    invalid: dict[str, int] = {}
    peak: dict[str, tuple[float, int | None, float | None, str]] = {}
    for tick in ticks:
        for signal in tick.signals:
            if signal.name not in observed:
                order.append(signal.name)
                observed[signal.name] = 0
                active[signal.name] = 0
                invalid[signal.name] = 0
                peak[signal.name] = (0.0, None, None, "")
            observed[signal.name] += 1
            if not _is_usable(signal.confidence):
                invalid[signal.name] += 1
                continue
            if signal.confidence > 0.0:
                active[signal.name] += 1
            if signal.confidence > peak[signal.name][0]:
                peak[signal.name] = (signal.confidence, tick.index, tick.ts, signal.reason)
    activities = [
        DetectorActivity(
            name=name,
            observed_ticks=observed[name],
            active_ticks=active[name],
            peak_confidence=peak[name][0],
            peak_index=peak[name][1],
            peak_ts=peak[name][2],
            peak_reason=peak[name][3],
            invalid_ticks=invalid[name],
        )
        for name in order
    ]
    activities.sort(key=lambda activity: (-activity.peak_confidence, activity.name))
    return tuple(activities)


def _actions_taken(ticks: Sequence[TickRecord]) -> tuple[ActionTaken, ...]:
    """Collect one entry per transition to a non-NONE arbitrated action."""
    taken: list[ActionTaken] = []
    previous = Action.NONE
    for tick in ticks:
        if tick.action is previous:
            continue
        previous = tick.action
        if tick.action is Action.NONE:
            continue
        taken.append(
            ActionTaken(
                action=tick.action,
                index=tick.index,
                ts=tick.ts,
                reason=tick.reason,
                emitted=not tick.dry_run,
            )
        )
    return tuple(taken)


def _peak_fused(ticks: Sequence[TickRecord]) -> TickRecord | None:
    """The tick with the highest *recorded* fused score, earliest first on a tie.

    Ticks whose ``fused`` is ``None`` are skipped, not read as zeros: they come
    from journal lines written before the fused column existed, and letting them
    win a comparison would report an unknown score as a measured peak of 0.00.
    ``None`` means no tick recorded a fused score at all.
    """
    best: TickRecord | None = None
    best_value: float | None = None
    for tick in ticks:
        value = tick.fused
        if value is None:
            continue
        if best_value is None or value > best_value:
            best, best_value = tick, value
    return best


@dataclass(slots=True)
class SessionReport:
    """End-of-session summary of one run: cost, detectors, actions, outcome.

    Built by :func:`build_session_report` from a sequence of
    :class:`TickRecord`\\ s plus the run's own metadata, so it depends on no
    live pipeline and can be rebuilt from a durable journal at any time.
    """

    run_id: str
    outcome: SessionOutcome
    status: str = ""
    session_key: str | None = None
    agent_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    events_recorded: int = 0
    events_analyzed: int = 0
    ticks_with_signals: int = 0
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    detectors: tuple[DetectorActivity, ...] = ()
    peak_fused: float | None = None
    peak_fused_index: int | None = None
    peak_fused_ts: float | None = None
    actions: tuple[ActionTaken, ...] = ()
    stop_reason: str | None = None
    dry_run: bool = False
    notes: tuple[str, ...] = ()

    @property
    def emitted_actions(self) -> tuple[ActionTaken, ...]:
        return tuple(action for action in self.actions if action.emitted)

    @property
    def severest_emitted(self) -> Action | None:
        emitted = self.emitted_actions
        return max((action.action for action in emitted), default=None) if emitted else None

    @property
    def peak_never_acted(self) -> bool:
        """A fused confidence was recorded, yet nothing was ever emitted.

        The headline diagnostic of this report: it tells a user how close the
        run came to an intervention that never happened.
        """
        return self.peak_fused is not None and not self.emitted_actions

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "outcome": self.outcome.value,
            "outcome_meaning": _OUTCOME_MEANING[self.outcome],
            "status": self.status,
            "session_key": self.session_key,
            "agent_id": self.agent_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "events_recorded": self.events_recorded,
            "events_analyzed": self.events_analyzed,
            "ticks_with_signals": self.ticks_with_signals,
            "cost": self.cost.to_dict(),
            "detectors": [activity.to_dict() for activity in self.detectors],
            "peak_fused": self.peak_fused,
            "peak_fused_index": self.peak_fused_index,
            "peak_fused_ts": self.peak_fused_ts,
            "peak_never_acted": self.peak_never_acted,
            "actions": [action.to_dict() for action in self.actions],
            "stop_reason": self.stop_reason,
            "dry_run": self.dry_run,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """One deterministic line per run, for a multi-run listing."""
        when = self.started_at or "unknown start"
        total = self.cost.total_usd
        cost = f"${total:.4f}" if total is not None else "cost n/a"
        peak = f"peak {_confidence(self.peak_fused)}" if self.peak_fused is not None else "peak n/a"
        return (
            f"{when}  {self.run_id}  {self.outcome.value}  "
            f"{self.events_recorded} event(s)  {peak}  {cost}"
        )

    def format(self) -> str:
        """Human-readable report: what happened, and what was never measured."""
        lines = [
            f"run {self.run_id}{self._identity_suffix()}",
            f"outcome: {self.outcome.value} -- {_OUTCOME_MEANING[self.outcome]}",
            f"status: {self.status or 'unknown'}   {self._timing()}",
            f"events: {self.events_recorded} recorded, {self.events_analyzed} analyzed",
            f"cost: {self.cost.describe()}",
        ]
        lines.extend(self._detector_lines())
        lines.append(self._fused_line())
        lines.extend(self._action_lines())
        if self.stop_reason:
            lines.append(f"stop reason: {self.stop_reason}")
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)

    def _identity_suffix(self) -> str:
        parts = []
        if self.session_key:
            parts.append(f"session {self.session_key}")
        if self.agent_id:
            parts.append(f"agent {self.agent_id}")
        return f" ({', '.join(parts)})" if parts else ""

    def _timing(self) -> str:
        started = self.started_at or "unknown start"
        ended = self.ended_at or "no recorded end"
        if self.duration_seconds is None:
            return f"{started} -> {ended}, duration unknown"
        return f"{started} -> {ended}, {self.duration_seconds:.1f}s"

    def _detector_lines(self) -> list[str]:
        if not self.detectors:
            if self.events_analyzed == 0:
                return ["detectors: no tool call was analyzed on this run, so no detector ever ran"]
            return [
                f"detectors: none recorded -- the journal carries no detector "
                f"confidence for the {self.events_analyzed} analyzed event(s), "
                f"so what they saw is unknown, not silent"
            ]
        reacted = sum(1 for activity in self.detectors if activity.reacted)
        lines = [
            f"detectors: {len(self.detectors)} wired, {reacted} reacted "
            f"over {self.ticks_with_signals} recorded tick(s)"
        ]
        width = max(len(activity.name) for activity in self.detectors)
        for activity in self.detectors:
            lines.append(f"  {activity.name.ljust(width)}  {self._activity_detail(activity)}")
        return lines

    @staticmethod
    def _activity_detail(activity: DetectorActivity) -> str:
        if activity.invalid_ticks:
            return (
                f"{activity.invalid_ticks} unusable reading(s) (NaN or out of range) "
                f"over {activity.observed_ticks} tick(s): this detector is misbehaving"
            )
        if not activity.reacted:
            return f"silent on all {activity.observed_ticks} tick(s)"
        detail = (
            f"peak {_confidence(activity.peak_confidence)} on event {activity.peak_index} "
            f"(t={activity.peak_ts}), non-zero on {activity.active_ticks} "
            f"of {activity.observed_ticks} tick(s)"
        )
        if activity.peak_reason:
            detail += f" -- {activity.peak_reason}"
        return detail

    def _fused_line(self) -> str:
        if self.peak_fused is None:
            return "fused score: none recorded for this run"
        if self.peak_fused <= 0.0:
            if self.emitted_actions:
                return (
                    "fused score: never left 0.00, yet an action was emitted via "
                    "override (fusion weight 0)"
                )
            return "fused score: never left 0.00 -- no actionable signal ever fired"
        where = f"on event {self.peak_fused_index} (t={self.peak_fused_ts})"
        peak = _confidence(self.peak_fused)
        if self.peak_never_acted:
            return (
                f"fused score: peaked at {peak} {where} and never reached an "
                "action tier, so nothing was emitted"
            )
        return f"fused score: peaked at {peak} {where}"

    def _action_lines(self) -> list[str]:
        if not self.actions:
            return ["actions: none arbitrated"]
        lines = [f"actions: {len(self.actions)} arbitrated"]
        for action in self.actions:
            suffix = "" if action.emitted else " (dry-run: never applied)"
            reason = f" -- {action.reason}" if action.reason else ""
            lines.append(
                f"  event {action.index} (t={action.ts}): {action.action.name}{suffix}{reason}"
            )
        return lines


def _resolve_outcome(
    *, status: str, stopped: bool, severest_emitted: Action | None
) -> SessionOutcome:
    """Pick the single most informative of the four outcomes.

    Priority ``stopped > escalated > interrupted > completed``. A run can be
    both stopped and interrupted; ``stopped`` is the stronger statement about
    what Pathlayer did, and ``status`` is reported alongside so the partial
    record is never hidden. Any status other than ``completed`` -- including a
    run still in flight -- maps to ``interrupted`` rather than to ``completed``:
    claiming a run finished cleanly because we never saw it finish is exactly
    the lie this report exists to avoid.
    """
    if stopped:
        return SessionOutcome.STOPPED
    if severest_emitted is not None and severest_emitted >= Action.ESCALATE:
        return SessionOutcome.ESCALATED
    if status != COMPLETED_STATUS:
        return SessionOutcome.INTERRUPTED
    return SessionOutcome.COMPLETED


def build_session_report(
    *,
    run_id: str,
    status: str,
    ticks: Sequence[TickRecord],
    events_recorded: int,
    cost: CostBreakdown | None = None,
    stopped: bool = False,
    session_key: str | None = None,
    agent_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_seconds: float | None = None,
    notes: Iterable[str] = (),
) -> SessionReport:
    """Aggregate a run's ticks and metadata into a :class:`SessionReport`.

    Pure: no I/O, no clock. ``ticks`` are the events that were actually
    analyzed, in order; events recorded past a passive STOP have no tick (no
    detector ran for them) and are counted only in ``events_recorded``.
    ``stopped`` is what the run's own record says, so a STOP is still reported
    for a run whose journal was truncated before the STOP tick was written.
    """
    ordered = sorted(ticks, key=lambda tick: tick.index)
    actions = _actions_taken(ordered)
    emitted = [action for action in actions if action.emitted]
    severest = max((action.action for action in emitted), default=None) if emitted else None
    stopped_now = stopped or (severest is Action.STOP)
    peak = _peak_fused(ordered)
    # Only an *emitted* STOP has a stop reason; a dry-run STOP was arbitrated
    # and discarded, and it is already reported as a would-be action.
    stop_tick = next(
        (tick for tick in reversed(ordered) if tick.action is Action.STOP and not tick.dry_run),
        None,
    )
    return SessionReport(
        run_id=run_id,
        outcome=_resolve_outcome(status=status, stopped=stopped_now, severest_emitted=severest),
        status=status,
        session_key=session_key,
        agent_id=agent_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        events_recorded=events_recorded,
        events_analyzed=len(ordered),
        ticks_with_signals=sum(1 for tick in ordered if tick.signals),
        cost=cost if cost is not None else CostBreakdown(),
        detectors=_detector_activity(ordered),
        peak_fused=peak.fused if peak is not None else None,
        peak_fused_index=peak.index if peak is not None else None,
        peak_fused_ts=peak.ts if peak is not None else None,
        actions=actions,
        stop_reason=stop_tick.reason if stop_tick is not None and stop_tick.reason else None,
        dry_run=any(tick.dry_run for tick in ordered),
        notes=tuple(notes),
    )
