"""Event pipeline (INF-3): wire capture -> detectors -> policy -> action.

The pipeline is the minimal in-memory backbone of Pathlayer. It receives
normalized ``ToolEvent``s (typically from an :class:`~pathlayer.adapters.Adapter`),
keeps the running history, runs every detector over that history, fuses the
resulting :class:`~pathlayer.detectors.base.Signal`s through a
:class:`~pathlayer.policies.Policy`, and returns the graded :class:`Action`.

It performs no I/O and no LLM calls; given the same event sequence it produces
the same action sequence. Arbitration is graded (ARB-3): the policy maps fused
confidence to a tier and the pipeline threads the :class:`HysteresisState` so a
brief spike does not flip the action. Only ``STOP`` halts the run; a compatible
actuator may enforce intermediate tiers (LangGraph implements WARN/SUMMARIZE),
while FALLBACK/ESCALATE remain unimplemented in V1.

:attr:`Pipeline.last_decision` is the one read-only window on the detection loop:
what the *last* processed event produced (each detector's raw confidence, the
fused score, the arbitrated action and its reason) as an immutable
:class:`PipelineDecision`. It is deliberately a single slot, not a timeline --
constant memory whatever the run length -- and building the durable chronology is
the consumer's job: INF-1 journals it, and OBS-2 projects it onto a
:class:`~pathlayer.postmortem.TickRecord` for the OpenClaw recorder's
``decisions.jsonl``. Both read the same arbitration rather than recomputing it.

PERF-2: each detector's incremental state (:meth:`Detector.create_state` /
:meth:`Detector.step`) lives here, one slot per detector instance, **never** on
the shared ``Detector`` object itself -- the same ``Detector`` instance can
(and, per the tests, must be able to) sit in two different ``Pipeline``s at
once without cross-talk. A detector that has not overridden ``step`` (i.e. is
still using the default batch-rebuild fallback from
:mod:`pathlayer.detectors.base`) is dispatched via the historical
``evaluate(history)`` path instead, sharing one ``tuple(self._history)`` build
across every such detector -- byte-for-byte the pre-PERF-2 behaviour, so
``LoopDetector``/``RetryDetector``/``BudgetDetector`` and any third-party
detector see zero change. Every new/updated state (detector states, the
progress-evidence tracker, hysteresis) is computed into a local variable first
and only assigned to ``self`` after the whole tick (detectors *and* the
policy) succeeds without raising -- mirroring the existing history
append/rollback below -- so a raising detector or policy leaves every prior
state untouched, ready for the same event to be retried.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pathlayer.detectors.base import Detector, RuntimeLimitProvider, Signal
from pathlayer.detectors.progress import ProgressAnchor
from pathlayer.events import EventKind, ToolEvent
from pathlayer.observability import (
    log_action,
    log_decision,
    log_signal,
    log_would_be_action,
)
from pathlayer.policies import Action, Policy
from pathlayer.policies.actuation import ActuationContext, EventSnapshot
from pathlayer.policies.base import HysteresisState
from pathlayer.postmortem import PostMortem

# Evidence key carrying the LOOP-2 precision positive-progress signal (see
# pathlayer.detectors.progress.is_advancing): 1.0 on a tick where the last
# event set a fresh record on an applicable progress anchor, else 0.0.
PROGRESS_EVIDENCE_KEY = "progress"

DEFAULT_RECENT_HISTORY = 10


@dataclass(slots=True, frozen=True)
class _EvidenceEntry:
    """One anchor's incremental state + last-seen level, for progress evidence."""

    anchor_state: Any
    previous_level: float | None


@dataclass(slots=True, frozen=True)
class _EvidenceState:
    """Immutable incremental state backing :meth:`Pipeline._evidence` (PERF-2).

    Equivalent-by-construction to ``is_advancing(progress_anchors, history)`` on
    the full history at every tick (see the module docstring on ``step``'s
    contract) -- ``is_advancing`` itself is untouched and remains the pure
    batch specification, used directly by anyone who wants the O(n) reference
    computation (e.g. tests). ``last_advancing`` lets a non-``TOOL_CALL`` event
    pass through unchanged: ``is_advancing`` filters to ``TOOL_CALL`` first, so
    the filtered sequence does not grow on any other event kind and the result
    cannot change either.
    """

    entries: tuple[_EvidenceEntry, ...]
    last_advancing: bool


def _create_evidence_state(anchors: Sequence[ProgressAnchor]) -> _EvidenceState:
    entries = tuple(
        _EvidenceEntry(anchor_state=anchor.create_state(), previous_level=None)
        for anchor in anchors
    )
    return _EvidenceState(entries=entries, last_advancing=False)


def _step_evidence(
    state: _EvidenceState, anchors: Sequence[ProgressAnchor], event: ToolEvent
) -> tuple[_EvidenceState, bool]:
    """Pure fold step behind :meth:`Pipeline._evidence`'s replacement (PERF-2).

    See ``_EvidenceState``.
    """
    if event.kind is not EventKind.TOOL_CALL:
        return state, state.last_advancing
    next_entries: list[_EvidenceEntry] = []
    advancing = False
    for anchor, entry in zip(anchors, state.entries, strict=True):
        new_anchor_state, snapshot = anchor.step(entry.anchor_state, event)
        current = snapshot.level
        previous = entry.previous_level
        if previous is not None and current is not None and current > previous:
            advancing = True
        next_entries.append(_EvidenceEntry(anchor_state=new_anchor_state, previous_level=current))
    return _EvidenceState(entries=tuple(next_entries), last_advancing=advancing), advancing


@dataclass(slots=True)
class WouldBeAction:
    """An arbitrated action dry-run observed but did not emit.

    Recorded only on a *transition to a non-NONE action* (the arbitrated action
    changed from the previous event and is not NONE), so a sustained STOP yields
    a single entry, not one per event, and a recovery back to NONE records
    nothing. ``index`` / ``ts`` of the first entry are the detection latency.
    """

    index: int
    ts: float
    action: Action
    signals: tuple[Signal, ...]
    reason: str


@dataclass(slots=True, frozen=True)
class PipelineDecision:
    """Canonical immutable record of one successfully evaluated pipeline tick.

    Reporting types such as OBS-2's ``TickRecord`` project this record instead
    of recomputing policy state. ``ts`` is event-sourced, and ``reason`` is empty
    when arbitration chose ``NONE``.
    """

    index: int
    ts: float
    signals: tuple[Signal, ...]
    evidence: Mapping[str, float]
    fused: float
    reason: str
    arbitrated_action: Action
    emitted_action: Action
    dry_run: bool


@dataclass(slots=True, frozen=True)
class _DecisionInputs:
    """Inputs retained until a consumer asks for the last decision."""

    index: int
    ts: float
    signals: tuple[Signal, ...]
    evidence: Mapping[str, float]
    arbitrated_action: Action
    emitted_action: Action
    dry_run: bool


class Pipeline:
    """Drives events through detectors and the arbitration policy.

    In enforcement mode (the default), once arbitration decides ``STOP`` the
    pipeline halts: it records a :class:`~pathlayer.postmortem.PostMortem` and
    ignores any further events (a clean flag-based stop, not a process kill).

    In dry-run (``dry_run=True``) the pipeline only observes: it never halts and
    emits no action (``process`` always returns ``NONE``); the actions it would
    have taken are exposed via :attr:`would_be_actions`. See :meth:`process`.
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        policy: Policy | None = None,
        *,
        recent_history: int = DEFAULT_RECENT_HISTORY,
        logger: logging.Logger | None = None,
        dry_run: bool = False,
        progress_anchors: Sequence[ProgressAnchor] | None = None,
    ) -> None:
        if recent_history < 0:
            raise ValueError("recent_history must be >= 0")
        self._detectors: list[Detector] = list(detectors)
        self._policy: Policy = policy if policy is not None else Policy()
        self._history: list[ToolEvent] = []
        self._actions: list[Action] = []
        self._recent_history = recent_history
        self._logger = logger
        self._dry_run = dry_run
        self._stopped = False
        self._post_mortem: PostMortem | None = None
        # ACT-1: the context of the last emitted non-NONE action, consumed by
        # wire() to drive an Actuator. None until the first non-NONE action, and
        # always None in dry-run (nothing is emitted there).
        self._last_context: ActuationContext | None = None
        self._last_decision_inputs: _DecisionInputs | None = None
        self._last_decision: PipelineDecision | None = None
        self._would_be_actions: list[WouldBeAction] = []
        # Last arbitrated action, to record would-be actions only on transition.
        self._last_would_be: Action = Action.NONE
        # Hysteresis machine state (ARB-3), threaded across events by the policy.
        self._hysteresis = HysteresisState()
        # LOOP-2 precision: anchors used to compute the positive-progress
        # evidence fed to the policy (never a Detector, never in `signals`).
        # Empty by default, so a caller that does not opt in sees no evidence
        # and therefore no suppression -- identical to pre-LOOP-2 behaviour.
        self._progress_anchors: list[ProgressAnchor] = list(progress_anchors or [])
        self._evidence_state: _EvidenceState | None = (
            _create_evidence_state(self._progress_anchors) if self._progress_anchors else None
        )
        # PERF-2: a detector that overrides `step` (together with
        # `create_state`) gets its own incremental state slot, threaded here --
        # never on the shared Detector instance (see the module docstring). A
        # detector still using the default fallback (`type(d).step is
        # Detector.step`) needs no slot: it is dispatched via the historical
        # `evaluate(history)` path below, unchanged.
        self._incremental: list[bool] = [type(d).step is not Detector.step for d in self._detectors]
        self._detector_states: list[Any] = [
            d.create_state() if inc else None
            for d, inc in zip(self._detectors, self._incremental, strict=True)
        ]
        # RES-2: validated once, here, so a misconfigured pipeline (two
        # RuntimeLimitProvider detectors) fails at construction -- not
        # silently later, the first time something reads max_duration_seconds
        # (e.g. mid-run, from OpenClawRunRecorder's pipeline_factory path).
        limits = [
            detector.max_duration_seconds
            for detector in self._detectors
            if isinstance(detector, RuntimeLimitProvider)
        ]
        if len(limits) > 1:
            raise ValueError("a pipeline may have only one runtime-limit provider")
        self._max_duration_seconds: int | None = min(limits) if limits else None

    @property
    def history(self) -> tuple[ToolEvent, ...]:
        """Events actually processed, in arrival order (excludes post-stop events)."""
        return tuple(self._history)

    @property
    def actions(self) -> tuple[Action, ...]:
        """Action *emitted* for each processed event, in order.

        In dry-run this is all ``NONE`` (nothing is emitted); the actions
        arbitration *would* have taken live in :attr:`would_be_actions`.
        """
        return tuple(self._actions)

    @property
    def dry_run(self) -> bool:
        """True if the pipeline only observes (emits no action, never halts)."""
        return self._dry_run

    @property
    def would_be_actions(self) -> tuple[WouldBeAction, ...]:
        """Actions arbitration would have emitted in dry-run.

        One entry per transition to a non-NONE action (see :class:`WouldBeAction`).
        Always empty outside dry-run.
        """
        return tuple(self._would_be_actions)

    @property
    def stopped(self) -> bool:
        """True once a STOP action has halted the run."""
        return self._stopped

    @property
    def max_duration_seconds(self) -> int | None:
        """Configured RES-2 hard limit, if a ResourceDetector is enabled.

        This exposes configuration only. Clock ownership remains entirely at
        the runtime sink boundary, preserving Pipeline's deterministic replay.
        Validated once at construction (see ``__init__``); reading this
        property never raises.
        """
        return self._max_duration_seconds

    @property
    def post_mortem(self) -> PostMortem | None:
        """The post-mortem captured at STOP, or None if the run never stopped."""
        return self._post_mortem

    @property
    def actuation_context(self) -> ActuationContext | None:
        """Context of the last emitted non-NONE action (ACT-1), else None.

        Consumed by :func:`~pathlayer.policies.wiring.wire` right after
        :meth:`process` to drive an :class:`~pathlayer.policies.actuators.Actuator`.
        Always None in dry-run (no action is emitted).
        """
        return self._last_context

    @property
    def last_decision(self) -> PipelineDecision | None:
        """Decision snapshot for the most recent successfully processed event."""
        inputs = self._last_decision_inputs
        if inputs is not None and self._last_decision is None:
            action = inputs.arbitrated_action
            self._last_decision = PipelineDecision(
                index=inputs.index,
                ts=inputs.ts,
                signals=inputs.signals,
                evidence=inputs.evidence,
                fused=self._policy.fuse(inputs.signals, inputs.evidence),
                reason=self._action_reason(action, inputs.signals, inputs.evidence),
                arbitrated_action=action,
                emitted_action=inputs.emitted_action,
                dry_run=inputs.dry_run,
            )
        return self._last_decision

    def process(self, event: ToolEvent, *, event_id: str | None = None) -> Action:
        """Process one event and return the arbitrated action.

        Detectors receive an immutable snapshot of the history so a buggy
        detector cannot mutate pipeline state. If a detector or the policy
        raises, the event is rolled back so ``history`` and ``actions`` stay in
        sync. After a STOP the pipeline is halted: further events are ignored
        and STOP is returned without touching the history.

        In dry-run the pipeline only observes: detectors and arbitration run and
        are logged, but no action is emitted (always returns ``NONE``) and the
        run never halts, so the full would-be trajectory can be inspected via
        :attr:`would_be_actions`.

        PERF-2: an incremental detector (see the module docstring) is advanced
        via ``step(state, event)`` -- no full-history rebuild. Any detector
        still on the batch fallback shares one ``tuple(self._history)`` build,
        computed only if at least one such detector remains -- identical to the
        pre-PERF-2 behaviour for those detectors. Every next-state (detector
        states, evidence tracker, hysteresis) is held in a local variable and
        committed to ``self`` only after the whole ``try`` block succeeds, so a
        raising detector/policy leaves every prior state untouched, exactly
        like the existing ``self._history.pop()`` rollback below.
        """
        # ``event_id`` belongs to persistence/integration boundaries. The pure
        # pipeline accepts it for API compatibility but deliberately ignores it.
        _ = event_id
        if self._stopped:
            return Action.STOP
        self._history.append(event)
        try:
            batch_history: tuple[ToolEvent, ...] | None = None
            if not all(self._incremental):
                batch_history = tuple(self._history)
            signals: list[Signal] = []
            next_detector_states = list(self._detector_states)
            for i, detector in enumerate(self._detectors):
                if self._incremental[i]:
                    new_state, signal = detector.step(self._detector_states[i], event)
                    next_detector_states[i] = new_state
                else:
                    assert batch_history is not None
                    signal = detector.evaluate(batch_history)
                signals.append(signal)
            next_evidence_state = self._evidence_state
            if self._progress_anchors:
                assert next_evidence_state is not None
                next_evidence_state, advancing = _step_evidence(
                    next_evidence_state, self._progress_anchors, event
                )
                evidence: Mapping[str, float] = {PROGRESS_EVIDENCE_KEY: 1.0 if advancing else 0.0}
            else:
                evidence = {}
            next_state, action = self._policy.step(self._hysteresis, signals, evidence)
        except Exception:
            self._history.pop()
            raise
        # Commit every next-state only after the tick succeeds, so a raising
        # detector/policy leaves history, detector states, the evidence
        # tracker and hysteresis all untouched.
        self._detector_states = next_detector_states
        self._evidence_state = next_evidence_state
        self._hysteresis = next_state
        if self._dry_run:
            emitted = self._observe(event, signals, action, evidence)
            self._last_decision_inputs = _DecisionInputs(
                index=len(self._history) - 1,
                ts=event.ts,
                signals=tuple(signals),
                evidence=MappingProxyType(dict(evidence)),
                arbitrated_action=action,
                emitted_action=emitted,
                dry_run=True,
            )
            self._last_decision = None
            return emitted
        self._actions.append(action)
        self._log(event, signals, action, evidence)
        if action is Action.STOP:
            self._stopped = True
            self._post_mortem = self._build_post_mortem(signals, evidence)
        if action is not Action.NONE:
            # Built after the STOP branch so post_mortem is available and shared
            # by identity (None for every non-STOP tier). Only for an emitted
            # non-NONE action -- the transition wire() acts on.
            self._last_context = self._build_context(event, signals, action, evidence)
        self._last_decision_inputs = _DecisionInputs(
            index=len(self._history) - 1,
            ts=event.ts,
            signals=tuple(signals),
            evidence=MappingProxyType(dict(evidence)),
            arbitrated_action=action,
            emitted_action=action,
            dry_run=False,
        )
        self._last_decision = None
        return action

    def _observe(
        self,
        event: ToolEvent,
        signals: list[Signal],
        action: Action,
        evidence: Mapping[str, float],
    ) -> Action:
        """Dry-run path: record/log the arbitrated action without emitting it.

        Nothing reaches the agent (returns ``NONE``, never halts). Signals and
        the decision are logged on every event; a :class:`WouldBeAction` is
        recorded (and logged) only on a transition -- when the arbitrated action
        changed since the last event -- so a sustained STOP records once.
        """
        index = len(self._history) - 1
        self._actions.append(Action.NONE)
        self._log_signals_and_decision(event, signals, action, evidence)
        if action is not self._last_would_be:
            self._last_would_be = action
            if action is not Action.NONE:
                self._would_be_actions.append(
                    WouldBeAction(
                        index=index,
                        ts=event.ts,
                        action=action,
                        signals=tuple(signals),
                        reason=self._action_reason(action, signals, evidence),
                    )
                )
                if self._logger is not None:
                    log_would_be_action(self._logger, action, index=index, ts=event.ts)
        return Action.NONE

    def run(self, events: Iterable[ToolEvent]) -> list[Action]:
        """Process events until the first STOP (inclusive) or exhaustion.

        In dry-run the pipeline never halts, so ``run`` always processes every
        event; the would-be STOP is recorded in :attr:`would_be_actions`.
        """
        actions: list[Action] = []
        for event in events:
            actions.append(self.process(event))
            if self._stopped:
                break
        return actions

    def restore(self, events: Iterable[ToolEvent]) -> list[Action]:
        """Replay durable events without emitting duplicate observability logs.

        Restoration is only valid on a fresh pipeline. It uses the ordinary
        deterministic processing path, so detector, evidence and hysteresis
        state are rebuilt from their public event-sourced specification.
        """
        if self._history:
            raise ValueError("restore requires a fresh pipeline")
        return [decision.emitted_action for decision in self.restore_decisions(events)]

    def restore_decisions(self, events: Iterable[ToolEvent]) -> list[PipelineDecision]:
        """Replay a fresh pipeline and return every reconstructed decision."""
        if self._history:
            raise ValueError("restore requires a fresh pipeline")
        logger = self._logger
        self._logger = None
        try:
            decisions: list[PipelineDecision] = []
            for event in events:
                self.process(event)
                decision = self.last_decision
                if decision is None:
                    break
                decisions.append(decision)
                if self.stopped:
                    break
            return decisions
        finally:
            self._logger = logger

    def sink(self, event: ToolEvent) -> None:
        """Adapter-compatible sink (see :data:`pathlayer.adapters.base.EventSink`)."""
        self.process(event)

    def close(self, status: str = "completed") -> None:
        """Close a logical run; a pure in-memory pipeline has nothing to flush."""
        if status not in {"completed", "interrupted", "stopped"}:
            raise ValueError(f"invalid terminal status {status!r}")

    def _log(
        self, event: ToolEvent, signals: list[Signal], action: Action, evidence: Mapping[str, float]
    ) -> None:
        if self._logger is None:
            return
        self._log_signals_and_decision(event, signals, action, evidence)
        if action is not Action.NONE:
            log_action(self._logger, action, index=len(self._history) - 1, ts=event.ts)

    def _log_signals_and_decision(
        self,
        event: ToolEvent,
        signals: list[Signal],
        action: Action,
        evidence: Mapping[str, float],
    ) -> None:
        """Log every signal and the arbitration decision (common to both modes).

        Every signal is logged at its **raw** confidence regardless of
        suppression (LOOP-2 precision): ``evidence`` only affects what
        ``fused``/``action`` mean, never what is written to the log, so an
        operator can always see a suppression happened instead of a silent
        ``loop=0.0`` (OBS-1: every signal, decision and action, honestly).
        """
        if self._logger is None:
            return
        index = len(self._history) - 1
        for signal in signals:
            log_signal(self._logger, signal, index=index, ts=event.ts)
        fused = self._policy.fuse(signals, evidence)
        log_decision(self._logger, action, signals, fused=fused, index=index, ts=event.ts)

    def _action_reason(
        self, action: Action, signals: Sequence[Signal], evidence: Mapping[str, float]
    ) -> str:
        """Human-readable reason for an action, from the driving signal.

        When a priority override (ARB-4) *determined* the committed action (its
        tier equals ``action``), name the signal that triggered it
        (:meth:`~pathlayer.policies.Policy.override_signal`) so the post-mortem
        matches the override -- e.g. "budget" -- even if that detector carries a
        fusion weight of 0. When the fused path escalated beyond the override (or
        no override fired), use
        :meth:`~pathlayer.policies.Policy.dominant_signal` (the actionable signal
        with the largest weighted contribution), so a STOP driven by convergence
        names the signal that drove it and never an advisory or currently
        suppressed one (LOOP-2 precision). Falls back to the action's own name
        when no actionable signal exists (a custom policy can act without
        detectors), so the reason matches the arbitrated action -- e.g. "warn"
        for a would-be WARN.
        """
        if action is Action.NONE:
            return ""
        driver: Signal | None = None
        if self._policy.override_action(signals, evidence) == action:
            driver = self._policy.override_signal(signals, evidence)
        if driver is None:
            driver = self._policy.dominant_signal(signals, evidence)
        if driver is not None:
            return f"{driver.name}: {driver.reason}" if driver.reason else driver.name
        return action.name.lower()

    def _build_context(
        self,
        event: ToolEvent,
        signals: list[Signal],
        action: Action,
        evidence: Mapping[str, float],
    ) -> ActuationContext:
        """Build the ACT-1 actuation context for an emitted non-NONE action.

        Reuses the tick's own ``signals``/``evidence`` so the fused score and
        reason match exactly what drove the decision. ``post_mortem`` is shared
        by identity with :attr:`post_mortem` (non-None only at STOP). The event
        snapshot excludes ``args``/``result``/``metadata`` (see
        :class:`~pathlayer.policies.actuation.EventSnapshot`).
        """
        return ActuationContext(
            action=action,
            fused=self._policy.fuse(signals, evidence),
            reason=self._action_reason(action, signals, evidence),
            signals=tuple(signals),
            event=EventSnapshot(
                tool=event.tool,
                kind=event.kind,
                ts=event.ts,
                index=len(self._history) - 1,
                cost_usd=event.cost_usd,
            ),
            post_mortem=self._post_mortem,
        )

    def _build_post_mortem(
        self, signals: list[Signal], evidence: Mapping[str, float]
    ) -> PostMortem:
        # recent_history == 0 must mean "no tail"; a plain [-0:] slice would
        # return the whole history because -0 == 0.
        recent = self._history[-self._recent_history :] if self._recent_history else []
        return PostMortem(
            action=Action.STOP,
            reason=self._action_reason(Action.STOP, signals, evidence),
            stopped_at_index=len(self._history) - 1,
            events_processed=len(self._history),
            signals=tuple(signals),
            recent_history=tuple(recent),
        )
