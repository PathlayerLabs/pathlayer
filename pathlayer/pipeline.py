"""Event pipeline (INF-3): wire capture -> detectors -> policy -> action.

The pipeline is the minimal in-memory backbone of Pathlayer. It receives
normalized ``ToolEvent``s (typically from an :class:`~pathlayer.adapters.Adapter`),
keeps the running history, runs every detector over that history, fuses the
resulting :class:`~pathlayer.detectors.base.Signal`s through a
:class:`~pathlayer.policies.Policy`, and returns the graded :class:`Action`.

It performs no I/O and no LLM calls; given the same event sequence it produces
the same action sequence. Arbitration is graded (ARB-3): the policy maps fused
confidence to a tier and the pipeline threads the :class:`HysteresisState` so a
brief spike does not flip the action. Only ``STOP`` halts the run; intermediate
tiers (WARN..ESCALATE) are decided and logged but, in V1, not yet enforced.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import ToolEvent
from pathlayer.observability import (
    log_action,
    log_decision,
    log_signal,
    log_would_be_action,
)
from pathlayer.policies import Action, Policy
from pathlayer.policies.base import HysteresisState
from pathlayer.postmortem import PostMortem

DEFAULT_RECENT_HISTORY = 10


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
        self._would_be_actions: list[WouldBeAction] = []
        # Last arbitrated action, to record would-be actions only on transition.
        self._last_would_be: Action = Action.NONE
        # Hysteresis machine state (ARB-3), threaded across events by the policy.
        self._hysteresis = HysteresisState()

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
    def post_mortem(self) -> PostMortem | None:
        """The post-mortem captured at STOP, or None if the run never stopped."""
        return self._post_mortem

    def process(self, event: ToolEvent) -> Action:
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
        """
        if self._stopped:
            return Action.STOP
        self._history.append(event)
        try:
            history = tuple(self._history)
            signals = [detector.evaluate(history) for detector in self._detectors]
            next_state, action = self._policy.step(self._hysteresis, signals)
        except Exception:
            self._history.pop()
            raise
        # Commit hysteresis state only after the tick succeeds, so a raising
        # detector/policy leaves both history and hysteresis untouched.
        self._hysteresis = next_state
        if self._dry_run:
            return self._observe(event, signals, action)
        self._actions.append(action)
        self._log(event, signals, action)
        if action is Action.STOP:
            self._stopped = True
            self._post_mortem = self._build_post_mortem(signals)
        return action

    def _observe(self, event: ToolEvent, signals: list[Signal], action: Action) -> Action:
        """Dry-run path: record/log the arbitrated action without emitting it.

        Nothing reaches the agent (returns ``NONE``, never halts). Signals and
        the decision are logged on every event; a :class:`WouldBeAction` is
        recorded (and logged) only on a transition -- when the arbitrated action
        changed since the last event -- so a sustained STOP records once.
        """
        index = len(self._history) - 1
        self._actions.append(Action.NONE)
        self._log_signals_and_decision(event, signals, action)
        if action is not self._last_would_be:
            self._last_would_be = action
            if action is not Action.NONE:
                self._would_be_actions.append(
                    WouldBeAction(
                        index=index,
                        ts=event.ts,
                        action=action,
                        signals=tuple(signals),
                        reason=self._action_reason(action, signals),
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

    def sink(self, event: ToolEvent) -> None:
        """Adapter-compatible sink (see :data:`pathlayer.adapters.base.EventSink`)."""
        self.process(event)

    def _log(self, event: ToolEvent, signals: list[Signal], action: Action) -> None:
        if self._logger is None:
            return
        self._log_signals_and_decision(event, signals, action)
        if action is not Action.NONE:
            log_action(self._logger, action, index=len(self._history) - 1, ts=event.ts)

    def _log_signals_and_decision(
        self, event: ToolEvent, signals: list[Signal], action: Action
    ) -> None:
        """Log every signal and the arbitration decision (common to both modes)."""
        if self._logger is None:
            return
        index = len(self._history) - 1
        for signal in signals:
            log_signal(self._logger, signal, index=index, ts=event.ts)
        fused = self._policy.fuse(signals)
        log_decision(self._logger, action, signals, fused=fused, index=index, ts=event.ts)

    def _action_reason(self, action: Action, signals: Sequence[Signal]) -> str:
        """Human-readable reason for an action, from the driving signal.

        When a priority override (ARB-4) *determined* the committed action (its
        tier equals ``action``), name the signal that triggered it
        (:meth:`~pathlayer.policies.Policy.override_signal`) so the post-mortem
        matches the override -- e.g. "budget" -- even if that detector carries a
        fusion weight of 0. When the fused path escalated beyond the override (or
        no override fired), use
        :meth:`~pathlayer.policies.Policy.dominant_signal` (the actionable signal
        with the largest weighted contribution), so a STOP driven by convergence
        names the signal that drove it and never an advisory one. Falls back to
        the action's own name when no actionable signal exists (a custom policy
        can act without detectors), so the reason matches the arbitrated action
        -- e.g. "warn" for a would-be WARN.
        """
        driver: Signal | None = None
        if self._policy.override_action(signals) == action:
            driver = self._policy.override_signal(signals)
        if driver is None:
            driver = self._policy.dominant_signal(signals)
        if driver is not None:
            return f"{driver.name}: {driver.reason}" if driver.reason else driver.name
        return action.name.lower()

    def _build_post_mortem(self, signals: list[Signal]) -> PostMortem:
        # recent_history == 0 must mean "no tail"; a plain [-0:] slice would
        # return the whole history because -0 == 0.
        recent = self._history[-self._recent_history :] if self._recent_history else []
        return PostMortem(
            action=Action.STOP,
            reason=self._action_reason(Action.STOP, signals),
            stopped_at_index=len(self._history) - 1,
            events_processed=len(self._history),
            signals=tuple(signals),
            recent_history=tuple(recent),
        )
