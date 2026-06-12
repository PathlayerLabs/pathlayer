"""Event pipeline (INF-3): wire capture -> detectors -> policy -> action.

The pipeline is the minimal in-memory backbone of Pathlayer. It receives
normalized ``ToolEvent``s (typically from an :class:`~pathlayer.adapters.Adapter`),
keeps the running history, runs every detector over that history, fuses the
resulting :class:`~pathlayer.detectors.base.Signal`s through a
:class:`~pathlayer.policies.Policy`, and returns the graded :class:`Action`.

It performs no I/O and no LLM calls; given the same event sequence it produces
the same action sequence. Canonicalization, real detectors, real arbitration
and stop/post-mortem land in later issues -- with stub detectors and the
default policy the pipeline simply yields ``Action.NONE``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import ToolEvent
from pathlayer.observability import log_action, log_decision, log_signal
from pathlayer.policies import Action, Policy
from pathlayer.postmortem import PostMortem

DEFAULT_RECENT_HISTORY = 10


class Pipeline:
    """Drives events through detectors and the arbitration policy.

    Once arbitration decides ``STOP`` the pipeline halts: it records a
    :class:`~pathlayer.postmortem.PostMortem` and ignores any further events
    (a clean flag-based stop, not a process kill).
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        policy: Policy | None = None,
        *,
        recent_history: int = DEFAULT_RECENT_HISTORY,
        logger: logging.Logger | None = None,
    ) -> None:
        if recent_history < 0:
            raise ValueError("recent_history must be >= 0")
        self._detectors: list[Detector] = list(detectors)
        self._policy: Policy = policy if policy is not None else Policy()
        self._history: list[ToolEvent] = []
        self._actions: list[Action] = []
        self._recent_history = recent_history
        self._logger = logger
        self._stopped = False
        self._post_mortem: PostMortem | None = None

    @property
    def history(self) -> tuple[ToolEvent, ...]:
        """Events actually processed, in arrival order (excludes post-stop events)."""
        return tuple(self._history)

    @property
    def actions(self) -> tuple[Action, ...]:
        """Action decided for each processed event, in order."""
        return tuple(self._actions)

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
        """
        if self._stopped:
            return Action.STOP
        self._history.append(event)
        try:
            history = tuple(self._history)
            signals = [detector.evaluate(history) for detector in self._detectors]
            action = self._policy.decide(signals)
        except Exception:
            self._history.pop()
            raise
        self._actions.append(action)
        self._log(event, signals, action)
        if action is Action.STOP:
            self._stopped = True
            self._post_mortem = self._build_post_mortem(signals)
        return action

    def run(self, events: Iterable[ToolEvent]) -> list[Action]:
        """Process events until the first STOP (inclusive) or exhaustion."""
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
        index = len(self._history) - 1
        for signal in signals:
            log_signal(self._logger, signal, index=index, ts=event.ts)
        log_decision(self._logger, action, signals, index=index, ts=event.ts)
        if action is not Action.NONE:
            log_action(self._logger, action, index=index, ts=event.ts)

    def _build_post_mortem(self, signals: list[Signal]) -> PostMortem:
        if signals:
            peak = max(signals, key=lambda signal: signal.confidence)
            reason = f"{peak.name}: {peak.reason}" if peak.reason else peak.name
        else:
            # A custom policy could in principle STOP with no signals.
            reason = "stop"
        # recent_history == 0 must mean "no tail"; a plain [-0:] slice would
        # return the whole history because -0 == 0.
        recent = self._history[-self._recent_history :] if self._recent_history else []
        return PostMortem(
            action=Action.STOP,
            reason=reason,
            stopped_at_index=len(self._history) - 1,
            events_processed=len(self._history),
            signals=tuple(signals),
            recent_history=tuple(recent),
        )
