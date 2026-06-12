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

from collections.abc import Iterable, Sequence

from pathlayer.detectors.base import Detector
from pathlayer.events import ToolEvent
from pathlayer.policies import Action, Policy


class Pipeline:
    """Drives events through detectors and the arbitration policy."""

    def __init__(self, detectors: Sequence[Detector], policy: Policy | None = None) -> None:
        self._detectors: list[Detector] = list(detectors)
        self._policy: Policy = policy if policy is not None else Policy()
        self._history: list[ToolEvent] = []
        self._actions: list[Action] = []

    @property
    def history(self) -> tuple[ToolEvent, ...]:
        """Events seen so far, in arrival order."""
        return tuple(self._history)

    @property
    def actions(self) -> tuple[Action, ...]:
        """Action decided for each processed event, in order."""
        return tuple(self._actions)

    def process(self, event: ToolEvent) -> Action:
        """Process one event and return the arbitrated action.

        Detectors receive an immutable snapshot of the history so a buggy
        detector cannot mutate pipeline state. If a detector or the policy
        raises, the event is rolled back so ``history`` and ``actions`` stay in
        sync for subsequent calls.
        """
        self._history.append(event)
        try:
            history = tuple(self._history)
            signals = [detector.evaluate(history) for detector in self._detectors]
            action = self._policy.decide(signals)
        except Exception:
            self._history.pop()
            raise
        self._actions.append(action)
        return action

    def run(self, events: Iterable[ToolEvent]) -> list[Action]:
        """Process a batch of events, returning one action per event."""
        return [self.process(event) for event in events]

    def sink(self, event: ToolEvent) -> None:
        """Adapter-compatible sink (see :data:`pathlayer.adapters.base.EventSink`)."""
        self.process(event)
