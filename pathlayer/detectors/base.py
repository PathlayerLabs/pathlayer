"""Detector interface and signal type.

Every detector consumes the running event history and returns a `Signal`: a
continuous confidence in [0, 1] plus a short human-readable reason. The
arbitration layer (`pathlayer.policies`) fuses signals into an action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from pathlayer.events import EventKind, ToolEvent


@dataclass(slots=True)
class Signal:
    name: str
    confidence: float  # in [0, 1]; higher = more likely a failure/no-progress
    reason: str = ""


class Detector(ABC):
    """Base class for all detectors.

    ``event_kinds`` declares which :class:`~pathlayer.events.EventKind`\\ s the
    detector reasons about. ``None`` (the default) means "all events", which
    preserves the historical behaviour. A detector that only reasons about agent
    *actions* (loop/retry/no_progress) opts into ``{EventKind.TOOL_CALL}`` and
    calls :meth:`_relevant` at the top of :meth:`evaluate`, so that injected
    non-action events (e.g. ``LLM_CALL`` cost events) cannot fool it -- a
    reasoning agent issuing several LLM calls in a row must not look like a loop,
    nor inflate the no-progress CUSUM. Keeping the filter *inside* ``evaluate``
    (not in the pipeline) keeps each detector a pure function of the history it
    declares, identical whether driven by the pipeline or called directly.
    """

    name: str = "detector"
    event_kinds: frozenset[EventKind] | None = None

    @abstractmethod
    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        """Return a confidence signal given the observed event history."""
        raise NotImplementedError

    def _relevant(self, history: Sequence[ToolEvent]) -> Sequence[ToolEvent]:
        """Filter ``history`` to the event kinds this detector reasons about.

        No-op when ``event_kinds`` is ``None`` (returns the input unchanged).
        """
        kinds = self.event_kinds
        if kinds is None:
            return history
        return [event for event in history if event.kind in kinds]
