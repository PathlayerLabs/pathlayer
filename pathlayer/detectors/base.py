"""Detector interface and signal type.

Every detector consumes the running event history and returns a `Signal`: a
continuous confidence in [0, 1] plus a short human-readable reason. The
arbitration layer (`pathlayer.policies`) fuses signals into an action.

PERF-2 adds an optional incremental fold, ``create_state``/``step``, alongside
``evaluate``. ``evaluate(history)`` stays the pure specification -- same
history in, same ``Signal`` out, forever. ``step`` is only ever a *faster
equivalent implementation* of that same contract: folding ``step`` one event at
a time from ``create_state()`` must produce, after each event, the same
``Signal`` as calling ``evaluate`` on the prefix ending at that event (verified
by prefix-by-prefix equivalence tests, not just asserted). Neither method is
abstract, so a detector that implements only ``evaluate`` (every custom or
third-party detector written before PERF-2) keeps working unmodified: the
default ``step`` below rebuilds the growing history and calls ``evaluate`` on
it -- correct, just not O(1) amortized. A detector opts into the faster path by
overriding *both* ``create_state`` and ``step`` together (see
``NoProgressDetector`` for the reference implementation); overriding only one
of the two breaks the pairing (``step`` would receive a state shape
``create_state`` never produces) and is a programming error, not a supported
configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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

    def create_state(self) -> Any:
        """Return a fresh incremental state for :meth:`step` (PERF-2).

        Default: an empty tuple, consumed by the default :meth:`step` below as
        the batch-rebuild fallback. A detector overriding both this and
        :meth:`step` for the O(1)-amortized path returns whatever state shape
        its own :meth:`step` expects instead.
        """
        return ()

    def step(self, state: Any, event: ToolEvent) -> tuple[Any, Signal]:
        """Advance the incremental fold by one event; see the module docstring.

        Default (batch fallback): append ``event`` to the state history and call
        :meth:`evaluate` on the result -- correct for every detector, including
        ones that never heard of PERF-2, just O(n) per call like ``evaluate``
        itself (no worse than calling ``evaluate`` on the growing history
        directly). ``event_kinds`` filtering still applies because ``evaluate``
        performs it internally via :meth:`_relevant`.
        """
        history = (*state, event)
        return history, self.evaluate(history)

    def _relevant(self, history: Sequence[ToolEvent]) -> Sequence[ToolEvent]:
        """Filter ``history`` to the event kinds this detector reasons about.

        No-op when ``event_kinds`` is ``None`` (returns the input unchanged).
        """
        kinds = self.event_kinds
        if kinds is None:
            return history
        return [event for event in history if event.kind in kinds]
