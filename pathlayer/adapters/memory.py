"""In-memory adapter (V1 test/replay adapter).

Replays a sequence of pre-recorded ``ToolEvent``s into the detector pipeline.
It exists so the pipeline can be driven from tests and offline trace replay
without a real agent framework. Like every adapter it is a passive observer:
it only pushes events to the sink, it never inspects or mutates them.

This is not a framework integration; the first concrete *framework* adapter is
OpenClaw, which lands later in ``ADP-1``.
"""

from __future__ import annotations

from collections.abc import Iterable

from pathlayer.adapters.base import Adapter, EventSink
from pathlayer.events import ToolEvent


class InMemoryAdapter(Adapter):
    """Replays a fixed list of events, in order, to the sink.

    Events passed at construction are emitted on :meth:`attach`. Additional
    events can be pushed one at a time with :meth:`feed` (e.g. a live test
    driver). The adapter keeps no other state and performs no I/O.
    """

    def __init__(self, sink: EventSink, events: Iterable[ToolEvent] | None = None) -> None:
        super().__init__(sink)
        self._events: list[ToolEvent] = list(events) if events is not None else []

    def attach(self) -> None:
        """Replay every queued event to the sink, in order."""
        for event in self._events:
            self.emit(event)

    def detach(self) -> None:
        """No resources to release; kept for interface symmetry."""

    def feed(self, event: ToolEvent) -> None:
        """Emit a single event immediately (passive: no inspection)."""
        self.emit(event)
