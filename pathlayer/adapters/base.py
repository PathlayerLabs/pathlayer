"""Abstract adapter interface.

An adapter observes an agent run and yields normalized `ToolEvent`s. It must
remain a passive side-effect (observer pattern) and never mutate the agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pathlayer.events import ToolEvent

# Callback invoked for every captured event (e.g. fed to the detector pipeline).
EventSink = Callable[[ToolEvent], None]


@runtime_checkable
class RunAwareEventSink(Protocol):
    """Callable event sink with explicit logical-run lifecycle hooks.

    Adapters may use the lifecycle methods when they observe a real run
    boundary. Existing callers can keep treating the object as an EventSink.
    """

    def __call__(self, event: ToolEvent) -> None: ...

    def start_run(self, *, started_at: float | None = None) -> None: ...

    def end_run(self) -> None: ...


class Adapter(ABC):
    """Base class for all framework adapters."""

    def __init__(self, sink: EventSink) -> None:
        self._sink = sink

    @abstractmethod
    def attach(self) -> None:
        """Install hooks on the target framework. Must be non-intrusive."""
        raise NotImplementedError

    @abstractmethod
    def detach(self) -> None:
        """Remove hooks and release any resources."""
        raise NotImplementedError

    def emit(self, event: ToolEvent) -> None:
        self._sink(event)
