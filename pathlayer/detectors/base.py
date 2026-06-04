"""Detector interface and signal type.

Every detector consumes the running event history and returns a `Signal`: a
continuous confidence in [0, 1] plus a short human-readable reason. The
arbitration layer (`pathlayer.policies`) fuses signals into an action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from pathlayer.events import ToolEvent


@dataclass(slots=True)
class Signal:
    name: str
    confidence: float  # in [0, 1]; higher = more likely a failure/no-progress
    reason: str = ""


class Detector(ABC):
    """Base class for all detectors."""

    name: str = "detector"

    @abstractmethod
    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        """Return a confidence signal given the observed event history."""
        raise NotImplementedError
