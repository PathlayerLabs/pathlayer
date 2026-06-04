"""ResourceDetector (class iii: resource / lifecycle runaways).

Covers the sub-agent RAM-exhaustion loop and the crash-resume loop -- both
invisible to a tool-call-only model. V1 is DETECT-ONLY (warn/escalate); hard
memory enforcement is deferred to the runtime phase.

Signals: sub-agent spawn count, delegation depth, run-restart count,
wall-clock; optional RSS sampling. Cross-run state persists in local SQLite.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.events import ToolEvent


class ResourceDetector(Detector):
    name = "resource"

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        # TODO: count spawns/depth/restarts/wall-clock; read cross-run state.
        return Signal(self.name, confidence=0.0)
