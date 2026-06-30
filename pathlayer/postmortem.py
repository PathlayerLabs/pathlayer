"""Post-mortem (ACT-5): the structured record written when a run is stopped.

When arbitration decides ``STOP``, Pathlayer halts the run and captures *why*:
the triggering action, a human-readable reason, where in the stream it fired,
the detector signals at that step, and a tail of recent events. It is a plain
data record -- serializable to JSON for offline inspection and replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathlayer.detectors.base import Signal
from pathlayer.events import ToolEvent
from pathlayer.policies import Action


@dataclass(slots=True)
class PostMortem:
    action: Action
    reason: str
    stopped_at_index: int
    events_processed: int
    signals: tuple[Signal, ...]
    recent_history: tuple[ToolEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.name,
            "reason": self.reason,
            "stopped_at_index": self.stopped_at_index,
            "events_processed": self.events_processed,
            "signals": [
                {"name": s.name, "confidence": s.confidence, "reason": s.reason}
                for s in self.signals
            ],
            "recent_history": [
                {
                    "tool": e.tool,
                    "args": e.args,
                    "kind": e.kind.value,
                    "cost_usd": e.cost_usd,
                    "ts": e.ts,
                    # The single most useful metadata field for a forensic read of
                    # a budget/retry stop; surfaced explicitly rather than dumping
                    # the whole (potentially large/sensitive) metadata blob.
                    "error": e.metadata.get("error"),
                }
                for e in self.recent_history
            ],
        }

    def write(self, path: str | Path) -> None:
        """Write the post-mortem as indented JSON to ``path``."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
