"""VAL-1: on-disk trace format + loader/writer for recorded agent runs.

A *trace* is a labeled sequence of :class:`~pathlayer.events.ToolEvent`. It is
the unit of validation: replaying it through the pipeline tells us whether the
detectors fire when (and only when) they should.

On disk a trace is a single JSON object::

    {
      "label": "loop",
      "description": "why this run is a loop / no_progress / healthy",
      "events": [ {"tool": "list_dir", "args": {"path": "src"}}, ... ]
    }

Each event is the serialized form of a ``ToolEvent``; fields left at their
dataclass default are omitted on write and restored on read, so simple traces
stay terse. The format is built on ``ToolEvent`` -- it is not a second event
schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathlayer.events import EventKind, ToolEvent

HEALTHY_LABEL = "healthy"

# Neutral label for a trace whose class was never established -- a run recorded
# live (ADP-2) is the case that needs it: reaching the end of a run proves the
# agent finished, never that it was healthy. It is deliberately **not** part of
# :data:`CURATED_LABELS`: the golden set (VAL-2) requires a minimum number of
# traces per curated class, and the replay harness (VAL-3) scores
# ``healthy`` against everything else -- neither statement is meaningful for an
# unclassified run, so such a trace never belongs to the scored corpus.
UNKNOWN_LABEL = "unknown"

# The three hand-curated behavioural classes of the golden set.
CURATED_LABELS: tuple[str, ...] = ("loop", "no_progress", HEALTHY_LABEL)

VALID_LABELS: tuple[str, ...] = (*CURATED_LABELS, UNKNOWN_LABEL)


@dataclass(slots=True)
class Trace:
    """A labeled, ordered sequence of events recorded from one agent run."""

    label: str
    events: list[ToolEvent] = field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(f"invalid trace label {self.label!r}; expected one of {VALID_LABELS}")


def event_to_dict(event: ToolEvent) -> dict[str, Any]:
    """Serialize a ``ToolEvent``, omitting fields left at their default."""
    data: dict[str, Any] = {"tool": event.tool}
    if event.args:
        data["args"] = event.args
    if event.result is not None:
        data["result"] = event.result
    if event.cost_usd:
        data["cost_usd"] = event.cost_usd
    if event.ts:
        data["ts"] = event.ts
    if event.kind is not EventKind.TOOL_CALL:
        data["kind"] = event.kind.value
    if event.run_id is not None:
        data["run_id"] = event.run_id
    if event.metadata:
        data["metadata"] = event.metadata
    return data


def event_from_dict(data: dict[str, Any]) -> ToolEvent:
    """Rebuild a ``ToolEvent`` from a dict, defaulting any missing field."""
    return ToolEvent(
        tool=data["tool"],
        args=dict(data.get("args", {})),
        result=data.get("result"),
        cost_usd=data.get("cost_usd", 0.0),
        ts=data.get("ts", 0.0),
        kind=EventKind(data.get("kind", EventKind.TOOL_CALL.value)),
        run_id=data.get("run_id"),
        metadata=dict(data.get("metadata", {})),
    )


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    """Serialize a ``Trace`` to a plain JSON-ready dict."""
    data: dict[str, Any] = {"label": trace.label}
    if trace.description:
        data["description"] = trace.description
    if trace.metadata:
        data["metadata"] = trace.metadata
    data["events"] = [event_to_dict(event) for event in trace.events]
    return data


def trace_from_dict(data: dict[str, Any]) -> Trace:
    """Rebuild a ``Trace`` from a dict produced by :func:`trace_to_dict`."""
    return Trace(
        label=data["label"],
        events=[event_from_dict(event) for event in data.get("events", [])],
        description=data.get("description", ""),
        metadata=dict(data.get("metadata", {})),
    )


def load_trace(path: str | Path) -> Trace:
    """Load a labeled trace from a JSON file."""
    raw = Path(path).read_text(encoding="utf-8")
    return trace_from_dict(json.loads(raw))


def dump_trace(trace: Trace, path: str | Path) -> None:
    """Write a trace to a JSON file (pretty-printed, UTF-8, trailing newline)."""
    payload = json.dumps(trace_to_dict(trace), indent=2, ensure_ascii=False)
    Path(path).write_text(payload + "\n", encoding="utf-8")
