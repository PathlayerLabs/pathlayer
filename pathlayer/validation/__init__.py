"""Validation tooling: on-disk trace format, loader/writer and (later) the
replay harness used to score detector quality offline.

Everything here is built on :class:`pathlayer.events.ToolEvent` -- there is no
parallel event schema.
"""

from pathlayer.validation.traces import (
    VALID_LABELS,
    Trace,
    dump_trace,
    event_from_dict,
    event_to_dict,
    load_trace,
    trace_from_dict,
    trace_to_dict,
)

__all__ = [
    "VALID_LABELS",
    "Trace",
    "dump_trace",
    "event_from_dict",
    "event_to_dict",
    "load_trace",
    "trace_from_dict",
    "trace_to_dict",
]
