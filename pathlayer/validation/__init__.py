"""Offline tooling for trace replay and benchmark evidence collection.

Behavioral traces reuse :class:`pathlayer.events.ToolEvent`; COMP-1 state
observations use the separate, explicitly correlated ``StateEvidence``
sidecar. Neither format changes the runtime event schema.
"""

from pathlayer.state_collectors import (
    collect_filesystem_evidence,
    collect_pytest_evidence,
    dump_state_evidence_export,
    load_state_evidence_export,
    unavailable_state_evidence,
)
from pathlayer.validation.corpus import CorpusSummary, build_corpus
from pathlayer.validation.mutations import (
    inject_cost_spike,
    inject_loop,
    inject_no_progress,
    inject_resource_spike,
    inject_retry_storm,
)
from pathlayer.validation.replay import (
    ReplayResult,
    Report,
    load_traces,
    replay_trace,
    score,
)
from pathlayer.validation.traces import (
    CURATED_LABELS,
    HEALTHY_LABEL,
    UNKNOWN_LABEL,
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
    "CURATED_LABELS",
    "HEALTHY_LABEL",
    "UNKNOWN_LABEL",
    "VALID_LABELS",
    "CorpusSummary",
    "ReplayResult",
    "Report",
    "Trace",
    "build_corpus",
    "collect_filesystem_evidence",
    "collect_pytest_evidence",
    "dump_state_evidence_export",
    "dump_trace",
    "event_from_dict",
    "event_to_dict",
    "inject_cost_spike",
    "inject_loop",
    "inject_no_progress",
    "inject_resource_spike",
    "inject_retry_storm",
    "load_state_evidence_export",
    "load_trace",
    "load_traces",
    "replay_trace",
    "score",
    "trace_from_dict",
    "trace_to_dict",
    "unavailable_state_evidence",
]
