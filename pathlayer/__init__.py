"""Pathlayer: runtime guardrail layer for AI agents.

Sits between an agent and its environment (LLM + tools + APIs) to detect
loops, repeated failures, lack of progress and cost explosions, and to
intervene during execution.

Detection is deterministic/statistical: no LLM in the detection loop.
"""

from pathlayer.adapters.base import RunAwareEventSink
from pathlayer.config import ConfigError, PathlayerConfig, build_pipeline, load_config
from pathlayer.pipeline import Pipeline, PipelineDecision, WouldBeAction
from pathlayer.postmortem import (
    CodingDiagnostic,
    PostMortem,
    SessionOutcome,
    SessionReport,
    TickRecord,
    build_coding_diagnostic,
    build_session_report,
)
from pathlayer.store import PersistentPipeline, SQLiteStateStore, StateStoreError

__version__ = "0.1.0"

__all__ = [
    "CodingDiagnostic",
    "ConfigError",
    "PathlayerConfig",
    "PersistentPipeline",
    "Pipeline",
    "PipelineDecision",
    "PostMortem",
    "RunAwareEventSink",
    "SQLiteStateStore",
    "SessionOutcome",
    "SessionReport",
    "StateStoreError",
    "TickRecord",
    "WouldBeAction",
    "__version__",
    "build_coding_diagnostic",
    "build_pipeline",
    "build_session_report",
    "load_config",
]
