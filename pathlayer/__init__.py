"""Pathlayer: runtime guardrail layer for AI agents.

Sits between an agent and its environment (LLM + tools + APIs) to detect
loops, repeated failures, lack of progress and cost explosions, and to
intervene during execution.

Detection is deterministic/statistical: no LLM in the detection loop.
"""

from pathlayer.config import ConfigError, PathlayerConfig, build_pipeline, load_config
from pathlayer.pipeline import Pipeline, PipelineDecision, WouldBeAction
from pathlayer.postmortem import PostMortem
from pathlayer.store import PersistentPipeline, SQLiteStateStore, StateStoreError

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "PathlayerConfig",
    "PersistentPipeline",
    "Pipeline",
    "PipelineDecision",
    "PostMortem",
    "SQLiteStateStore",
    "StateStoreError",
    "WouldBeAction",
    "__version__",
    "build_pipeline",
    "load_config",
]
