"""Pathlayer: runtime guardrail layer for AI agents.

Sits between an agent and its environment (LLM + tools + APIs) to detect
loops, repeated failures, lack of progress and cost explosions, and to
intervene during execution.

Detection is deterministic/statistical: no LLM in the detection loop.
"""

from pathlayer.config import ConfigError, PathlayerConfig, build_pipeline, load_config
from pathlayer.pipeline import Pipeline, WouldBeAction
from pathlayer.postmortem import PostMortem

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "PathlayerConfig",
    "Pipeline",
    "PostMortem",
    "WouldBeAction",
    "__version__",
    "build_pipeline",
    "load_config",
]
