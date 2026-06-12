"""Pathlayer: runtime guardrail layer for AI agents.

Sits between an agent and its environment (LLM + tools + APIs) to detect
loops, repeated failures, lack of progress and cost explosions, and to
intervene during execution.

Detection is deterministic/statistical: no LLM in the detection loop.
"""

from pathlayer.pipeline import Pipeline

__version__ = "0.0.1"

__all__ = ["Pipeline", "__version__"]
