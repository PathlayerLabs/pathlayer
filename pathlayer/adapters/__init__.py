"""Capture layer: framework adapters.

Each adapter hooks into a specific agent framework and emits normalized
`ToolEvent`s. Adapters are passive observers; they never modify the agent flow.

First concrete adapter: OpenClaw. Planned: LangGraph, OpenAI Agents SDK,
CrewAI, AutoGen, browser-use. Long term: OpenTelemetry.
"""

from pathlayer.adapters.base import Adapter

__all__ = ["Adapter"]
