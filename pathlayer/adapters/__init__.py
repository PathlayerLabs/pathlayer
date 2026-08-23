"""Capture layer: framework adapters.

Each adapter hooks into a specific agent framework and emits normalized
`ToolEvent`s. Adapters are passive observers; they never modify the agent flow.

Concrete adapters: OpenClaw (ADP-1, this package), LangGraph/LangChain (ADP-0,
`pathlayer.adapters.langgraph` -- not re-exported here since it pulls in an
optional dependency). Planned: OpenAI Agents SDK, CrewAI, AutoGen, browser-use.
Long term: OpenTelemetry.
"""

from pathlayer.adapters.base import Adapter, RunAwareEventSink
from pathlayer.adapters.http_ingest import HttpIngestAdapter
from pathlayer.adapters.memory import InMemoryAdapter
from pathlayer.adapters.openclaw import OpenClawAdapter

__all__ = [
    "Adapter",
    "HttpIngestAdapter",
    "InMemoryAdapter",
    "OpenClawAdapter",
    "RunAwareEventSink",
]
