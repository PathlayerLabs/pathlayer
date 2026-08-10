"""OpenClaw adapter (ADP-1, issue #40): first real-agent capture adapter.

OpenClaw (https://github.com/openclaw/openclaw) runs as a separate Node.js
process (the Gateway), so there is no shared Python call stack to hook into
in-process the way :class:`~pathlayer.adapters.langgraph.LangGraphAdapter`
does. Capture instead goes over a loopback HTTP connection
(:class:`~pathlayer.adapters.http_ingest.HttpIngestAdapter`): a small
JavaScript plugin registered on OpenClaw's own plugin hook API
(https://docs.openclaw.ai/plugins/hooks) forwards each hook firing here
verbatim, and all translation logic lives in this module where it is typed,
reviewed and unit-tested without needing a live OpenClaw instance. See
``examples/openclaw_plugin/`` for the reference plugin and the runbook used to
validate it against a real OpenClaw instance.

Scope (V1): ``after_tool_call`` is translated into :class:`~pathlayer.events.ToolEvent`
(observation-only, matches the "passive observer" contract -- V1 never blocks or
rewrites a tool call). Lifecycle hooks (``before_agent_run``, ``agent_end``) are
handled by :mod:`pathlayer.adapters.openclaw_recorder` and are not forced into
``ToolEvent``. Other hook names are ignored rather than raising.
"""

from __future__ import annotations

from typing import Any

from pathlayer.adapters.base import Adapter, EventSink
from pathlayer.adapters.http_ingest import HttpIngestAdapter
from pathlayer.adapters.openclaw_lifecycle import (
    AFTER_TOOL_CALL,
    extract_openclaw_context,
    extract_openclaw_run_id,
)
from pathlayer.events import EventKind, ToolEvent

# The source key the reference plugin (examples/openclaw_plugin/) sends.
SOURCE = "openclaw"

# Documented default so the plugin has a fixed target with no extra
# coordination needed. Overridable (e.g. if the port is already taken). Kept
# in sync by hand with the same literal in
# examples/openclaw_plugin/pathlayer-plugin/index.js -- see the comment there.
DEFAULT_PORT = 8787


def translate_openclaw_event(payload: Any) -> list[ToolEvent]:
    """Translate one OpenClaw ``after_tool_call`` payload into ``ToolEvent``s.

    ``payload`` is whatever the reference plugin POSTs for a single hook
    firing: ``{"hook": <hook name>, "event": {...}, "ctx": {...}}``.

    Raises ``ValueError`` on a structurally invalid ``after_tool_call``
    payload (missing/falsy ``toolName``, or a non-object ``payload``).
    Unknown/unhandled hook names return an empty list.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"OpenClaw payload must be an object, got {type(payload).__name__}")
    if payload.get("hook") != AFTER_TOOL_CALL:
        return []

    event = payload.get("event") or {}
    ctx = payload.get("ctx") or {}
    if not isinstance(event, dict):
        event = {}
    if not isinstance(ctx, dict):
        ctx = {}
    if not event.get("toolName"):
        raise ValueError("after_tool_call payload missing 'event.toolName'")

    params = event.get("params")
    if not isinstance(params, dict):
        params = {}

    context = extract_openclaw_context(payload)
    metadata: dict[str, Any] = {}
    if context.agent_id is not None:
        metadata["agent_id"] = context.agent_id
    if context.session_key is not None:
        metadata["session_key"] = context.session_key
    if event.get("toolCallId") is not None:
        metadata["tool_call_id"] = event["toolCallId"]
    if event.get("error") is not None:
        metadata["error"] = event["error"]
    if event.get("durationMs") is not None:
        metadata["duration_ms"] = event["durationMs"]

    return [
        ToolEvent(
            tool=str(event["toolName"]),
            args=dict(params),
            result=event.get("result"),
            kind=EventKind.TOOL_CALL,
            run_id=extract_openclaw_run_id(payload),
            metadata=metadata,
        )
    ]


class OpenClawAdapter(Adapter):
    """Loopback HTTP adapter wired to the OpenClaw event translator.

    A thin wrapper over :class:`~pathlayer.adapters.http_ingest.HttpIngestAdapter`
    that fixes the translator and gives OpenClaw a documented default port.
    """

    def __init__(
        self, sink: EventSink, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT
    ) -> None:
        super().__init__(sink)
        self._http = HttpIngestAdapter(
            sink, translators={SOURCE: translate_openclaw_event}, host=host, port=port
        )

    @property
    def port(self) -> int:
        return self._http.port

    def attach(self) -> None:
        self._http.attach()

    def detach(self) -> None:
        self._http.detach()
