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

Scope (V1): only ``after_tool_call`` is translated (observation-only, matches
the "passive observer" contract -- V1 never blocks or rewrites a tool call).
Every other hook name is ignored (translated to zero events) rather than
raising, so wiring new hooks on the plugin side can never break existing
capture. Lifecycle events (``subagent_spawned``, session restarts, ...) for
``ResourceDetector``, and LLM cost capture (parity with ADP-0's ``LLM_CALL``),
are deliberately out of scope for this first cut -- same call
``LangGraphAdapter`` made for ``SUBAGENT_SPAWN`` -- to keep this PR small and
mergeable; both are candidates for a follow-up issue once the HTTP transport
itself is validated against a real OpenClaw instance.
"""

from __future__ import annotations

from typing import Any

from pathlayer.adapters.base import Adapter, EventSink
from pathlayer.adapters.http_ingest import HttpIngestAdapter
from pathlayer.events import EventKind, ToolEvent

# The source key the reference plugin (examples/openclaw_plugin/) sends.
SOURCE = "openclaw"

# Documented default so the plugin has a fixed target with no extra
# coordination needed. Overridable (e.g. if the port is already taken). Kept
# in sync by hand with the same literal in
# examples/openclaw_plugin/pathlayer-plugin/index.js -- see the comment there.
DEFAULT_PORT = 8787

_AFTER_TOOL_CALL = "after_tool_call"


def translate_openclaw_event(payload: Any) -> list[ToolEvent]:
    """Translate one OpenClaw plugin-hook payload into zero or more ``ToolEvent``s.

    ``payload`` is whatever the reference plugin POSTs for a single hook
    firing: ``{"hook": <hook name>, "event": {...}, "ctx": {...}}``, where
    ``event``/``ctx`` follow the public plugin hook reference
    (https://docs.openclaw.ai/plugins/hooks): ``event`` carries ``toolName``,
    ``params``, ``result``, ``error``, ``durationMs``; ``ctx`` carries
    ``agentId``, ``sessionKey``, ``runId``.

    Raises ``ValueError`` on a structurally invalid ``after_tool_call``
    payload (missing/falsy ``toolName``, or a non-object ``payload``) -- a
    real bug worth surfacing loudly rather than silently dropping the event.
    Unknown/unhandled hook names return an empty list instead, since those
    are expected and not an error. Malformed ``event``/``ctx``/``params``
    (e.g. not an object) fall back to empty rather than raising, since a
    stray non-object field there is best treated as "no extra data" coming
    from a best-effort, un-typed JavaScript forwarder.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"OpenClaw payload must be an object, got {type(payload).__name__}")
    if payload.get("hook") != _AFTER_TOOL_CALL:
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

    metadata: dict[str, Any] = {}
    if ctx.get("agentId") is not None:
        metadata["agent_id"] = ctx["agentId"]
    if ctx.get("sessionKey") is not None:
        metadata["session_key"] = ctx["sessionKey"]
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
            run_id=_first_str(event.get("runId"), event.get("toolCallId"), ctx.get("runId")),
            metadata=metadata,
        )
    ]


def _first_str(*values: Any) -> str | None:
    """The first non-``None`` value, stringified -- ``run_id`` is best-effort."""
    for value in values:
        if value is not None:
            return str(value)
    return None


class OpenClawAdapter(Adapter):
    """Loopback HTTP adapter wired to the OpenClaw event translator.

    A thin wrapper over :class:`~pathlayer.adapters.http_ingest.HttpIngestAdapter`
    that fixes the translator and gives OpenClaw a documented default port.

    Usage::

        adapter = OpenClawAdapter(pipeline.sink)
        adapter.attach()
        print(f"OpenClaw plugin should POST to http://127.0.0.1:{adapter.port}/ingest")
        ...
        adapter.detach()
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
