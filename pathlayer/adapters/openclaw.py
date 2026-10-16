"""OpenClaw adapter (ADP-1): first real-agent capture adapter.

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

Scope (V1): ``after_tool_call`` is translated into a tool event (observation-only,
matches the "passive observer" contract -- V1 never blocks or rewrites a tool
call). ``llm_output`` is translated separately only when it contains a model and
a non-negative total-token counter. When the provider does not expose the
input/output split, the configured input ratio yields a clearly labelled first
cost approximation; it is not an invoice-grade measurement. Lifecycle hooks
(``before_agent_run``, ``agent_end``) are handled by
:mod:`pathlayer.adapters.openclaw_recorder`. Other hook names are ignored rather
than raising.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pathlayer.adapters.base import Adapter, EventSink
from pathlayer.adapters.http_ingest import HttpIngestAdapter
from pathlayer.adapters.openclaw_lifecycle import (
    AFTER_TOOL_CALL,
    LLM_OUTPUT,
    extract_openclaw_context,
    extract_openclaw_run_id,
)
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pricing import ModelPrice, cost_of, has_price

if TYPE_CHECKING:
    from pathlayer.config import PathlayerConfig

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


def translate_openclaw_llm_event(
    payload: Any,
    *,
    prices: Mapping[str, ModelPrice] | None = None,
    input_ratio: float = 0.8,
) -> list[ToolEvent]:
    """Estimate an LLM cost from OpenClaw's reported total token counter.

    ``llm_output`` carries one model invocation and its ``usage.total``. Some
    OpenClaw/provider combinations do not expose the input/output split. In
    that case Pathlayer assigns ``total * input_ratio`` to input and the
    remainder to output before applying the local configured rates. The result
    is explicitly ``estimated_total_tokens``; it is a first approximation, not
    an invoice-grade or provider-measured cost.

    Missing/malformed counters produce no LLM event. Unknown models still
    produce an event marked ``unmeasured_unknown_model`` with zero cost, so a
    free configured model is never confused with absent pricing.
    """
    _validate_input_ratio(input_ratio)
    if not isinstance(payload, dict):
        raise ValueError(f"OpenClaw payload must be an object, got {type(payload).__name__}")
    if payload.get("hook") != LLM_OUTPUT:
        return []
    event = _as_mapping(payload.get("event"))
    ctx = _as_mapping(payload.get("ctx"))
    usage = _as_mapping(event.get("usage")) or _as_mapping(ctx.get("usage"))
    total = usage.get("total", usage.get("totalTokens"))
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        return []
    model_raw = event.get("model") or ctx.get("model") or ctx.get("resolvedModel")
    model = str(model_raw) if model_raw else None
    input_tokens = round(total * input_ratio)
    output_tokens = total - input_tokens
    priced = has_price(model, prices)
    metadata: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reported_total_tokens": total,
        "input_ratio": input_ratio,
        "cost_measurement": "estimated_total_tokens" if priced else "unmeasured_unknown_model",
    }
    context = extract_openclaw_context(payload)
    if context.agent_id is not None:
        metadata["agent_id"] = context.agent_id
    if context.session_key is not None:
        metadata["session_key"] = context.session_key
    return [
        ToolEvent(
            tool=model or "<llm>",
            kind=EventKind.LLM_CALL,
            cost_usd=cost_of(model, input_tokens, output_tokens, prices),
            run_id=context.run_id,
            metadata=metadata,
        )
    ]


def _validate_input_ratio(input_ratio: float) -> None:
    if (
        isinstance(input_ratio, bool)
        or not isinstance(input_ratio, (int, float))
        or not 0.0 <= input_ratio <= 1.0
    ):
        raise ValueError("input_ratio must be a finite number in [0, 1]")


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


class OpenClawAdapter(Adapter):
    """Loopback HTTP adapter wired to the OpenClaw event translator.

    A thin wrapper over :class:`~pathlayer.adapters.http_ingest.HttpIngestAdapter`
    that fixes the translator and gives OpenClaw a documented default port.
    """

    def __init__(
        self,
        sink: EventSink,
        *,
        prices: Mapping[str, ModelPrice] | None = None,
        input_ratio: float = 0.8,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
    ) -> None:
        _validate_input_ratio(input_ratio)
        super().__init__(sink)

        def _translate(payload: Any) -> list[ToolEvent]:
            return [
                *translate_openclaw_llm_event(payload, prices=prices, input_ratio=input_ratio),
                *translate_openclaw_event(payload),
            ]

        self._http = HttpIngestAdapter(sink, translators={SOURCE: _translate}, host=host, port=port)

    @classmethod
    def from_config(
        cls,
        sink: EventSink,
        config: PathlayerConfig,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
    ) -> OpenClawAdapter:
        """Build an adapter with local configured prices and estimate ratio."""
        return cls(
            sink,
            prices=config.pricing.effective_prices(),
            input_ratio=config.pricing.openclaw_input_ratio,
            host=host,
            port=port,
        )

    @property
    def port(self) -> int:
        return self._http.port

    def attach(self) -> None:
        self._http.attach()

    def detach(self) -> None:
        self._http.detach()
