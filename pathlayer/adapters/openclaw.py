"""OpenClaw adapter: first real-agent capture adapter.

OpenClaw (https://github.com/openclaw/openclaw) runs as a separate Node.js
process (the Gateway), so there is no shared Python call stack to hook into
in-process the way :class:`~pathlayer.adapters.langgraph.LangGraphAdapter`
does. Capture instead goes over a loopback HTTP connection
(:class:`~pathlayer.adapters.http_ingest.HttpIngestAdapter`): a small
JavaScript plugin registered on OpenClaw's own plugin hook API
(https://docs.openclaw.ai/plugins/hooks) forwards each hook firing here
verbatim, and all translation logic lives in this module where it is typed,
reviewed and unit-tested without needing a live OpenClaw instance. The
reference plugin ships in ``pathlayer/adapters/openclaw_plugin/``; see
``examples/openclaw_plugin/README.md`` for the runbook used to validate it
against a real OpenClaw instance.

Scope: ``after_tool_call`` is translated into a tool event. This adapter only
observes: it never blocks or rewrites a tool call. Blocking the tool calls of
a stopped run is an opt-in decision of the run recorder (see
``OpenClawRunRecorder.tool_call_decision``). ``llm_output`` is translated
separately when it carries usable token counters: its input, output and cache
counters are priced at local rates, and a total alone is split with the
configured input ratio. Either way the cost is a clearly labelled first
approximation, not an invoice-grade measurement. Lifecycle hooks
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
from pathlayer.pricing import ModelPrice, cost_of, resolve_price

if TYPE_CHECKING:
    from pathlayer.config import PathlayerConfig

# The source key the reference plugin (pathlayer/adapters/openclaw_plugin/) sends.
SOURCE = "openclaw"

# Documented default so the plugin has a fixed target with no extra
# coordination needed. Overridable (e.g. if the port is already taken). Kept
# in sync by hand with the same literal in
# pathlayer/adapters/openclaw_plugin/index.js -- see the comment there.
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
    split_usage: bool = True,
    unwrap_model_ids: bool = True,
) -> list[ToolEvent]:
    """Estimate an LLM cost from the token counters OpenClaw reports.

    OpenClaw 2026.7 sends one ``llm_output`` per run attempt, after its
    ``agent_end``, with ``usage`` summed over the attempt's model calls:
    ``{input, output, cacheRead, cacheWrite, total}``, a counter at zero being
    left out. When any of the four split counters is present, each one is
    priced at its own local rate (``token_split: "reported"``): with prompt
    caching most tokens are cache reads, billed at a fraction of the input rate.
    A model whose cache rates are unknown leaves its cache tokens unpriced and
    counts them in ``unpriced_cache_tokens``, an under-count rather than a guess.
    A call whose only tokens are such cache tokens priced nothing at all: it is
    marked ``unmeasured_unknown_cache_rate``, like an unknown model, rather than
    presented as an estimated $0.
    Without a split, the reported ``total`` is divided with ``input_ratio``
    (``token_split: "input_ratio"``). A split with a malformed counter falls back
    to the total, so a bad field never makes the estimate look precise.

    Either way the result is marked ``estimated_total_tokens``, the label for a
    cost estimated from token counters at local rates: a first approximation,
    not an invoice-grade or provider-measured cost. ``split_usage=False`` always
    divides the total, and ``unwrap_model_ids=False`` matches the raw model id
    without unwrapping a provider prefix: the rules of a run recorded by
    Pathlayer 0.1.0, so its report keeps the figure computed at the time. A
    split whose counters are all zero while the total is positive is not a
    split: the total is divided instead.

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
    total = _token_count(usage.get("total", usage.get("totalTokens")))
    split = _reported_split(usage) if split_usage else None
    if split is not None and not any(split) and total:
        split = None
    model_raw = event.get("model") or ctx.get("model") or ctx.get("resolvedModel")
    model = str(model_raw) if model_raw else None
    price = resolve_price(model, prices, unwrap_ids=unwrap_model_ids)
    # A model configured as free costs nothing, its cache tokens included.
    free = price is not None and not (price.input_usd_per_mtok or price.output_usd_per_mtok)
    metadata: dict[str, Any] = {}
    if split is not None:
        input_tokens, output_tokens, cache_read, cache_write = split
        metadata.update(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            token_split="reported",
        )
        unpriced = 0
        if price is not None and not free and price.cache_read_usd_per_mtok is None:
            unpriced += cache_read
        if price is not None and not free and price.cache_write_usd_per_mtok is None:
            unpriced += cache_write
        if unpriced:
            metadata["unpriced_cache_tokens"] = unpriced
        reasoning = _token_count(usage.get("reasoningTokens"))
        if reasoning is not None:
            # Kept as evidence only: providers count reasoning inside output.
            metadata["reasoning_tokens"] = reasoning
    elif total is not None:
        input_tokens = round(total * input_ratio)
        output_tokens = total - input_tokens
        cache_read = cache_write = 0
        metadata.update(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_ratio=input_ratio,
            token_split="input_ratio",
        )
    else:
        return []
    if total is not None:
        metadata["reported_total_tokens"] = total
    unpriced_cache = metadata.get("unpriced_cache_tokens", 0)
    all_tokens = input_tokens + output_tokens + cache_read + cache_write
    if price is None:
        metadata["cost_measurement"] = "unmeasured_unknown_model"
    elif unpriced_cache and unpriced_cache == all_tokens:
        metadata["cost_measurement"] = "unmeasured_unknown_cache_rate"
    else:
        metadata["cost_measurement"] = "estimated_total_tokens"
    context = extract_openclaw_context(payload)
    if context.agent_id is not None:
        metadata["agent_id"] = context.agent_id
    if context.session_key is not None:
        metadata["session_key"] = context.session_key
    return [
        ToolEvent(
            tool=model or "<llm>",
            kind=EventKind.LLM_CALL,
            cost_usd=cost_of(
                model,
                input_tokens,
                output_tokens,
                prices,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                unwrap_ids=unwrap_model_ids,
            ),
            run_id=context.run_id,
            metadata=metadata,
        )
    ]


_SPLIT_COUNTERS = ("input", "output", "cacheRead", "cacheWrite")


def _token_count(value: Any) -> int | None:
    """A token counter, or ``None`` unless it is a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _reported_split(usage: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    """``(input, output, cacheRead, cacheWrite)`` when OpenClaw reported a split.

    OpenClaw omits a counter at zero, so an absent one counts as 0 once any of
    the four is present. A present counter that is not a non-negative integer
    makes the whole split unusable.
    """
    if not any(name in usage for name in _SPLIT_COUNTERS):
        return None
    counts: list[int] = []
    for name in _SPLIT_COUNTERS:
        if name not in usage:
            counts.append(0)
            continue
        count = _token_count(usage[name])
        if count is None:
            return None
        counts.append(count)
    return counts[0], counts[1], counts[2], counts[3]


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
