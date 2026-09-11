"""LangGraph / LangChain capture adapter (ADP-0, first framework adapter).

Captures tool calls from a LangGraph/LangChain run into normalized
:class:`~pathlayer.events.ToolEvent`\\ s, via a passive
``BaseCallbackHandler``. It never alters the agent flow (observer pattern): it
only reads callback payloads and pushes events to the sink.

Scope (V1): tool-call capture for both ``invoke`` and ``ainvoke``, with **raw
lineage** -- ``parent_run_id`` is stored in ``metadata`` so delegation depth can
be reconstructed offline later. We do *not* classify ``SUBAGENT_SPAWN`` (a
name-based heuristic would be noisy); lifecycle events are out of scope here.

LLM cost (issue #77): ``on_llm_start``/``on_llm_end`` are paired by ``run_id``
(like tools) to emit a dedicated ``EventKind.LLM_CALL`` event carrying the
token cost in ``cost_usd``, so ``BudgetDetector`` can guard model spend (the
real cost is in tokens, not tool calls). Tokens come from the model response
(``usage_metadata`` per generation, falling back to ``llm_output``); the price
table lives in :mod:`pathlayer.pricing` and is overridable via ``prices``.
Missing tokens or an unpriceable model yield ``cost_usd=0.0`` (never a crash).
Action detectors ignore ``LLM_CALL`` events via their ``event_kinds`` filter.

Thread-safety: a single sync handler serves async runs too -- LangChain invokes
sync callbacks from a thread pool (``run_in_executor``) during ``ainvoke``. A
:class:`threading.Lock` therefore guards the ``pending`` map, the ``ts`` counter
*and* the :meth:`emit` call, so concurrent tool callbacks are fully serialized:
the (possibly non-thread-safe) downstream sink sees one event at a time and the
``ts`` order always matches the emission order. ``ts`` reflects the order in
which calls *complete*, which under genuinely parallel tools may differ from the
order in which they started.

This module imports ``langchain_core`` and so is an **optional** import: install
the ``langgraph`` extra (``pip install pathlayer[langgraph]``). It is
deliberately *not* re-exported from :mod:`pathlayer.adapters`.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult

from pathlayer.adapters.base import Adapter, EventSink, RunAwareEventSink
from pathlayer.events import PROGRESS_SUMMARY_TOOL, EventKind, ToolEvent
from pathlayer.pricing import ModelPrice, cost_of
from pathlayer.state_evidence import StateCapture, StateEvidenceProducer

if TYPE_CHECKING:
    from pathlayer.config import PathlayerConfig

# Bound on un-paired ``on_tool_start`` calls held in ``_pending``. A start that
# is never followed by an end/error (cancelled coroutine, interrupted run, a
# dropped end callback) would otherwise leak forever on a long-running agent
# attached once. We evict the oldest entry FIFO past this cap so memory stays
# bounded; the nominal start->end path is unaffected (it pops its own entry).
_MAX_PENDING = 1024

# Top-level coercion only: scalars/containers are emitted as-is, anything else is
# stringified. This handles the common case where a tool returns a non-serializable
# object directly so a VAL-1 round-trip (dump_trace/load) survives it. It is
# intentionally shallow (per the ADP-0 plan): a dict/list containing a nested
# non-serializable value would still fail ``dump_trace``. Deep coercion is out of
# scope here.
_JSON_FRIENDLY = (bool, int, float, str, dict, list)


def _unwrap_tool_message(output: Any) -> Any:
    """Return the underlying result carried by a LangChain ``ToolMessage``.

    A stock ``langgraph.prebuilt.ToolNode`` invokes each tool with the full
    ``ToolCall``, so LangChain hands the capture callback the wrapping
    ``ToolMessage`` -- not the tool's own return value. Left as-is that wrapper
    stringifies (below), hiding a structured ``{passed, failed}`` result from
    ``TestProgressAnchor`` (which requires a ``Mapping``) so ``NoProgressDetector``
    would never fire on a real ``ToolNode`` agent. We therefore unwrap it:

    - a structured ``artifact`` (the idiomatic ``response_format=
      "content_and_artifact"`` path) is the tool's real object -- use it as-is;
    - otherwise the ``content``; a JSON object/array *string* (the default
      serialization of a dict/list return) is parsed back to the container so
      the structure survives. Plain-text / scalar-looking content is left as a
      string (no surprise coercion of e.g. ``"11"`` to an int).

    Any non-``ToolMessage`` output is returned untouched.
    """
    if not isinstance(output, ToolMessage):
        return output
    if output.artifact is not None:
        return output.artifact
    content = output.content
    if isinstance(content, str):
        stripped = content.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except (ValueError, RecursionError):
                # ValueError covers malformed JSON (JSONDecodeError); RecursionError
                # guards deeply-nested content -- neither must escape the callback
                # (the module's "never a crash" contract).
                return content
    return content


def _coerce_result(output: Any) -> Any:
    """Coerce a tool output to a JSON-friendly value at the top level.

    A ``ToolMessage`` wrapper (emitted when a tool is run via ``ToolNode``) is
    unwrapped to the tool's real result first (see :func:`_unwrap_tool_message`).
    Scalars/dict/list are then kept as-is; anything else is stringified. Shallow
    by design: nested non-serializable values inside a dict/list are not coerced.
    """
    output = _unwrap_tool_message(output)
    if output is None or isinstance(output, _JSON_FRIENDLY):
        return output
    return str(output)


@dataclass(slots=True)
class _Pending:
    """A tool call seen at ``on_tool_start``, awaiting its end/error.

    ``kind`` is classified once, at start, from the tool name: the Pathlayer
    summary tool (``PROGRESS_SUMMARY_TOOL``, ACT-2) is captured as
    ``EventKind.PROGRESS_SUMMARY`` so the detectors ignore it; every other tool
    is a plain ``TOOL_CALL``. This is pure capture-time classification -- no tier
    decision descends into the adapter.
    """

    tool: str
    args: dict[str, Any]
    parent_run_id: str | None
    kind: EventKind = EventKind.TOOL_CALL
    state_capture: StateCapture | None = None


@dataclass(slots=True)
class _PendingLLM:
    """An LLM call seen at ``on_llm_start``, awaiting its end/error."""

    model: str | None
    parent_run_id: str | None


def _model_from_serialized(serialized: dict[str, Any] | None) -> str | None:
    """Best-effort model id from an ``on_llm_start`` ``serialized`` payload."""
    if not serialized:
        return None
    kwargs = serialized.get("kwargs") or {}
    model = kwargs.get("model") or kwargs.get("model_name")
    return str(model) if model else None


def _extract_tokens(response: LLMResult) -> tuple[int, int]:
    """Return ``(input_tokens, output_tokens)`` from an ``LLMResult``.

    Prefers per-generation ``message.usage_metadata`` (chat models); falls back
    to ``llm_output["token_usage"]`` (``prompt_tokens``/``completion_tokens``).
    Returns ``(0, 0)`` when neither is present (some providers report nothing).
    """
    input_tokens = 0
    output_tokens = 0
    found = False
    for generations in response.generations:
        # ``generations`` is the candidate list for one prompt; usage is reported
        # once per call, so count the first candidate that carries it and stop --
        # summing every candidate would multiply the (shared) prompt tokens when
        # n > 1. The outer loop still sums across prompts in a batch.
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) if message is not None else None
            if usage:
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
                found = True
                break
    if found:
        return input_tokens, output_tokens
    token_usage = (response.llm_output or {}).get("token_usage") or {}
    return (
        int(token_usage.get("prompt_tokens", 0) or 0),
        int(token_usage.get("completion_tokens", 0) or 0),
    )


def _model_from_response(response: LLMResult) -> str | None:
    """Model id reported on the response, if any (``llm_output['model_name']``)."""
    model = (response.llm_output or {}).get("model_name")
    return str(model) if model else None


class LangGraphAdapter(Adapter):
    """Capture tool calls from a LangGraph/LangChain run.

    Usage::

        adapter = LangGraphAdapter(pipeline.sink)
        adapter.attach()
        result = my_tool.invoke(payload, config=adapter.as_config())
        # ... or graph.invoke(state, config=adapter.as_config())
        adapter.detach()

    Pass :meth:`as_config` into any ``invoke``/``ainvoke`` call so LangChain
    routes its callbacks to this adapter's handler.
    """

    def __init__(
        self,
        sink: EventSink,
        prices: Mapping[str, ModelPrice] | None = None,
        *,
        state_evidence: StateEvidenceProducer | None = None,
    ) -> None:
        super().__init__(sink)
        self._handler = _LangChainCaptureHandler(self)
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, _Pending] = OrderedDict()
        self._pending_llm: OrderedDict[str, _PendingLLM] = OrderedDict()
        self._ts = 0
        self._prices = prices
        self._state_evidence = state_evidence

    @classmethod
    def from_config(
        cls,
        sink: EventSink,
        config: PathlayerConfig,
        *,
        state_evidence: StateEvidenceProducer | None = None,
    ) -> LangGraphAdapter:
        """Build an adapter with the configured local model price table.

        This is the official construction path when ``PathlayerConfig.pricing``
        should affect budget decisions. It deliberately does no remote price
        lookup: the configuration snapshot is the complete pricing authority.
        """
        return cls(sink, prices=config.pricing.effective_prices(), state_evidence=state_evidence)

    def attach(self) -> None:
        """Reset capture state (deterministic ``ts`` per attached run)."""
        with self._lock:
            self._pending.clear()
            self._pending_llm.clear()
            self._ts = 0

    def detach(self) -> None:
        """Drop any unpaired pending calls; the handler keeps no other state."""
        with self._lock:
            self._pending.clear()
            self._pending_llm.clear()
        if isinstance(self._sink, RunAwareEventSink):
            self._sink.end_run()

    def as_config(self) -> dict[str, Any]:
        """Return a LangChain ``RunnableConfig`` wiring our capture handler."""
        return {"callbacks": [self._handler]}

    def _start_root(self, parent_run_id: str | None) -> None:
        """Start RES-2 at the earliest observed root callback, idempotently."""
        if parent_run_id is None and isinstance(self._sink, RunAwareEventSink):
            self._sink.start_run()

    def _record_start(
        self, run_id: str, parent_run_id: str | None, tool: str, args: dict[str, Any]
    ) -> None:
        self._start_root(parent_run_id)
        kind = EventKind.PROGRESS_SUMMARY if tool == PROGRESS_SUMMARY_TOOL else EventKind.TOOL_CALL
        with self._lock:
            state_capture = None
            if self._state_evidence is not None and kind is EventKind.TOOL_CALL:
                overlapping = [
                    item for item in self._pending.values() if item.state_capture is not None
                ]
                if overlapping:
                    for item in overlapping:
                        assert item.state_capture is not None
                        item.state_capture = self._state_evidence.unavailable_capture(
                            item.state_capture.tool_event_ref
                        )
                    state_capture = self._state_evidence.unavailable_capture(run_id)
                else:
                    state_capture = self._state_evidence.capture_before(run_id)
            self._pending[run_id] = _Pending(
                tool=tool,
                args=args,
                parent_run_id=parent_run_id,
                kind=kind,
                state_capture=state_capture,
            )
            # Bound the map: drop the oldest never-paired start if we exceed the cap.
            if len(self._pending) > _MAX_PENDING:
                self._pending.popitem(last=False)

    def _record_end(self, run_id: str, output: Any) -> None:
        self._emit_paired(run_id, result=_coerce_result(output), error=None)

    def _record_error(self, run_id: str, error: BaseException) -> None:
        self._emit_paired(run_id, result=None, error=str(error))

    def _emit_paired(self, run_id: str, *, result: Any, error: str | None) -> None:
        """Pair an end/error with its start and emit, all under the lock.

        Holding the lock across pop + ``ts`` allocation + :meth:`emit` keeps the
        emission order aligned with ``ts`` and serializes concurrent callbacks
        for the downstream sink. Detectors never call back into the adapter, so
        there is no re-entrancy/deadlock risk.
        """
        with self._lock:
            pending = self._pending.pop(run_id, None)
            if pending is None:
                return
            ts = self._ts
            self._ts += 1
            metadata: dict[str, Any] = {"parent_run_id": pending.parent_run_id}
            if error is not None:
                metadata["error"] = error
            event = ToolEvent(
                tool=pending.tool,
                args=pending.args,
                result=result,
                ts=ts,
                kind=pending.kind,
                run_id=run_id,
                metadata=metadata,
            )
            if self._state_evidence is not None and pending.state_capture is not None:
                event = self._state_evidence.attach_after(event, pending.state_capture)
            self.emit(event)

    def _record_llm_start(self, run_id: str, parent_run_id: str | None, model: str | None) -> None:
        self._start_root(parent_run_id)
        with self._lock:
            self._pending_llm[run_id] = _PendingLLM(model=model, parent_run_id=parent_run_id)
            # Bound the map: drop the oldest never-paired start if we exceed the cap.
            if len(self._pending_llm) > _MAX_PENDING:
                self._pending_llm.popitem(last=False)

    def _record_llm_error(self, run_id: str) -> None:
        with self._lock:
            self._pending_llm.pop(run_id, None)

    def _record_llm_end(self, run_id: str, response: LLMResult) -> None:
        """Pair an ``on_llm_end`` with its start and emit a priced LLM event.

        Like :meth:`_emit_paired`, the pop + ``ts`` allocation + :meth:`emit` are
        held under the lock so LLM and tool events share one monotonic ``ts``.
        The model on the response takes precedence over the one seen at start.
        """
        input_tokens, output_tokens = _extract_tokens(response)
        with self._lock:
            pending = self._pending_llm.pop(run_id, None)
            if pending is None:
                return
            model = _model_from_response(response) or pending.model
            cost = cost_of(model, input_tokens, output_tokens, self._prices)
            ts = self._ts
            self._ts += 1
            self.emit(
                ToolEvent(
                    tool=model or "<llm>",
                    cost_usd=cost,
                    ts=ts,
                    kind=EventKind.LLM_CALL,
                    run_id=run_id,
                    metadata={
                        "parent_run_id": pending.parent_run_id,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )
            )


def _parent(parent_run_id: UUID | None) -> str | None:
    return str(parent_run_id) if parent_run_id is not None else None


class _LangChainCaptureHandler(BaseCallbackHandler):
    """Passive ``BaseCallbackHandler`` that forwards tool calls to the adapter.

    Sync on purpose: LangChain runs sync handlers in a thread pool during async
    runs, so this single handler covers both ``invoke`` and ``ainvoke``. All
    shared state lives behind the adapter's lock.
    """

    def __init__(self, adapter: LangGraphAdapter) -> None:
        self._adapter = adapter

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool = (serialized or {}).get("name") or "tool"
        args = dict(inputs) if inputs is not None else {"input": input_str}
        self._adapter._record_start(str(run_id), _parent(parent_run_id), tool, args)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._adapter._record_end(str(run_id), output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._adapter._record_error(str(run_id), error)

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        _ = serialized, inputs, run_id, kwargs
        self._adapter._start_root(_parent(parent_run_id))

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._adapter._record_llm_start(
            str(run_id), _parent(parent_run_id), _model_from_serialized(serialized)
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # Chat models fire ``on_chat_model_start`` instead of ``on_llm_start``;
        # both terminate in ``on_llm_end``, so register the pending call here too.
        self._adapter._record_llm_start(
            str(run_id), _parent(parent_run_id), _model_from_serialized(serialized)
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._adapter._record_llm_end(str(run_id), response)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._adapter._record_llm_error(str(run_id))
