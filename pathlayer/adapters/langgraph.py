"""LangGraph / LangChain capture adapter (ADP-0, first framework adapter).

Captures tool calls from a LangGraph/LangChain run into normalized
:class:`~pathlayer.events.ToolEvent`\\ s, via a passive
``BaseCallbackHandler``. It never alters the agent flow (observer pattern): it
only reads callback payloads and pushes events to the sink.

Scope (V1): tool-call capture for both ``invoke`` and ``ainvoke``, with **raw
lineage** -- ``parent_run_id`` is stored in ``metadata`` so delegation depth can
be reconstructed offline later. We do *not* classify ``SUBAGENT_SPAWN`` (a
name-based heuristic would be noisy); lifecycle events are out of scope here.

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

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from pathlayer.adapters.base import Adapter, EventSink
from pathlayer.events import ToolEvent

# Bound on un-paired ``on_tool_start`` calls held in ``_pending``. A start that
# is never followed by an end/error (cancelled coroutine, interrupted run, a
# dropped end callback) would otherwise leak forever on a long-running agent
# attached once. We evict the oldest entry FIFO past this cap so memory stays
# bounded; the nominal start->end path is unaffected (it pops its own entry).
_MAX_PENDING = 1024

# Top-level coercion only: scalars/containers are emitted as-is, anything else is
# stringified. This handles the common case where a tool returns a non-serializable
# object directly (e.g. a LangChain ``ToolMessage``) so a VAL-1 round-trip
# (dump_trace/load) survives it. It is intentionally shallow (per the ADP-0 plan):
# a dict/list containing a nested non-serializable value would still fail
# ``dump_trace``. Deep coercion is out of scope here.
_JSON_FRIENDLY = (bool, int, float, str, dict, list)


def _coerce_result(output: Any) -> Any:
    """Coerce a tool output to a JSON-friendly value at the top level.

    Scalars/dict/list are kept as-is; anything else is stringified. Shallow by
    design: nested non-serializable values inside a dict/list are not coerced.
    """
    if output is None or isinstance(output, _JSON_FRIENDLY):
        return output
    return str(output)


@dataclass(slots=True)
class _Pending:
    """A tool call seen at ``on_tool_start``, awaiting its end/error."""

    tool: str
    args: dict[str, Any]
    parent_run_id: str | None


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

    def __init__(self, sink: EventSink) -> None:
        super().__init__(sink)
        self._handler = _LangChainCaptureHandler(self)
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, _Pending] = OrderedDict()
        self._ts = 0

    def attach(self) -> None:
        """Reset capture state (deterministic ``ts`` per attached run)."""
        with self._lock:
            self._pending.clear()
            self._ts = 0

    def detach(self) -> None:
        """Drop any unpaired pending calls; the handler keeps no other state."""
        with self._lock:
            self._pending.clear()

    def as_config(self) -> dict[str, Any]:
        """Return a LangChain ``RunnableConfig`` wiring our capture handler."""
        return {"callbacks": [self._handler]}

    def _record_start(
        self, run_id: str, parent_run_id: str | None, tool: str, args: dict[str, Any]
    ) -> None:
        with self._lock:
            self._pending[run_id] = _Pending(tool=tool, args=args, parent_run_id=parent_run_id)
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
            self.emit(
                ToolEvent(
                    tool=pending.tool,
                    args=pending.args,
                    result=result,
                    ts=ts,
                    run_id=run_id,
                    metadata=metadata,
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
