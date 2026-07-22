"""LangGraph actuator (ACT-0/ACT-1): drain on STOP, inject a warning on WARN.

STOP (ACT-0) uses LangGraph's native cooperative graceful-shutdown mechanism
(>= 1.2, ``langgraph.runtime.RunControl`` / ``langgraph.errors.GraphDrained``):
thread-safe (a single attribute write), checkpointed, no new network endpoint,
no new timeout-safety problem. This is why LangGraph is the first concrete
actuator (see the ACT-0 plan): unlike an OpenClaw ``before_tool_call`` actuator,
it needs neither PERF-2 (no synchronous blocking hook waiting on Pathlayer) nor
a new authenticated endpoint.

WARN (ACT-1) injects a concise, non-sensitive advisory into the agent's own
context **without interrupting the run**. Because a graph node -- not the
actuator -- owns the agent's state, the actuator cannot mutate ``messages``
directly and stay within the layering guardrail (an actuator executes a decision,
it does not reach into the agent). Instead it holds a thread-safe warning buffer
that :meth:`apply_with_context` fills on a WARN transition, and exposes
:meth:`injection_node` -- a plain callable the integrator wires into their graph
(as a node, or as a ``pre_model_hook`` on a prebuilt agent) that drains the
buffer and returns the warnings as ``SystemMessage``\\ s. The actuator still
never decides anything: the WARN tier is already arbitrated by ``Policy`` and
arrives in the :class:`~pathlayer.policies.actuation.ActuationContext`.

SUMMARIZE/FALLBACK/ESCALATE remain no-ops here (see ``ACT-2..4``, #30-32).

This module imports ``langgraph``/``langchain_core`` and so is an **optional**
import, like :mod:`pathlayer.adapters.langgraph`: install the ``langgraph`` extra
(``pip install pathlayer[langgraph]``). It is deliberately *not* re-exported from
:mod:`pathlayer.policies`.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.runtime import RunControl

from pathlayer.policies.actions import Action
from pathlayer.policies.actuation import ActuationContext
from pathlayer.policies.actuators import Actuator
from pathlayer.postmortem import PostMortem

_DEFAULT_STOP_REASON = "pathlayer_stop"

# Bound on buffered, never-drained warnings. wire() enqueues at most one per
# transition *into* WARN (sustained WARN is de-duplicated), so this only grows
# under pathological WARN/NONE/WARN oscillation while the injection node is not
# wired. We drop the oldest past the cap so memory stays bounded; a wired node
# drains after every tick and never approaches it.
_MAX_BUFFERED_WARNINGS = 64


class LangGraphActuator(Actuator):
    """STOP -> ``RunControl.request_drain()``; WARN -> buffered context injection.

    Lifecycle: one instance per graph run, mirroring
    :class:`~pathlayer.adapters.langgraph.LangGraphAdapter`. Call :meth:`attach`
    once before starting the run -- it creates a fresh ``RunControl`` (a drained
    ``RunControl`` cannot be reused, per its own docstring) and clears any
    leftover warnings -- then pass :attr:`control` to
    ``graph.invoke()``/``ainvoke()`` and wire :meth:`injection_node` into the
    graph so WARN advisories reach the model::

        actuator = LangGraphActuator()
        actuator.attach()
        adapter = LangGraphAdapter(wire(pipeline, actuator))
        adapter.attach()
        builder.add_node("pathlayer_warn", actuator.injection_node())
        # ... edge it in *before* the model node, e.g. START -> pathlayer_warn -> agent
        try:
            result = graph.invoke(state, config=adapter.as_config(), control=actuator.control)
        except GraphDrained as e:
            ...  # checkpoint saved; e.reason carries the drain reason
        finally:
            adapter.detach()

    ``GraphDrained`` (``from langgraph.errors import GraphDrained``) must be
    caught by the integrator's own code around ``invoke``/``ainvoke``: Pathlayer
    does not own that call.

    A single lock guards ``_control`` **and** the warning buffer because both
    :meth:`apply`/:meth:`apply_with_context` are invoked from
    ``LangGraphAdapter``'s lock-held sink callback (a worker thread during
    ``ainvoke``) while :meth:`attach`/:attr:`control`/:meth:`injection_node`'s
    drain may run on the graph's own execution thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._control = RunControl()
        self._warnings: deque[str] = deque(maxlen=_MAX_BUFFERED_WARNINGS)

    @property
    def control(self) -> RunControl:
        """The current run's ``RunControl``; pass as ``control=`` to ``invoke``."""
        with self._lock:
            return self._control

    def attach(self) -> None:
        """Start a fresh run: new ``RunControl`` + empty warning buffer.

        Call before ``adapter.attach()``.
        """
        with self._lock:
            self._control = RunControl()
            self._warnings.clear()

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        if action is not Action.STOP:
            return
        reason = post_mortem.reason if post_mortem is not None else _DEFAULT_STOP_REASON
        with self._lock:
            self._control.request_drain(reason)

    def apply_with_context(self, context: ActuationContext) -> None:
        """Enforce a graded action from its full context (ACT-1).

        STOP delegates to :meth:`apply` (unchanged ACT-0 drain); WARN buffers a
        concise advisory for :meth:`injection_node` to deliver; the other tiers
        are reserved for ``ACT-2..4``.
        """
        if context.action is Action.STOP:
            self.apply(Action.STOP, context.post_mortem)
        elif context.action is Action.WARN:
            with self._lock:
                self._warnings.append(_format_warning(context))

    def drain_warnings(self) -> list[str]:
        """Atomically remove and return all buffered warning texts (may be empty)."""
        with self._lock:
            texts = list(self._warnings)
            self._warnings.clear()
        return texts

    def injection_node(
        self, messages_key: str = "messages"
    ) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        """Return a graph node that injects buffered WARN advisories.

        The returned callable drains the warning buffer and, if non-empty,
        returns ``{messages_key: [SystemMessage(...), ...]}`` -- an *append*
        under the standard ``add_messages`` reducer, so it never overwrites the
        agent's own messages. When there is nothing to inject it returns ``{}``
        (a no-op state update), so it is safe to keep on the hot path every tick.
        It reads no state, only Pathlayer's own buffer, so it never depends on
        the agent's private state shape beyond the messages key.

        Works both as a standalone ``StateGraph`` node and as a prebuilt agent's
        ``pre_model_hook`` (which likewise merges a ``messages`` update).
        """

        def _inject(state: Mapping[str, Any]) -> dict[str, Any]:
            texts = self.drain_warnings()
            if not texts:
                return {}
            return {messages_key: [SystemMessage(content=text) for text in texts]}

        return _inject


def _dominant_signal_name(context: ActuationContext) -> str:
    """The name of the highest-confidence signal, or the tier name if none.

    A detector ``name`` is a fixed vocabulary token (``loop``/``retry``/
    ``no_progress``/``budget``/...), safe to surface to the agent -- unlike a
    ``Signal.reason``, which a detector such as ``RetryDetector`` builds from the
    raw tool error and can therefore carry secrets/PII (see ``_format_warning``).
    ``NaN`` confidences sort as ``-inf`` so a broken detector never wins.
    """
    if not context.signals:
        return context.action.name.lower()
    best = max(
        context.signals,
        key=lambda s: s.confidence if s.confidence == s.confidence else float("-inf"),
    )
    return best.name


def _format_warning(context: ActuationContext) -> str:
    """Render a concise, non-sensitive advisory from an actuation context.

    Uses only fields that are safe to inject into the agent's own prompt: the
    dominant signal *name* (a detector-vocabulary token, see
    :func:`_dominant_signal_name`), the fused score, and the tool name / step
    index from the :class:`~pathlayer.policies.actuation.EventSnapshot`. It never
    uses ``context.reason`` (a free-form ``Signal.reason`` can embed the raw tool
    error -- secrets/PII, the SEC-3 #133 leak vector) nor ``args``/``result``/
    ``metadata`` (excluded from the snapshot for the same reason).
    """
    event = context.event
    return (
        f"[pathlayer] Reliability warning: possible {_dominant_signal_name(context)} "
        f"(confidence {context.fused:.2f}) at step {event.index} "
        f"after tool '{event.tool}'. "
        "Reassess whether you are making real progress before continuing; "
        "avoid repeating the same unproductive actions."
    )
