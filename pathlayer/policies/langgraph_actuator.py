"""LangGraph actuator (ACT-0/ACT-1/ACT-2): drain on STOP, inject on WARN,
gate a forced summary on SUMMARIZE.

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
it does not reach into the agent). Instead it holds a thread-safe injection buffer
that :meth:`apply_with_context` fills on a WARN transition, and exposes
:meth:`injection_node` -- a plain callable the integrator wires into their graph
(as a node, or as a ``pre_model_hook`` on a prebuilt agent) that drains the
buffer and returns the messages as ``SystemMessage``\\ s. The actuator still
never decides anything: the WARN tier is already arbitrated by ``Policy`` and
arrives in the :class:`~pathlayer.policies.actuation.ActuationContext`.

SUMMARIZE (ACT-2) forces the agent to emit a *structured* progress summary
before any further tool runs -- a **strict gate**. On a SUMMARIZE transition the
actuator (a) arms a thread-safe ``summary_pending`` flag and (b) buffers a safe
instruction (delivered by the same :meth:`injection_node` as WARN) telling the
agent to call the Pathlayer summary tool. Enforcement is a plain pre-tools graph
node, :meth:`summary_gate_node`, wired between the model and the tools node: while
armed, it answers every non-summary tool call with a "blocked, summarize first"
``ToolMessage`` and routes back to the model (via :meth:`summary_route`) instead
of executing it; a *valid* summary (all of ``completed``/``blocker``/
``next_action`` present, validated by
:class:`~pathlayer.policies.actuation.ProgressSummary`) is the only thing that
lifts the flag and is let through to run (and be captured). Two guardrails from
the design audit are load-bearing:

- **STOP stays the only hard termination.** The gate re-prompts, it never ends
  the run: a valid summary lifts it, and while it is closed *no tool executes*,
  so there is no tool-side runaway. Two exit paths keep it from being the sole
  exit: every rejected tool turn emits one safe synthetic
  ``pathlayer_summary_blocked`` ``TOOL_CALL`` through the same
  :class:`~pathlayer.adapters.base.EventSink` as capture. That deterministic
  evidence keeps Loop/Retry/NoProgress and Policy advancing until STOP drains
  a persistently refusing run. The gate never decides to stop by itself.
- **Fail-open inside the agent graph.** :meth:`summary_gate_node` /
  :meth:`summary_route` run *inside* the agent's own graph, outside the
  fail-open ``try`` in :func:`~pathlayer.policies.wiring.wire`. They are therefore
  defensive by construction: any internal anomaly (unexpected state shape,
  malformed message) degrades to "let the agent proceed" and never raises into
  the run, exactly as :meth:`injection_node` returns ``{}`` on an empty buffer.

FALLBACK/ESCALATE remain no-ops here (see ``ACT-3..4``, #31-32).

This module imports ``langgraph``/``langchain_core`` and so is an **optional**
import, like :mod:`pathlayer.adapters.langgraph`: install the ``langgraph`` extra
(``pip install pathlayer[langgraph]``). It is deliberately *not* re-exported from
:mod:`pathlayer.policies`.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.runtime import RunControl

from pathlayer.adapters.base import EventSink
from pathlayer.events import PROGRESS_SUMMARY_TOOL, SUMMARY_BLOCKED_TOOL, ToolEvent
from pathlayer.policies.actions import Action
from pathlayer.policies.actuation import ActuationContext, ProgressSummary
from pathlayer.policies.actuators import Actuator
from pathlayer.postmortem import PostMortem

_DEFAULT_STOP_REASON = "pathlayer_stop"

# Route labels returned by :meth:`LangGraphActuator.summary_route`. Public
# constants so an integrator maps its conditional edge without hard-coding the
# strings (``{TOOLS_ROUTE: "tools", MODEL_ROUTE: "agent"}``).
TOOLS_ROUTE = "tools"
MODEL_ROUTE = "model"

# Fixed, safe text answered to a tool call blocked by the summary gate. Carries
# no signal reason / args / result -- only the standing instruction.
_SUMMARY_BLOCK_INSTRUCTION = (
    f"[pathlayer] Blocked: emit a progress summary before any other tool runs. "
    f"Call {PROGRESS_SUMMARY_TOOL}(completed, blocker, next_action) now, then continue."
)

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
        sink = wire(pipeline, actuator)
        adapter = LangGraphAdapter(sink)
        adapter.attach()
        builder.add_node("pathlayer_warn", actuator.injection_node())
        builder.add_node("pathlayer_summary_gate", actuator.summary_gate_node(sink))
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
        # ACT-2 strict gate: armed on a SUMMARIZE transition, lifted only by a
        # valid summary passing through summary_gate_node. Guarded by the same
        # lock as the buffer/control (see the class docstring on threading).
        self._summary_pending = False

    @property
    def control(self) -> RunControl:
        """The current run's ``RunControl``; pass as ``control=`` to ``invoke``."""
        with self._lock:
            return self._control

    @property
    def summary_pending(self) -> bool:
        """True while the ACT-2 gate is armed (a valid summary is still owed)."""
        with self._lock:
            return self._summary_pending

    def _disarm_summary_gate(self) -> None:
        """Lift the gate atomically after success or to fail open on an anomaly."""
        with self._lock:
            self._summary_pending = False

    def attach(self) -> None:
        """Start a fresh run: new ``RunControl``, empty buffer, disarmed gate.

        Call before ``adapter.attach()``.
        """
        with self._lock:
            self._control = RunControl()
            self._warnings.clear()
            self._summary_pending = False

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        if action is not Action.STOP:
            return
        reason = post_mortem.reason if post_mortem is not None else _DEFAULT_STOP_REASON
        with self._lock:
            self._control.request_drain(reason)

    def apply_with_context(self, context: ActuationContext) -> None:
        """Enforce a graded action from its full context (ACT-1/ACT-2).

        STOP delegates to :meth:`apply` (unchanged ACT-0 drain); WARN buffers a
        concise advisory for :meth:`injection_node` to deliver; SUMMARIZE arms
        the strict gate and buffers a safe instruction to call the summary tool;
        FALLBACK/ESCALATE stay no-ops (``ACT-3..4``).
        """
        if context.action is Action.STOP:
            self.apply(Action.STOP, context.post_mortem)
        elif context.action is Action.WARN:
            with self._lock:
                self._warnings.append(_format_warning(context))
        elif context.action is Action.SUMMARIZE:
            with self._lock:
                self._summary_pending = True
                self._warnings.append(_format_summary_instruction(context))

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

    def summary_tool(self) -> BaseTool:
        """Return the Pathlayer structured-summary tool (ACT-2).

        A plain LangChain tool named ``PROGRESS_SUMMARY_TOOL`` taking exactly
        ``completed``/``blocker``/``next_action``. Add it to the agent's tool set
        so, while the gate is armed, the model has a sanctioned way to comply.
        Executing the tool atomically lifts the gate. Validation and routing
        still happen in :meth:`summary_gate_node`, but the flag is deliberately
        not cleared until the tool function actually runs: an unregistered or
        failed tool must not satisfy the requirement merely because its call
        looked syntactically valid. The adapter captures its execution as
        ``EventKind.PROGRESS_SUMMARY`` (which the detectors ignore).
        """

        @tool(PROGRESS_SUMMARY_TOOL)
        def pathlayer_progress_summary(completed: str, blocker: str, next_action: str) -> str:
            """Record a structured progress summary: what you have completed, what
            is blocking you, and your single concrete next action. Call this when
            Pathlayer asks for a checkpoint, before running any other tool."""
            self._disarm_summary_gate()
            return "progress summary recorded"

        return pathlayer_progress_summary

    def summary_gate_node(
        self, on_blocked: EventSink, messages_key: str = "messages"
    ) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        """Return a pre-tools graph node enforcing the ACT-2 strict gate.

        ``on_blocked`` must be the same serialized sink used by the capture
        adapter (normally ``wire(pipeline, actuator)``). Wire the returned node
        between the model and the tools node. When the gate is disarmed it is a
        pass-through (returns ``{}``). When armed it inspects the most recent
        tool-calling ``AIMessage``:

        - A **valid** summary call (see
          :class:`~pathlayer.policies.actuation.ProgressSummary`) is the only call
          kept on the message -- any sibling tool calls in the same turn are
          dropped -- so the tools node runs *only* the summary. The gate remains
          armed until :meth:`summary_tool` executes successfully.
        - Otherwise every tool call is answered with a fixed "blocked, summarize
          first" :class:`~langchain_core.messages.ToolMessage`, and one safe
          synthetic ``pathlayer_summary_blocked`` event is sent to ``on_blocked``
          so the existing detectors/policy can escalate persistent refusal to
          STOP. The calls stay on the ``AIMessage`` so there is no orphan tool
          response, and
          :meth:`summary_route` sends the graph back to the model. Nothing
          executes, but refusal remains observable rather than looping forever.

        Defensive by construction: it never raises into the run (any anomaly
        disarms the gate and returns ``{}`` -- let the agent proceed), because it
        executes inside the agent graph, outside
        :func:`~pathlayer.policies.wiring.wire`'s fail-open guard.
        """

        def _gate(state: Mapping[str, Any]) -> dict[str, Any]:
            try:
                with self._lock:
                    pending = self._summary_pending
                if not pending:
                    return {}
                messages = _messages_of(state, messages_key)
                if messages is None:
                    self._disarm_summary_gate()
                    return {}
                if not messages:
                    return {}
                last_ai = _last_ai_message(messages)
                if last_ai is None or not last_ai.tool_calls:
                    return {}
                valid_summary = _first_valid_summary_call(last_ai.tool_calls)
                if valid_summary is not None:
                    if len(last_ai.tool_calls) == 1:
                        # Summary is the only call: let it run untouched.
                        return {}
                    kept = last_ai.model_copy(update={"tool_calls": [valid_summary]})
                    return {messages_key: [kept]}
                # No valid summary: keep the refusal observable so the existing
                # detectors/policy can escalate to STOP. This event contains no
                # model/tool payload and is deterministic for every refusal.
                on_blocked(
                    ToolEvent(
                        tool=SUMMARY_BLOCKED_TOOL,
                        result={"error": "progress summary required"},
                    )
                )
                blocked = [
                    ToolMessage(
                        content=_SUMMARY_BLOCK_INSTRUCTION,
                        tool_call_id=_tool_call_id(call),
                        name=_tool_call_name(call) or PROGRESS_SUMMARY_TOOL,
                    )
                    for call in last_ai.tool_calls
                    if _tool_call_id(call)
                ]
                if not blocked:
                    return {}
                return {messages_key: blocked}
            except Exception:
                # Fail-open: never trap the agent on an internal anomaly. Lifting
                # here makes summary_route use normal tool routing for this turn.
                self._disarm_summary_gate()
                return {}

        return _gate

    def summary_route(
        self,
        messages_key: str = "messages",
        *,
        tools: str = TOOLS_ROUTE,
        model: str = MODEL_ROUTE,
    ) -> Callable[[Mapping[str, Any]], str]:
        """Return a conditional-edge function to place *after* the gate node.

        Returns ``tools`` when the most recent tool-calling ``AIMessage`` still
        has an *unanswered* tool call to run (the summary let through), else
        ``model`` (every call already answered -- the blocked path -- so loop
        back for another turn). This is a summary-aware ``tools_condition``:
        unlike the stock one it also routes back to the model when the calls are
        present but already answered by blocking ``ToolMessage``\\ s.

        It also consults the armed flag directly: while the gate is armed and the
        turn carries no *valid* summary, it always routes to ``model`` -- even if
        an unanswered tool call remains. That closes the one path the block loop
        cannot cover, a tool call with no ``id`` (which :meth:`summary_gate_node`
        cannot answer with a ``ToolMessage``): without this it would look
        "unanswered" and run, bypassing the strict gate. Fail-open: an
        uninterpretable/no-tool state disarms the gate before returning to the
        model, while an exception during routing a known tool turn routes to
        ``tools``. Neither path can remain trapped in the gate.
        """

        def _route(state: Mapping[str, Any]) -> str:
            try:
                messages = _messages_of(state, messages_key)
                if not messages:
                    self._disarm_summary_gate()
                    return model
                last_ai = _last_ai_message(messages)
                if last_ai is None or not last_ai.tool_calls:
                    self._disarm_summary_gate()
                    return model
                last_ai_index = next(
                    index
                    for index in range(len(messages) - 1, -1, -1)
                    if messages[index] is last_ai
                )
                with self._lock:
                    pending = self._summary_pending
                if pending and _first_valid_summary_call(last_ai.tool_calls) is None:
                    # Gate armed, no valid summary this turn: never run a tool.
                    # Covers id-less calls the gate cannot answer -- go back to
                    # the model instead of letting them slip through.
                    return model
                answered = {
                    m.tool_call_id
                    for m in messages[last_ai_index + 1 :]
                    if isinstance(m, ToolMessage) and m.tool_call_id
                }
                has_unanswered = any(
                    _tool_call_id(call) not in answered for call in last_ai.tool_calls
                )
                return tools if has_unanswered else model
            except Exception:
                return tools

        return _route


def _messages_of(state: Mapping[str, Any], messages_key: str) -> Sequence[BaseMessage] | None:
    """Extract the messages list from a graph state, defensively (may be None)."""
    if not isinstance(state, Mapping):
        return None
    messages = state.get(messages_key)
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return None
    return messages


def _last_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    """The most recent ``AIMessage`` in ``messages``, or ``None``."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _tool_call_name(call: Any) -> str | None:
    """Tool name from a LangChain ``ToolCall`` (a dict at runtime); ``None`` if absent.

    A non-mapping ``call`` (never emitted by LangChain, but possible if a state
    is malformed) yields ``None`` so the gate treats it as a non-summary call --
    part of the fail-open posture.
    """
    name = call.get("name") if isinstance(call, Mapping) else None
    return name if isinstance(name, str) else None


def _tool_call_args(call: Any) -> Any:
    """Args mapping from a LangChain ``ToolCall`` (a dict at runtime), else ``None``."""
    return call.get("args") if isinstance(call, Mapping) else None


def _tool_call_id(call: Any) -> str | None:
    """Id from a LangChain ``ToolCall`` (a dict at runtime); ``None`` if absent."""
    call_id = call.get("id") if isinstance(call, Mapping) else None
    return call_id if isinstance(call_id, str) else None


def _first_valid_summary_call(tool_calls: Sequence[Any]) -> Any | None:
    """First tool call that is a *valid* Pathlayer summary, else ``None``.

    A call counts only when its name is ``PROGRESS_SUMMARY_TOOL`` **and** its
    args validate as a :class:`~pathlayer.policies.actuation.ProgressSummary`
    (all three fields present, non-blank). A summary attempt with a missing
    field does not lift the gate -- exactly the "failed validation keeps the
    gate closed" rule.
    """
    for call in tool_calls:
        if _tool_call_name(call) != PROGRESS_SUMMARY_TOOL:
            continue
        if ProgressSummary.from_tool_args(_tool_call_args(call)) is not None:
            return call
    return None


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


def _format_summary_instruction(context: ActuationContext) -> str:
    """Render the safe ACT-2 instruction to emit a structured summary.

    Same safety contract as :func:`_format_warning`: only the dominant signal
    *name*, the fused score and the event snapshot (tool / step) -- never
    ``context.reason``/``Signal.reason``/``args``/``result``/``metadata`` (the
    SEC-3 #133 leak vector).
    """
    event = context.event
    return (
        f"[pathlayer] Progress checkpoint required: possible "
        f"{_dominant_signal_name(context)} (confidence {context.fused:.2f}) at step "
        f"{event.index} after tool '{event.tool}'. Before any other tool runs, call "
        f"{PROGRESS_SUMMARY_TOOL}(completed, blocker, next_action) with an honest, "
        "concrete summary of what you have completed, what is blocking you, and your "
        "single next action."
    )
