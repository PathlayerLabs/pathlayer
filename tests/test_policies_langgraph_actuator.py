"""Tests for LangGraphActuator (ACT-0): STOP -> RunControl.request_drain().

Skipped entirely unless the optional `langgraph` dependency is present.
Exercises the real `RunControl` (not a fake): `apply()`/`attach()`/`control`
are asserted against its actual `drain_requested`/`drain_reason` properties.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import SystemMessage
from langgraph.runtime import RunControl

from pathlayer.detectors.base import Signal
from pathlayer.events import EventKind
from pathlayer.policies.actions import Action
from pathlayer.policies.actuation import ActuationContext, EventSnapshot
from pathlayer.policies.langgraph_actuator import LangGraphActuator
from pathlayer.postmortem import PostMortem

# A stand-in for the SEC-3 leak vector: a RetryDetector builds its
# Signal.reason from the raw tool error, which can carry secrets. The injected
# warning must never surface it.
_SECRET_IN_REASON = "db timeout: password=hunter2"


def _post_mortem(reason: str = "loop: repeated sequence") -> PostMortem:
    return PostMortem(
        action=Action.STOP,
        reason=reason,
        stopped_at_index=3,
        events_processed=4,
        signals=(),
        recent_history=(),
    )


def _context(
    action: Action,
    *,
    signal_name: str = "no_progress",
    signal_reason: str = _SECRET_IN_REASON,
    fused: float = 0.6,
    tool: str = "edit_file",
    index: int = 7,
    post_mortem: PostMortem | None = None,
) -> ActuationContext:
    # signal_reason defaults to a secret-bearing string so every WARN path is
    # exercised against the leak vector unless a test opts out (signal_name="").
    signals = (
        (Signal(name=signal_name, confidence=fused, reason=signal_reason),) if signal_name else ()
    )
    return ActuationContext(
        action=action,
        fused=fused,
        reason=signal_reason,
        signals=signals,
        event=EventSnapshot(tool=tool, kind=EventKind.TOOL_CALL, ts=1.0, index=index, cost_usd=0.0),
        post_mortem=post_mortem,
    )


def test_fresh_control_is_not_drained() -> None:
    actuator = LangGraphActuator()
    assert isinstance(actuator.control, RunControl)
    assert actuator.control.drain_requested is False


def test_apply_stop_drains_with_post_mortem_reason() -> None:
    actuator = LangGraphActuator()
    actuator.apply(Action.STOP, _post_mortem("loop: repeated sequence"))
    assert actuator.control.drain_requested is True
    assert actuator.control.drain_reason == "loop: repeated sequence"


def test_apply_stop_without_post_mortem_uses_default_reason() -> None:
    actuator = LangGraphActuator()
    actuator.apply(Action.STOP, None)
    assert actuator.control.drain_requested is True
    assert actuator.control.drain_reason == "pathlayer_stop"


@pytest.mark.parametrize(
    "action",
    [Action.NONE, Action.WARN, Action.SUMMARIZE, Action.FALLBACK, Action.ESCALATE],
)
def test_apply_non_stop_tiers_never_drain(action: Action) -> None:
    actuator = LangGraphActuator()
    actuator.apply(action, None)
    assert actuator.control.drain_requested is False


def test_repeated_stop_apply_is_idempotent() -> None:
    # Mirrors what wire() itself already dedups by transition, but the
    # actuator must also tolerate being called more than once directly
    # (LangGraph's cooperative drain can still emit events between the drain
    # request and the actual halt).
    actuator = LangGraphActuator()
    actuator.apply(Action.STOP, _post_mortem("first"))
    actuator.apply(Action.STOP, _post_mortem("second"))
    assert actuator.control.drain_requested is True
    assert actuator.control.drain_reason == "second"


def test_attach_creates_a_fresh_control_distinct_from_previous() -> None:
    actuator = LangGraphActuator()
    first_control = actuator.control
    actuator.apply(Action.STOP, None)
    assert first_control.drain_requested is True

    actuator.attach()
    second_control = actuator.control
    assert second_control is not first_control
    assert second_control.drain_requested is False


def test_concurrent_apply_and_attach_do_not_raise() -> None:
    actuator = LangGraphActuator()
    errors: list[BaseException] = []

    def hammer_apply() -> None:
        for _ in range(200):
            try:
                actuator.apply(Action.STOP, None)
            except BaseException as exc:  # pragma: no cover - failure path only
                errors.append(exc)

    def hammer_attach() -> None:
        for _ in range(200):
            try:
                actuator.attach()
            except BaseException as exc:  # pragma: no cover - failure path only
                errors.append(exc)

    threads = [
        threading.Thread(target=hammer_apply),
        threading.Thread(target=hammer_attach),
        threading.Thread(target=hammer_apply),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# --------------------------------------------------------------------------- #
# ACT-1: WARN buffers a non-sensitive advisory; injection_node delivers it.
# --------------------------------------------------------------------------- #


def test_warn_buffers_a_warning_and_never_drains() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.WARN, signal_name="no_progress"))
    # A WARN must not touch the drain control (the run keeps going).
    assert actuator.control.drain_requested is False
    texts = actuator.drain_warnings()
    assert len(texts) == 1
    # The advisory names the dominant *signal* (a safe detector token).
    assert "no_progress" in texts[0]
    assert "[pathlayer]" in texts[0]
    # Drain is destructive: a second drain is empty.
    assert actuator.drain_warnings() == []


def test_warning_text_excludes_sensitive_fields() -> None:
    # SEC-3: the advisory is built from safe fields only (dominant signal
    # NAME, fused, tool, step). It must never surface Signal.reason, which a
    # RetryDetector fills with the raw tool error (secrets/PII).
    actuator = LangGraphActuator()
    actuator.apply_with_context(
        _context(
            Action.WARN, signal_name="retry", signal_reason=_SECRET_IN_REASON, index=7, fused=0.61
        )
    )
    (text,) = actuator.drain_warnings()
    # Safe fields are present...
    assert "retry" in text
    assert "edit_file" in text
    assert "step 7" in text
    assert "0.61" in text
    # ...and the secret-bearing reason is absent.
    assert _SECRET_IN_REASON not in text
    assert "hunter2" not in text


def test_warning_falls_back_to_tier_name_without_signals() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.WARN, signal_name=""))
    (text,) = actuator.drain_warnings()
    assert "warn" in text


def test_injection_node_returns_system_messages_then_clears() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.WARN, signal_name="loop"))
    node = actuator.injection_node()

    update = node({"messages": []})
    messages = update["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert "loop" in messages[0].content
    # Buffer drained: a second call is a no-op update (nothing to inject).
    assert node({"messages": []}) == {}


def test_injection_node_is_noop_without_warnings() -> None:
    actuator = LangGraphActuator()
    assert actuator.injection_node()({"messages": []}) == {}


def test_injection_node_honours_custom_messages_key() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.WARN))
    update = actuator.injection_node(messages_key="llm_input_messages")({})
    assert set(update) == {"llm_input_messages"}


def test_apply_with_context_stop_still_drains() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.STOP, post_mortem=_post_mortem("budget: ceiling")))
    assert actuator.control.drain_requested is True
    assert actuator.control.drain_reason == "budget: ceiling"
    # STOP does not enqueue a warning.
    assert actuator.drain_warnings() == []


@pytest.mark.parametrize("action", [Action.FALLBACK, Action.ESCALATE])
def test_apply_with_context_reserved_tiers_are_noops(action: Action) -> None:
    # SUMMARIZE is handled by ACT-2 (below); FALLBACK/ESCALATE stay reserved.
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(action))
    assert actuator.control.drain_requested is False
    assert actuator.drain_warnings() == []
    assert actuator.summary_pending is False


def test_attach_clears_buffered_warnings() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.WARN))
    actuator.attach()
    assert actuator.drain_warnings() == []


def test_warning_buffer_is_bounded() -> None:
    # More WARNs than the cap: the buffer keeps only the most recent ones rather
    # than growing without bound if the injection node is never wired.
    actuator = LangGraphActuator()
    for i in range(200):
        actuator.apply_with_context(_context(Action.WARN, index=i))
    texts = actuator.drain_warnings()
    assert len(texts) == 64
    # The retained window is the most recent (oldest dropped).
    assert "step 199" in texts[-1]


def test_concurrent_warn_and_drain_do_not_raise() -> None:
    actuator = LangGraphActuator()
    errors: list[BaseException] = []

    def buffer_warnings() -> None:
        for _ in range(500):
            try:
                actuator.apply_with_context(_context(Action.WARN))
            except BaseException as exc:  # pragma: no cover - failure path only
                errors.append(exc)

    def drain() -> None:
        for _ in range(500):
            try:
                actuator.drain_warnings()
            except BaseException as exc:  # pragma: no cover - failure path only
                errors.append(exc)

    threads = [
        threading.Thread(target=buffer_warnings),
        threading.Thread(target=drain),
        threading.Thread(target=buffer_warnings),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# --------------------------------------------------------------------------- #
# ACT-2: SUMMARIZE arms a strict gate that forces a structured summary before
# any other tool runs.
# --------------------------------------------------------------------------- #

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from pathlayer.events import PROGRESS_SUMMARY_TOOL, SUMMARY_BLOCKED_TOOL, ToolEvent  # noqa: E402
from pathlayer.policies.langgraph_actuator import MODEL_ROUTE, TOOLS_ROUTE  # noqa: E402


def _tool_call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _summary_args(
    completed: str = "wrote a guard clause",
    blocker: str = "the parser still fails on empty input",
    next_action: str = "add a regression test",
) -> dict[str, object]:
    return {"completed": completed, "blocker": blocker, "next_action": next_action}


def _discard_event(_event: ToolEvent) -> None:
    """No-op sink for gate tests that do not inspect refusal evidence."""


def test_summarize_arms_gate_and_buffers_instruction() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE, signal_name="no_progress"))
    # The gate is armed and never touches the drain control (SUMMARIZE never halts).
    assert actuator.summary_pending is True
    assert actuator.control.drain_requested is False
    # A safe instruction was buffered for injection_node to deliver.
    (text,) = actuator.drain_warnings()
    assert PROGRESS_SUMMARY_TOOL in text
    assert "no_progress" in text


def test_summary_instruction_excludes_sensitive_fields() -> None:
    # Same SEC-3 contract as WARN: the instruction names the signal but
    # never leaks Signal.reason (raw tool error -> secrets/PII).
    actuator = LangGraphActuator()
    actuator.apply_with_context(
        _context(Action.SUMMARIZE, signal_name="retry", signal_reason=_SECRET_IN_REASON)
    )
    (text,) = actuator.drain_warnings()
    assert "retry" in text
    assert _SECRET_IN_REASON not in text
    assert "hunter2" not in text


def test_attach_disarms_the_gate() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    assert actuator.summary_pending is True
    actuator.attach()
    assert actuator.summary_pending is False


def test_gate_is_passthrough_when_disarmed() -> None:
    actuator = LangGraphActuator()
    gate = actuator.summary_gate_node(_discard_event)
    state = {"messages": [AIMessage(content="", tool_calls=[_tool_call("search", {}, "c1")])]}
    assert gate(state) == {}


def test_gate_blocks_a_normal_tool_and_answers_it() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    refusal_events: list[ToolEvent] = []
    gate = actuator.summary_gate_node(refusal_events.append)
    ai = AIMessage(content="", tool_calls=[_tool_call("search", {"q": "x"}, "c1")])
    update = gate({"messages": [HumanMessage(content="go"), ai]})
    # The blocked call is answered with a ToolMessage; the gate stays armed.
    messages = update["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == "c1"
    assert PROGRESS_SUMMARY_TOOL in messages[0].content
    assert actuator.summary_pending is True
    assert len(refusal_events) == 1
    assert refusal_events[0].tool == SUMMARY_BLOCKED_TOOL
    assert refusal_events[0].kind is EventKind.TOOL_CALL
    assert refusal_events[0].args == {}
    assert refusal_events[0].result == {"error": "progress summary required"}
    # summary_route sends the (fully-answered) turn back to the model.
    route = actuator.summary_route()
    assert route({"messages": [ai, messages[0]]}) == MODEL_ROUTE


def test_gate_lets_a_valid_summary_through_and_lifts() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    gate = actuator.summary_gate_node(_discard_event)
    summary = actuator.summary_tool()
    ai = AIMessage(
        content="", tool_calls=[_tool_call(PROGRESS_SUMMARY_TOOL, _summary_args(), "c1")]
    )
    # A lone valid summary is left untouched (returns {}), so the tools node runs it.
    assert gate({"messages": [ai]}) == {}
    # Validation alone never lifts the gate: the tool must actually execute.
    assert actuator.summary_pending is True
    # And it routes to the tools node (unanswered summary call still pending).
    assert actuator.summary_route()({"messages": [ai]}) == TOOLS_ROUTE
    summary.invoke(_summary_args())
    assert actuator.summary_pending is False


def test_gate_keeps_only_the_summary_when_emitted_with_other_tools() -> None:
    # Parallel tool calls in one AIMessage: only the summary passes; the sibling
    # is dropped (re-issuable next turn), never executed.
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    gate = actuator.summary_gate_node(_discard_event)
    ai = AIMessage(
        content="",
        tool_calls=[
            _tool_call("search", {"q": "x"}, "c1"),
            _tool_call(PROGRESS_SUMMARY_TOOL, _summary_args(), "c2"),
        ],
    )
    update = gate({"messages": [ai]})
    (kept,) = update["messages"]
    assert isinstance(kept, AIMessage)
    assert [c["name"] for c in kept.tool_calls] == [PROGRESS_SUMMARY_TOOL]
    assert kept.id == ai.id  # same message, rewritten in place (add_messages replace)
    assert actuator.summary_pending is True
    actuator.summary_tool().invoke(_summary_args())
    assert actuator.summary_pending is False


def test_gate_keeps_blocking_on_an_invalid_summary() -> None:
    # A summary attempt missing a field does NOT lift the gate: it is blocked
    # like any other call and the gate stays armed.
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    gate = actuator.summary_gate_node(_discard_event)
    bad = _summary_args()
    del bad["next_action"]
    ai = AIMessage(content="", tool_calls=[_tool_call(PROGRESS_SUMMARY_TOOL, bad, "c1")])
    update = gate({"messages": [ai]})
    (blocked,) = update["messages"]
    assert isinstance(blocked, ToolMessage)
    assert actuator.summary_pending is True


@pytest.mark.parametrize(
    ("state", "expected_pending"),
    [
        ({}, False),
        ({"messages": []}, True),
        ({"messages": [HumanMessage(content="hi")]}, True),
        ({"messages": [AIMessage(content="done, no tools")]}, True),
        ({"other_key": 123}, False),
        ("not a mapping", False),
    ],
)
def test_gate_is_fail_open_on_unexpected_state(state: object, expected_pending: bool) -> None:
    # Defensive by construction: any anomaly degrades to "let the agent proceed"
    # ({}) and never raises inside the agent graph.
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    gate = actuator.summary_gate_node(_discard_event)
    assert gate(state) == {}  # type: ignore[arg-type]
    assert actuator.summary_pending is expected_pending


@pytest.mark.parametrize(
    "state",
    [
        "not a mapping",
        {"messages": []},
        {"messages": [HumanMessage(content="hi")]},
    ],
)
def test_summary_route_disarms_on_uninterpretable_state(state: object) -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    route = actuator.summary_route()
    assert route(state) == MODEL_ROUTE  # type: ignore[arg-type]
    assert actuator.summary_pending is False


def test_gate_ignores_tool_calls_without_an_id() -> None:
    # A tool call carrying no id cannot be answered with a matching ToolMessage,
    # so it is skipped: the gate produces no update and stays armed rather than
    # emitting an orphan block message.
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    gate = actuator.summary_gate_node(_discard_event)
    ai = AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": None}])
    assert gate({"messages": [ai]}) == {}
    assert actuator.summary_pending is True
    # And the route must NOT let that id-less call slip through to the tools node:
    # while the gate is armed with no valid summary, it goes back to the model.
    # (Without consulting the armed flag it would look "unanswered" and run,
    # bypassing the strict gate.)
    assert actuator.summary_route()({"messages": [ai]}) == MODEL_ROUTE


def test_route_sends_armed_gate_without_valid_summary_back_to_model() -> None:
    # Even a normal id-bearing call that has not yet been answered routes to the
    # model while the gate is armed and no valid summary is present: the strict
    # gate never runs a tool on an armed, unsummarized turn.
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    ai = AIMessage(content="", tool_calls=[_tool_call("search", {"q": "x"}, "c1")])
    assert actuator.summary_route()({"messages": [ai]}) == MODEL_ROUTE
    # Disarmed, the same unanswered call routes to the tools node as usual.
    actuator.attach()
    assert actuator.summary_route()({"messages": [ai]}) == TOOLS_ROUTE


def test_route_ignores_old_answer_when_tool_call_id_is_reused() -> None:
    # Only ToolMessages after the most recent AIMessage answer that turn. An old
    # response with the same id must not make a new call look already executed.
    actuator = LangGraphActuator()
    old_ai = AIMessage(content="", tool_calls=[_tool_call("search", {}, "reused")])
    old_answer = ToolMessage(content="old result", tool_call_id="reused")
    new_ai = AIMessage(content="", tool_calls=[_tool_call("search", {}, "reused")])
    assert actuator.summary_route()({"messages": [old_ai, old_answer, new_ai]}) == TOOLS_ROUTE


def test_gate_fails_open_if_blocked_event_sink_raises() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    ai = AIMessage(content="", tool_calls=[_tool_call("search", {}, "c1")])

    def broken_sink(_event: ToolEvent) -> None:
        raise RuntimeError("pipeline unavailable")

    assert actuator.summary_gate_node(broken_sink)({"messages": [ai]}) == {}
    assert actuator.summary_pending is False
    assert actuator.summary_route()({"messages": [ai]}) == TOOLS_ROUTE


def test_gate_and_route_fail_open_if_an_internal_helper_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both run inside the agent graph: if any internal step raises unexpectedly,
    # they must degrade to "let the agent proceed", never propagate.
    import pathlayer.policies.langgraph_actuator as mod

    def _boom(_messages: object) -> object:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(mod, "_last_ai_message", _boom)
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    ai = AIMessage(content="", tool_calls=[_tool_call("search", {}, "c1")])
    assert actuator.summary_gate_node(_discard_event)({"messages": [ai]}) == {}
    assert actuator.summary_pending is False
    assert actuator.summary_route()({"messages": [ai]}) == TOOLS_ROUTE


def test_summary_tool_has_the_stable_name_and_lifts_only_when_executed() -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(Action.SUMMARIZE))
    summary = actuator.summary_tool()
    assert summary.name == PROGRESS_SUMMARY_TOOL
    assert actuator.summary_pending is True
    result = summary.invoke(_summary_args())
    assert isinstance(result, str)
    assert actuator.summary_pending is False


def test_gate_concurrent_arm_and_evaluate_do_not_raise() -> None:
    actuator = LangGraphActuator()
    gate = actuator.summary_gate_node(_discard_event)
    ai = AIMessage(content="", tool_calls=[_tool_call("search", {}, "c1")])
    errors: list[BaseException] = []

    def arm() -> None:
        for _ in range(300):
            try:
                actuator.apply_with_context(_context(Action.SUMMARIZE))
            except BaseException as exc:  # pragma: no cover - failure path only
                errors.append(exc)

    def evaluate() -> None:
        for _ in range(300):
            try:
                gate({"messages": [ai]})
            except BaseException as exc:  # pragma: no cover - failure path only
                errors.append(exc)

    threads = [
        threading.Thread(target=arm),
        threading.Thread(target=evaluate),
        threading.Thread(target=evaluate),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
