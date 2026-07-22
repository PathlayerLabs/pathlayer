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

# A stand-in for the SEC-3 (#133) leak vector: a RetryDetector builds its
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
    # SEC-3 (#133): the advisory is built from safe fields only (dominant signal
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


@pytest.mark.parametrize("action", [Action.SUMMARIZE, Action.FALLBACK, Action.ESCALATE])
def test_apply_with_context_reserved_tiers_are_noops(action: Action) -> None:
    actuator = LangGraphActuator()
    actuator.apply_with_context(_context(action))
    assert actuator.control.drain_requested is False
    assert actuator.drain_warnings() == []


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
