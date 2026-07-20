"""Tests for LangGraphActuator (ACT-0): STOP -> RunControl.request_drain().

Skipped entirely unless the optional `langgraph` dependency is present.
Exercises the real `RunControl` (not a fake): `apply()`/`attach()`/`control`
are asserted against its actual `drain_requested`/`drain_reason` properties.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("langgraph")

from langgraph.runtime import RunControl

from pathlayer.policies.actions import Action
from pathlayer.policies.langgraph_actuator import LangGraphActuator
from pathlayer.postmortem import PostMortem


def _post_mortem(reason: str = "loop: repeated sequence") -> PostMortem:
    return PostMortem(
        action=Action.STOP,
        reason=reason,
        stopped_at_index=3,
        events_processed=4,
        signals=(),
        recent_history=(),
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
