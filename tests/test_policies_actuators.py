"""Tests for the Actuator contract (ACT-0): Actuator ABC + LoggingActuator.

No framework involved: this is the pure, framework-agnostic contract every
concrete actuator (LangGraph, future OpenClaw) implements.
"""

from __future__ import annotations

import logging

import pytest

from pathlayer.policies.actions import Action
from pathlayer.policies.actuators import Actuator, LoggingActuator
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


def test_actuator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Actuator()  # type: ignore[abstract]


def test_logging_actuator_logs_on_stop(caplog: pytest.LogCaptureFixture) -> None:
    actuator = LoggingActuator()
    with caplog.at_level(logging.WARNING):
        actuator.apply(Action.STOP, _post_mortem("loop: repeated sequence"))
    assert any("STOP" in record.message for record in caplog.records)
    assert any("loop: repeated sequence" in record.message for record in caplog.records)


def test_logging_actuator_falls_back_to_action_name_without_post_mortem(
    caplog: pytest.LogCaptureFixture,
) -> None:
    actuator = LoggingActuator()
    with caplog.at_level(logging.WARNING):
        actuator.apply(Action.WARN, None)
    assert any("WARN" in record.message and "warn" in record.message for record in caplog.records)


def test_logging_actuator_uses_provided_logger() -> None:
    custom_logger = logging.getLogger("pathlayer.test.custom_actuator")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    custom_logger.addHandler(handler)
    custom_logger.setLevel(logging.WARNING)

    actuator = LoggingActuator(logger=custom_logger)
    try:
        actuator.apply(Action.ESCALATE, None)
    finally:
        custom_logger.removeHandler(handler)

    assert len(records) == 1
    assert "ESCALATE" in records[0].getMessage()
