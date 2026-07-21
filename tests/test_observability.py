"""Tests for structured logging (OBS-1).

Pin the structured payloads, JSON rendering, event-sourced timestamps, and the
fact that logging is passive: a pipeline produces identical results with or
without a logger attached.
"""

from __future__ import annotations

import json
import logging

import pytest

from pathlayer import Pipeline
from pathlayer.detectors.base import Signal
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import ToolEvent
from pathlayer.observability import (
    JsonFormatter,
    get_logger,
    log_action,
    log_decision,
    log_signal,
    log_would_be_action,
)
from pathlayer.policies import Action


def _records(caplog: pytest.LogCaptureFixture, kind: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "pathlayer", {}).get("kind") == kind]


def test_get_logger_uses_pathlayer_namespace() -> None:
    assert get_logger().name == "pathlayer"


def test_log_signal_payload(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("pathlayer.test.signal")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_signal(logger, Signal("loop", 0.7, "3x repeat"), index=2, ts=1.5)

    (record,) = _records(caplog, "signal")
    payload = record.pathlayer
    assert payload["name"] == "loop"
    assert payload["confidence"] == 0.7
    assert payload["index"] == 2
    assert payload["ts"] == 1.5


def test_log_signal_omits_diagnostics_when_empty(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("pathlayer.test.signal.no_diagnostics")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_signal(logger, Signal("no_progress", 0.0), index=0, ts=0.0)

    (record,) = _records(caplog, "signal")
    assert "diagnostics" not in record.pathlayer


def test_log_signal_includes_diagnostics_when_present(caplog: pytest.LogCaptureFixture) -> None:
    # A degraded producer (malformed payload, capacity cap, guarded reversion)
    # must be visible even on an otherwise-silent (confidence 0) tick -- that
    # is precisely what closes the "silent degradation" gap (NOP-5 follow-up).
    logger = get_logger("pathlayer.test.signal.diagnostics")
    diagnostics = ("coding_tests: target 'suite': malformed test_outcomes payload ignored",)
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_signal(logger, Signal("no_progress", 0.0, diagnostics=diagnostics), index=0, ts=0.0)

    (record,) = _records(caplog, "signal")
    assert record.pathlayer["diagnostics"] == list(diagnostics)


def test_log_decision_and_action_payloads(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("pathlayer.test.decision")
    with caplog.at_level(logging.INFO, logger=logger.name):
        signals = [Signal("loop", 0.95)]
        log_decision(logger, Action.STOP, signals, fused=0.95, index=4, ts=2.0)
        log_action(logger, Action.STOP, index=4, ts=2.0)

    (decision,) = _records(caplog, "decision")
    assert decision.pathlayer["action"] == "STOP"
    assert decision.pathlayer["fused"] == 0.95
    assert decision.pathlayer["signals"] == [{"name": "loop", "confidence": 0.95}]

    (action,) = _records(caplog, "action")
    assert action.pathlayer["action"] == "STOP"


def test_log_would_be_action_uses_distinct_kind(caplog: pytest.LogCaptureFixture) -> None:
    # A suppressed (dry-run) action must be distinguishable from an emitted one:
    # kind=would_be_action, never kind=action.
    logger = get_logger("pathlayer.test.wouldbe")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_would_be_action(logger, Action.STOP, index=4, ts=2.0)

    (record,) = _records(caplog, "would_be_action")
    assert record.pathlayer["action"] == "STOP"
    assert record.pathlayer["index"] == 4
    assert record.pathlayer["ts"] == 2.0
    assert _records(caplog, "action") == []


def test_json_formatter_renders_payload() -> None:
    logger = logging.getLogger("pathlayer.test.json")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 0, "signal", (), None)
    record.pathlayer = {"kind": "signal", "name": "loop", "confidence": 0.5}

    rendered = json.loads(JsonFormatter().format(record))
    assert rendered["kind"] == "signal"
    assert rendered["name"] == "loop"
    assert rendered["level"] == "INFO"


def test_json_formatter_without_payload() -> None:
    logger = logging.getLogger("pathlayer.test.json.plain")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 0, "hello", (), None)

    rendered = json.loads(JsonFormatter().format(record))
    assert rendered["message"] == "hello"
    assert "kind" not in rendered


def test_pipeline_logs_signal_and_decision_per_event(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("pathlayer.test.pipeline")
    pipeline = Pipeline([LoopDetector()], logger=logger)
    events = [ToolEvent(tool="read", args={"path": "a.py"}, ts=float(i)) for i in range(3)]

    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline.run(events)

    assert len(_records(caplog, "signal")) == 3
    assert len(_records(caplog, "decision")) == 3


def test_pipeline_logs_action_on_stop(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("pathlayer.test.stop")
    pipeline = Pipeline([LoopDetector()], logger=logger)
    events = [ToolEvent(tool="read", args={"path": "a.py"}, ts=float(i)) for i in range(12)]

    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline.run(events)

    actions = _records(caplog, "action")
    assert actions
    assert all(r.pathlayer["action"] == "STOP" for r in actions)


def test_logged_timestamp_comes_from_event(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("pathlayer.test.ts")
    pipeline = Pipeline([LoopDetector()], logger=logger)

    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline.process(ToolEvent(tool="read", ts=42.0))

    (signal,) = _records(caplog, "signal")
    assert signal.pathlayer["ts"] == 42.0


def test_logging_is_passive() -> None:
    events = [ToolEvent(tool="read", args={"path": "a.py"}, ts=float(i)) for i in range(12)]

    silent = Pipeline([LoopDetector()]).run(events)
    loud = Pipeline([LoopDetector()], logger=get_logger("pathlayer.test.passive")).run(events)

    assert silent == loud
