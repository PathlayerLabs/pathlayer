"""Structured logging (OBS-1): record every signal, decision and action.

Built on the standard library ``logging`` (zero dependencies, local-first).
Each record carries a structured ``pathlayer`` payload (kind, event index,
event-sourced timestamp, and the relevant fields) so a run can be inspected or
replayed offline. :class:`JsonFormatter` renders those payloads as JSON lines.

Logging is a passive side effect: it never alters the agent flow. Timestamps
come from the events (``ToolEvent.ts``), not from the wall clock.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from pathlayer.detectors.base import Signal
from pathlayer.policies import Action

LOGGER_NAME = "pathlayer"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


class JsonFormatter(logging.Formatter):
    """Render the structured ``pathlayer`` payload of a record as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"level": record.levelname, "message": record.getMessage()}
        data = getattr(record, "pathlayer", None)
        if data is not None:
            payload.update(data)
        return json.dumps(payload, default=str, sort_keys=True)


def _emit(logger: logging.Logger, kind: str, index: int, ts: float, **fields: Any) -> None:
    logger.info(kind, extra={"pathlayer": {"kind": kind, "index": index, "ts": ts, **fields}})


def log_signal(logger: logging.Logger, signal: Signal, *, index: int, ts: float) -> None:
    _emit(
        logger,
        "signal",
        index,
        ts,
        name=signal.name,
        confidence=signal.confidence,
        reason=signal.reason,
    )


def log_decision(
    logger: logging.Logger,
    action: Action,
    signals: Sequence[Signal],
    *,
    index: int,
    ts: float,
) -> None:
    _emit(
        logger,
        "decision",
        index,
        ts,
        action=action.name,
        signals=[{"name": s.name, "confidence": s.confidence} for s in signals],
    )


def log_action(logger: logging.Logger, action: Action, *, index: int, ts: float) -> None:
    _emit(logger, "action", index, ts, action=action.name)
