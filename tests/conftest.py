"""Shared test helpers: load labeled fixture traces from ``traces/``.

The on-disk trace format is intentionally minimal for Sprint 1 (a JSON object
with a ``label`` and a list of ``events``); the formal schema is VAL-1.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from pathlayer.events import ToolEvent

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def load_trace(name: str) -> list[ToolEvent]:
    data = json.loads((_TRACES_DIR / name).read_text(encoding="utf-8"))
    return [ToolEvent(tool=event["tool"], args=event.get("args", {})) for event in data["events"]]


@pytest.fixture
def trace() -> Callable[[str], list[ToolEvent]]:
    return load_trace
