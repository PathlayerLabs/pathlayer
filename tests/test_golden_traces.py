"""VAL-2: guard the curated golden set of labeled traces.

These tests discover *every* committed trace under ``traces/`` (recursively),
load it through the VAL-1 loader, and assert the corpus stays well-formed and
balanced: a valid label, non-empty events, and at least five runs per class.
The set anchors threshold tuning and regression tests, so it must not silently
shrink or drift.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from pathlayer.events import ToolEvent
from pathlayer.validation import VALID_LABELS, load_trace

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"

MIN_PER_CLASS = 5

# Sorted for deterministic test ordering / ids.
_TRACE_FILES = sorted(_TRACES_DIR.rglob("*.json"))


def test_traces_dir_is_populated() -> None:
    assert _TRACE_FILES, f"no trace files discovered under {_TRACES_DIR}"


@pytest.mark.parametrize("path", _TRACE_FILES, ids=lambda p: str(p.relative_to(_TRACES_DIR)))
def test_trace_is_well_formed(path: Path) -> None:
    trace = load_trace(path)
    assert trace.label in VALID_LABELS
    assert trace.events, f"{path} has no events"
    assert all(isinstance(event, ToolEvent) for event in trace.events)
    assert all(event.tool for event in trace.events), f"{path} has an event with no tool"


@pytest.mark.parametrize("path", _TRACE_FILES, ids=lambda p: str(p.relative_to(_TRACES_DIR)))
def test_trace_has_description(path: Path) -> None:
    assert load_trace(path).description.strip(), f"{path} is missing a description"


def test_at_least_five_traces_per_class() -> None:
    counts = Counter(load_trace(path).label for path in _TRACE_FILES)
    for label in VALID_LABELS:
        n = counts[label]
        msg = f"class {label!r} has {n} traces, expected >= {MIN_PER_CLASS}"
        assert n >= MIN_PER_CLASS, msg
