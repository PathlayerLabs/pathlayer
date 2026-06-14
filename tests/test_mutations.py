"""VAL-4: tests for the mutation engine."""

from __future__ import annotations

import copy

import pytest

from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import ToolEvent
from pathlayer.validation import (
    Trace,
    inject_cost_spike,
    inject_loop,
    inject_no_progress,
    inject_retry_storm,
)


def _healthy() -> Trace:
    """A varied, advancing run with no repetition (loop detector silent)."""
    return Trace(
        label="healthy",
        description="varied run",
        events=[
            ToolEvent(tool="list_dir", args={"path": "src"}, cost_usd=0.1, ts=1.0),
            ToolEvent(tool="read_file", args={"path": "src/a.py"}, cost_usd=0.1, ts=2.0),
            ToolEvent(tool="edit_file", args={"path": "src/a.py"}, cost_usd=0.2, ts=3.0),
            ToolEvent(tool="run_tests", args={"target": "tests/a"}, cost_usd=0.2, ts=4.0),
            ToolEvent(tool="git_commit", args={"message": "fix a"}, cost_usd=0.1, ts=5.0),
        ],
    )


def _loop_fires_during_replay(trace: Trace) -> bool:
    """True if the loop detector fires on any growing prefix of the trace.

    LoopDetector only flags a *trailing* loop (an agent looping right now), so a
    loop injected mid-trace is only visible while it is the tail -- exactly how
    the VAL-3 replay harness will see it as events stream in.
    """
    detector = LoopDetector()
    return any(
        detector.evaluate(trace.events[:i]).confidence > 0.0
        for i in range(1, len(trace.events) + 1)
    )


def _expectations(trace: Trace) -> list[dict[str, object]]:
    return trace.metadata["expectations"]


# --------------------------------------------------------------------------- #
# inject_loop
# --------------------------------------------------------------------------- #
def test_inject_loop_label_and_expectation() -> None:
    out = inject_loop(_healthy(), start_idx=1, sequence_length=2, n_repeats=3)
    assert out.label == "loop"
    exp = _expectations(out)
    assert exp == [{"detector": "loop", "should_fire": True, "earliest_event_idx": 4}]


def test_inject_loop_lengthens_and_fires() -> None:
    base = _healthy()
    out = inject_loop(base, start_idx=1, sequence_length=2, n_repeats=3)
    assert len(out.events) == len(base.events) + 2 * 3
    assert _loop_fires_during_replay(out)


def test_inject_loop_does_not_mutate_input() -> None:
    base = _healthy()
    snapshot = copy.deepcopy(base)
    inject_loop(base, start_idx=0, sequence_length=1, n_repeats=2)
    assert base == snapshot


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_idx": 0, "sequence_length": 0, "n_repeats": 1},
        {"start_idx": 0, "sequence_length": 1, "n_repeats": 0},
        {"start_idx": 4, "sequence_length": 3, "n_repeats": 1},
        {"start_idx": -1, "sequence_length": 1, "n_repeats": 1},
    ],
)
def test_inject_loop_validates(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        inject_loop(_healthy(), **kwargs)


# --------------------------------------------------------------------------- #
# inject_retry_storm
# --------------------------------------------------------------------------- #
def test_inject_retry_storm_label_and_expectations() -> None:
    out = inject_retry_storm(_healthy(), start_idx=2, n_events=5, tool="api_call", error="429")
    assert out.label == "loop"
    detectors = {e["detector"] for e in _expectations(out)}
    assert detectors == {"retry", "loop"}


def test_inject_retry_storm_is_a_period_one_loop() -> None:
    out = inject_retry_storm(_healthy(), start_idx=2, n_events=5, tool="api_call", error="429")
    # Identical failing calls canonicalize to one token -> the loop detector fires.
    assert _loop_fires_during_replay(out)


def test_inject_retry_storm_does_not_mutate_input() -> None:
    base = _healthy()
    snapshot = copy.deepcopy(base)
    inject_retry_storm(base, start_idx=0, n_events=3, tool="x", error="boom")
    assert base == snapshot


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_idx": 0, "n_events": 1, "tool": "x", "error": "e"},
        {"start_idx": 99, "n_events": 3, "tool": "x", "error": "e"},
        {"start_idx": -1, "n_events": 3, "tool": "x", "error": "e"},
    ],
)
def test_inject_retry_storm_validates(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        inject_retry_storm(_healthy(), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# inject_no_progress
# --------------------------------------------------------------------------- #
def test_inject_no_progress_label_and_expectation() -> None:
    out = inject_no_progress(_healthy(), start_idx=2)
    assert out.label == "no_progress"
    assert _expectations(out) == [
        {"detector": "no_progress", "should_fire": True, "earliest_event_idx": 3}
    ]


def test_inject_no_progress_marks_tail_and_stays_distinct() -> None:
    out = inject_no_progress(_healthy(), start_idx=2)
    for event in out.events[2:]:
        assert event.metadata.get("progress") is False
    # Events stay distinct, so the repetition detector must NOT fire.
    assert not _loop_fires_during_replay(out)


def test_inject_no_progress_does_not_mutate_input() -> None:
    base = _healthy()
    snapshot = copy.deepcopy(base)
    inject_no_progress(base, start_idx=1)
    assert base == snapshot


@pytest.mark.parametrize("start_idx", [-1, 5, 99])
def test_inject_no_progress_validates(start_idx: int) -> None:
    with pytest.raises(ValueError):
        inject_no_progress(_healthy(), start_idx)


# --------------------------------------------------------------------------- #
# inject_cost_spike
# --------------------------------------------------------------------------- #
def test_inject_cost_spike_preserves_label_and_inflates_cost() -> None:
    base = _healthy()
    out = inject_cost_spike(base, event_idx=3, multiplier=50.0)
    assert out.label == base.label  # unchanged
    assert out.events[3].cost_usd == pytest.approx(base.events[3].cost_usd * 50.0)
    assert _expectations(out) == [
        {"detector": "budget", "should_fire": True, "earliest_event_idx": 3}
    ]


def test_inject_cost_spike_zero_cost_uses_unit_base() -> None:
    trace = Trace(label="healthy", events=[ToolEvent(tool="x")])
    out = inject_cost_spike(trace, event_idx=0, multiplier=10.0)
    assert out.events[0].cost_usd == pytest.approx(10.0)


def test_inject_cost_spike_does_not_mutate_input() -> None:
    base = _healthy()
    snapshot = copy.deepcopy(base)
    inject_cost_spike(base, event_idx=0, multiplier=10.0)
    assert base == snapshot


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event_idx": 0, "multiplier": 1.0},
        {"event_idx": 0, "multiplier": 0.5},
        {"event_idx": 99, "multiplier": 2.0},
        {"event_idx": -1, "multiplier": 2.0},
    ],
)
def test_inject_cost_spike_validates(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        inject_cost_spike(_healthy(), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# shared properties
# --------------------------------------------------------------------------- #
def test_base_metadata_is_preserved_and_not_shared() -> None:
    base = Trace(label="healthy", metadata={"seed": 7}, events=[ToolEvent(tool="x")])
    out = inject_cost_spike(base, event_idx=0, multiplier=2.0)
    assert out.metadata["seed"] == 7
    out.metadata["seed"] = 999
    assert base.metadata["seed"] == 7  # deep-copied, not aliased
