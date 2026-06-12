"""Tests for the LoopDetector (LOOP-1).

Cover the two directions that matter: a trivial loop must produce high
confidence; a healthy, varied run must stay near zero. Also pin the K
threshold, determinism, and the synergy with canonicalization (a loop whose
arguments carry fresh volatile noise each turn is still caught).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import ToolEvent


def _repeat(block: list[tuple[str, dict[str, object]]], times: int) -> list[ToolEvent]:
    return [ToolEvent(tool=tool, args=args) for _ in range(times) for tool, args in block]


def test_empty_history_is_zero() -> None:
    assert LoopDetector().evaluate([]).confidence == 0.0


def test_healthy_varied_run_is_zero() -> None:
    history = [
        ToolEvent(tool="read", args={"path": "a.py"}),
        ToolEvent(tool="edit", args={"path": "a.py"}),
        ToolEvent(tool="run_tests", args={}),
        ToolEvent(tool="read", args={"path": "b.py"}),
        ToolEvent(tool="edit", args={"path": "b.py"}),
    ]
    assert LoopDetector().evaluate(history).confidence == 0.0


def test_single_action_loop_detected() -> None:
    history = _repeat([("read", {"path": "a.py"})], times=6)
    signal = LoopDetector().evaluate(history)
    assert signal.confidence > 0.5


def test_two_action_loop_detected() -> None:
    history = _repeat([("list", {"p": "x"}), ("read", {"p": "y"})], times=6)
    signal = LoopDetector().evaluate(history)
    assert signal.confidence > 0.5


def test_confidence_increases_with_repetitions() -> None:
    few = LoopDetector().evaluate(_repeat([("read", {"p": "a"})], times=3))
    many = LoopDetector().evaluate(_repeat([("read", {"p": "a"})], times=8))
    assert many.confidence > few.confidence


def test_confidence_in_unit_interval() -> None:
    signal = LoopDetector().evaluate(_repeat([("read", {"p": "a"})], times=20))
    assert 0.0 <= signal.confidence <= 1.0


def test_k_threshold_blocks_short_loops() -> None:
    detector = LoopDetector(k=4)
    # Block repeated only 3 times: below k -> silent.
    assert detector.evaluate(_repeat([("read", {"p": "a"})], times=3)).confidence == 0.0
    # Repeated 4 times: at k -> fires.
    assert detector.evaluate(_repeat([("read", {"p": "a"})], times=4)).confidence > 0.0


def test_detects_loop_despite_volatile_args() -> None:
    # Same logical action each turn, but a fresh request_id every time.
    history = [
        ToolEvent(tool="fetch", args={"url": "/status", "request_id": f"r-{i}"}) for i in range(8)
    ]
    assert LoopDetector().evaluate(history).confidence > 0.5


def test_is_deterministic() -> None:
    history = _repeat([("a", {}), ("b", {})], times=5)
    detector = LoopDetector()
    assert detector.evaluate(history) == detector.evaluate(history)


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be"):
        LoopDetector(k=1)


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        LoopDetector(window=3)  # default k=2 needs window >= 4


def test_window_detects_recent_loop_in_long_history() -> None:
    # A long, varied prefix then a clear loop in the recent tail: still caught,
    # and per-step cost stays bounded by the window.
    prefix = [ToolEvent(tool=f"step_{i}", args={}) for i in range(50)]
    loop = _repeat([("read", {"p": "a"})], times=6)
    assert LoopDetector(window=12).evaluate(prefix + loop).confidence > 0.5


def test_window_ignores_loop_outside_window() -> None:
    # The loop has scrolled out of the window; only the varied tail remains.
    loop = _repeat([("read", {"p": "a"})], times=6)
    tail = [ToolEvent(tool=f"step_{i}", args={}) for i in range(20)]
    assert LoopDetector(window=10).evaluate(loop + tail).confidence == 0.0


def test_fixture_loop_has_high_confidence(trace: Callable[[str], list[ToolEvent]]) -> None:
    history = trace("loop_trivial.json")
    signal = LoopDetector().evaluate(history)
    assert signal.confidence >= 0.9
    assert signal.name == "loop"
