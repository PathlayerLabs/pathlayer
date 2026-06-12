"""End-to-end test for Stop + post-mortem (ACT-5) -- the Sprint 1 demo.

An in-memory adapter replays a looping trace; the LoopDetector raises
confidence, arbitration decides STOP, the pipeline halts, and a post-mortem is
captured. A healthy trace runs to completion with no post-mortem.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from pathlayer import Pipeline, PostMortem
from pathlayer.adapters import InMemoryAdapter
from pathlayer.detectors.base import Signal
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import ToolEvent
from pathlayer.policies import Action, Policy


def _healthy() -> list[ToolEvent]:
    return [
        ToolEvent(tool="read", args={"path": "a.py"}),
        ToolEvent(tool="edit", args={"path": "a.py"}),
        ToolEvent(tool="run_tests", args={}),
        ToolEvent(tool="read", args={"path": "b.py"}),
    ]


def test_loop_is_caught_and_stopped(trace: Callable[[str], list[ToolEvent]]) -> None:
    events = trace("loop_trivial.json")
    pipeline = Pipeline([LoopDetector()])
    adapter = InMemoryAdapter(pipeline.sink, events)

    adapter.attach()

    assert pipeline.stopped is True
    assert pipeline.post_mortem is not None
    assert pipeline.post_mortem.action is Action.STOP
    assert "loop" in pipeline.post_mortem.reason


def test_events_after_stop_are_ignored(trace: Callable[[str], list[ToolEvent]]) -> None:
    events = trace("loop_trivial.json")
    pipeline = Pipeline([LoopDetector()])
    InMemoryAdapter(pipeline.sink, events).attach()

    # The loop is caught before the trace is exhausted; the tail is not processed.
    assert len(pipeline.history) < len(events)
    assert pipeline.actions[-1] is Action.STOP
    assert pipeline.actions.count(Action.STOP) == 1


def test_run_stops_at_first_stop(trace: Callable[[str], list[ToolEvent]]) -> None:
    events = trace("loop_trivial.json")
    actions = Pipeline([LoopDetector()]).run(events)
    assert actions[-1] is Action.STOP
    assert actions.count(Action.STOP) == 1
    assert len(actions) < len(events)


def test_healthy_run_has_no_post_mortem() -> None:
    pipeline = Pipeline([LoopDetector()])
    pipeline.run(_healthy())
    assert pipeline.stopped is False
    assert pipeline.post_mortem is None


def test_post_mortem_is_deterministic(trace: Callable[[str], list[ToolEvent]]) -> None:
    events = trace("loop_trivial.json")
    first = Pipeline([LoopDetector()])
    first.run(events)
    second = Pipeline([LoopDetector()])
    second.run(events)
    assert first.post_mortem is not None
    assert second.post_mortem is not None
    assert first.post_mortem.stopped_at_index == second.post_mortem.stopped_at_index


def test_post_mortem_to_dict_shape(trace: Callable[[str], list[ToolEvent]]) -> None:
    pipeline = Pipeline([LoopDetector()])
    pipeline.run(trace("loop_trivial.json"))
    assert pipeline.post_mortem is not None
    data = pipeline.post_mortem.to_dict()
    assert data["action"] == "STOP"
    assert data["events_processed"] == len(pipeline.history)
    assert isinstance(data["signals"], list) and data["signals"]
    assert isinstance(data["recent_history"], list) and data["recent_history"]


def test_post_mortem_write_roundtrip(
    trace: Callable[[str], list[ToolEvent]], tmp_path: Path
) -> None:
    pipeline = Pipeline([LoopDetector()])
    pipeline.run(trace("loop_trivial.json"))
    assert pipeline.post_mortem is not None

    out = tmp_path / "postmortem.json"
    pipeline.post_mortem.write(out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["action"] == "STOP"
    assert "loop" in loaded["reason"]


def test_post_mortem_recent_history_is_bounded() -> None:
    # A long single-action loop; recent_history must stay capped.
    events = [ToolEvent(tool="read", args={"path": "a.py"}) for _ in range(40)]
    pipeline = Pipeline([LoopDetector()], recent_history=5)
    pipeline.run(events)
    assert pipeline.post_mortem is not None
    assert len(pipeline.post_mortem.recent_history) == 5


def test_post_mortem_is_a_dataclass_instance(trace: Callable[[str], list[ToolEvent]]) -> None:
    pipeline = Pipeline([LoopDetector()])
    pipeline.run(trace("loop_trivial.json"))
    assert isinstance(pipeline.post_mortem, PostMortem)


def test_negative_recent_history_raises() -> None:
    with pytest.raises(ValueError, match="recent_history"):
        Pipeline([LoopDetector()], recent_history=-1)


def test_zero_recent_history_means_no_tail() -> None:
    events = [ToolEvent(tool="read", args={"path": "a.py"}) for _ in range(8)]
    pipeline = Pipeline([LoopDetector()], recent_history=0)
    pipeline.run(events)
    assert pipeline.post_mortem is not None
    # recent_history=0 must yield an empty tail, not the whole history.
    assert pipeline.post_mortem.recent_history == ()


class _AlwaysStopPolicy(Policy):
    def decide(self, signals: Sequence[Signal]) -> Action:
        return Action.STOP


def test_post_mortem_handles_empty_signals() -> None:
    # A custom policy can STOP with no detectors/signals; must not crash.
    pipeline = Pipeline([], policy=_AlwaysStopPolicy())
    pipeline.process(ToolEvent(tool="read"))
    assert pipeline.post_mortem is not None
    assert pipeline.post_mortem.signals == ()
    assert pipeline.post_mortem.reason == "stop"
