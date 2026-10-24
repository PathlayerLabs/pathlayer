"""Regression + calibration tests for the DEMO-2 example (examples/demo_no_progress.py).

The demo lives in ``examples/`` (outside the ``pathlayer`` package and outside
coverage), so it is loaded by path -- like ``tests/test_demo1.py``. These tests
guard the flagship no-progress story end to end and lock the thresholds that
make it safe:

* the slow burn is caught through the real runtime path and escalates the full
  ``warn -> summarize -> stop`` ladder, with loop/retry provably silent;
* the healthy counterpart is never intervened on;
* the retained DEMO-2 policy fires **zero STOP and zero SUMMARIZE** on every
  healthy corpus trace (a WARN nudge on a genuinely-slow trace is allowed and
  documented), while a negative baseline proves loop/retry must be advisory;
* the trace length that actually reaches STOP is pinned (a 12-event fixture only
  reaches SUMMARIZE; the escalation to STOP needs the longer tail).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.policies import Action
from pathlayer.validation.traces import load_trace

_DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "demo_no_progress.py"
_TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


def _load_demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_no_progress", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the demo uses `from __future__ import annotations`,
    # and TypedDict/dataclass field resolution looks the module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


def _churn(pairs: int) -> list[ToolEvent]:
    """A flat-progress churn: a different edit each turn, always the same red suite."""
    events: list[ToolEvent] = []
    for i in range(pairs):
        events.append(ToolEvent(tool="edit_file", args={"patch": f"a{i}"}, ts=float(2 * i)))
        events.append(
            ToolEvent(
                tool="run_tests",
                args={"target": "t"},
                result={"passed": 11, "failed": 3},
                ts=float(2 * i + 1),
            )
        )
    return events


def _actions(config: PathlayerConfig, events: list[ToolEvent]) -> tuple[list[Action], bool]:
    pipeline = build_pipeline(config)
    pipeline.run(events)
    return list(pipeline.actions), pipeline.stopped


def _max_confidence(detector, events: list[ToolEvent]) -> float:  # type: ignore[no-untyped-def]
    return max(
        (detector.evaluate(events[:i]).confidence for i in range(1, len(events) + 1)),
        default=0.0,
    )


# --------------------------------------------------------------------------- #
# Live graph: the flagship slow-burn story, end to end.
# --------------------------------------------------------------------------- #


def test_slow_burn_escalates_full_ladder_and_stops() -> None:
    pipeline, events, drained = demo._run_live(
        demo._stuck_model({"n": 0}), [demo.edit_file, demo.run_tests]
    )

    # STOP was enforced through the real drain path, well before the naive cap.
    assert drained is True
    assert pipeline.stopped is True
    assert len(pipeline.history) < demo.NAIVE_CAP

    # The graded ladder is shown in full and in order.
    assert demo._transitions(pipeline) == ["WARN", "SUMMARIZE", "STOP"]
    raw = list(pipeline.actions)
    assert raw.index(Action.WARN) < raw.index(Action.SUMMARIZE) < raw.index(Action.STOP)

    # The forced summary actually executed (captured as PROGRESS_SUMMARY, which
    # the detectors ignore).
    assert any(event.kind is EventKind.PROGRESS_SUMMARY for event in events)

    # Only no_progress drives it: repetition detectors stay silent on the varied
    # activity, while the no_progress CUSUM saturates.
    action_events = demo._action_events(events)
    assert _max_confidence(LoopDetector(), action_events) == 0.0
    assert _max_confidence(RetryDetector(), action_events) == 0.0
    assert _max_confidence(NoProgressDetector(coding_anchors()), action_events) == 1.0

    pm = pipeline.post_mortem
    assert pm is not None and "no_progress" in pm.reason


def test_healthy_live_run_is_never_intervened() -> None:
    pipeline, events, drained = demo._run_live(
        demo._healthy_model({"n": 0}), [demo.edit_file, demo.run_tests_improving]
    )

    assert drained is False
    assert pipeline.stopped is False
    assert all(action is Action.NONE for action in pipeline.actions)
    # Genuine progress keeps the no_progress signal low.
    assert _max_confidence(NoProgressDetector(coding_anchors()), demo._action_events(events)) < 0.3


# --------------------------------------------------------------------------- #
# Calibration: the retained DEMO-2 policy is safe on the healthy corpus.
# --------------------------------------------------------------------------- #


def test_demo_policy_never_stops_or_summarizes_on_healthy_corpus() -> None:
    # The load-bearing anti-false-positive guarantee: sweep every healthy trace
    # under the exact demo config and assert not one STOP or SUMMARIZE. A WARN is
    # tolerated (non-interrupting nudge) and is expected on genuinely-slow traces
    # like slow_but_real_progress / rich_new_target_progress -- documented here
    # so a future WARN there is not mistaken for a regression.
    traces = sorted((_TRACES_DIR / "healthy").glob("*.json"))
    assert traces, "expected a non-empty healthy corpus to calibrate against"

    violations: dict[str, list[str]] = {}
    for path in traces:
        actions, stopped = _actions(demo.demo_config(), load_trace(path).events)
        forbidden = [a.name for a in actions if a not in (Action.NONE, Action.WARN)]
        if stopped or forbidden:
            violations[path.stem] = [*forbidden, "STOPPED"] if stopped else forbidden
    assert violations == {}, (
        f"demo policy intervened (STOP/SUMMARIZE) on healthy traces: {violations}"
    )


def test_advisory_loop_retry_is_load_bearing() -> None:
    # Negative baseline that makes the advisory choice non-tautological: with
    # loop/retry *actionable* (same thresholds, no advisory), a healthy trace is
    # forced to SUMMARIZE by the loop/retry fusion (loop 0.75 + retry 0.50 ->
    # 0.875) -- exactly the false positive the demo config suppresses by making
    # loop/retry observe-only.
    actionable = PathlayerConfig()
    actionable.policy.tier_thresholds = {"warn": 0.3, "summarize": 0.5}
    actionable.policy.escalate_ticks = 2
    actions, _ = _actions(
        actionable, load_trace(_TRACES_DIR / "healthy" / "retry_then_succeed.json").events
    )
    assert Action.SUMMARIZE in actions

    # The demo config (loop/retry advisory) silences that exact trace.
    demo_actions, demo_stopped = _actions(
        demo.demo_config(), load_trace(_TRACES_DIR / "healthy" / "retry_then_succeed.json").events
    )
    assert Action.SUMMARIZE not in demo_actions
    assert not demo_stopped


# --------------------------------------------------------------------------- #
# Trace length: STOP needs the long tail; a 12-event fixture only summarizes.
# --------------------------------------------------------------------------- #


def test_short_churn_reaches_summarize_not_stop() -> None:
    # The 12-event flagship fixture (and an equivalent 6-pair churn) escalates to
    # SUMMARIZE but the two-tick hysteresis lands STOP past the trace's end.
    churn_trace = load_trace(_TRACES_DIR / "no_progress" / "failing_edit_churn.json")
    actions, stopped = _actions(demo.demo_config(), churn_trace.events)
    assert Action.SUMMARIZE in actions
    assert Action.STOP not in actions
    assert not stopped

    # An equivalent 6-pair (12-event) churn also escalates to SUMMARIZE but no
    # further -- asserting the tier (not just "not stopped") makes this fail on a
    # sub-detection regression, not just a spurious STOP.
    actions_6, stopped_6 = _actions(demo.demo_config(), _churn(6))
    assert Action.SUMMARIZE in actions_6
    assert Action.STOP not in actions_6
    assert not stopped_6


def test_extended_churn_reaches_stop() -> None:
    # The demo generates CHURN_PAIRS (10) pairs precisely so the escalation
    # reaches STOP with margin.
    actions, stopped = _actions(demo.demo_config(), _churn(demo.CHURN_PAIRS))
    assert stopped
    transitions = [a.name for a in dict.fromkeys(a for a in actions if a is not Action.NONE)]
    assert transitions == ["WARN", "SUMMARIZE", "STOP"]


# --------------------------------------------------------------------------- #
# CLI: before/after output carries the required numeric evidence (DEMO-2 DoD).
# --------------------------------------------------------------------------- #


def test_main_prints_before_after_evidence(capsys) -> None:  # type: ignore[no-untyped-def]
    demo.main()
    out = capsys.readouterr().out

    assert "Without Pathlayer" in out
    assert "STOP (run drained)" in out
    assert "WARN -> SUMMARIZE -> STOP" in out
    assert "no_progress=1.00" in out
    assert "Forced summary       : yes" in out
    assert "Work avoided" in out
    assert "no intervention" in out
