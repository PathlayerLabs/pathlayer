"""OBS-2: per-tick records, the end-of-session report, and ``pathlayer-report``.

The recurring assertion across this file is an honesty one: the report must never
present an absence of measurement as a result. An unmeasured cost is reported as
unmeasured (never ``$0.00``), a journal without detector confidences is reported
as carrying none (never "no detector reacted"), and a run nobody saw finish is
``interrupted`` (never ``completed``).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_report import (
    COST_BUDGET_DISABLED,
    COST_BUDGET_ENABLED,
    COST_BUDGET_STATUS_KEY,
    COST_BUDGET_UNKNOWN,
    build_run_report,
    main,
    tick_columns,
    write_run_report,
)
from pathlayer.adapters.openclaw_run_store import RunStore, append_jsonl, atomic_write_json
from pathlayer.config import (
    BudgetConfig,
    DetectorsConfig,
    PathlayerConfig,
    PipelineSection,
    PolicySection,
)
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.proof import ProofSpec, compute_spec_hash
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.policies.base import PolicyConfig
from pathlayer.postmortem import (
    CostBreakdown,
    SessionOutcome,
    SignalObservation,
    TickRecord,
    build_coding_diagnostic,
    build_session_report,
)
from pathlayer.validation.traces import load_trace

# A run of >= 5 identical tool calls reaches loop confidence 0.9375+, above the
# default stop_threshold (0.9), so the default pipeline STOPs -- no double needed.
_STOP_REPEATS = 6


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _Constant(Detector):
    """A detector pinned at one confidence, to script a tick precisely."""

    def __init__(self, name: str, confidence: float, reason: str = "") -> None:
        self.name = name
        self._confidence = confidence
        self._reason = reason

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        return Signal(self.name, confidence=self._confidence, reason=self._reason)


def _ctx(run_id: str) -> dict[str, Any]:
    return {"runId": run_id, "sessionKey": "sess-1", "agentId": "agent-1"}


def _before(run_id: str) -> dict[str, Any]:
    return {"hook": "before_agent_run", "event": {"prompt": "fix it"}, "ctx": _ctx(run_id)}


def _tool(name: str, run_id: str, call: str, path: str = "a.py") -> dict[str, Any]:
    return {
        "hook": "after_tool_call",
        "event": {
            "toolName": name,
            "params": {"path": path},
            "result": "ok",
            "toolCallId": call,
            "runId": run_id,
        },
        "ctx": _ctx(run_id),
    }


def _end(run_id: str, *, success: bool = True) -> dict[str, Any]:
    return {
        "hook": "agent_end",
        "event": {"success": success, "durationMs": 4200},
        "ctx": _ctx(run_id),
    }


def _looping(run_id: str, count: int = _STOP_REPEATS) -> list[dict[str, Any]]:
    return [_tool("read", run_id, f"c{i}") for i in range(count)]


def _record_run(
    runs_root: Path,
    run_id: str,
    tools: Sequence[dict[str, Any]],
    *,
    config: PathlayerConfig | None = None,
    finish: bool = True,
) -> None:
    recorder = OpenClawRunRecorder(runs_root, config=config)
    recorder.process_payload(_before(run_id))
    for payload in tools:
        recorder.process_payload(payload)
    if finish:
        recorder.process_payload(_end(run_id))


def _run(runs_root: Path, run_id: str) -> Any:
    run = RunStore(runs_root).resolve_run(run_id)
    assert run is not None
    return run


def _decisions(run: Any) -> list[dict[str, Any]]:
    lines = run.decisions_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def _tick(index: int, **overrides: Any) -> TickRecord:
    # ``fused`` defaults to 0.0, not to None: this helper stands in for a tick a
    # live pipeline produced, and such a tick always recorded a fused score. Only
    # a journal line predating OBS-2 has none, and those tests pass fused=None.
    defaults: dict[str, Any] = {"index": index, "ts": float(index), "fused": 0.0}
    defaults.update(overrides)
    return TickRecord(**defaults)


# --------------------------------------------------------------------------- #
# TickRecord.from_decision: the contract the OpenClaw recorder persists
# --------------------------------------------------------------------------- #


def _projected_tick(pipeline: Pipeline) -> TickRecord | None:
    """Exactly what the recorder writes: a projection of the canonical decision.

    The pipeline owns one decision slot (INF-1); OBS-2 never keeps a second
    representation, so these tests read the same value the journal receives.
    """
    decision = pipeline.last_decision
    return None if decision is None else TickRecord.from_decision(decision)


def test_projected_tick_is_none_before_any_event() -> None:
    pipeline = Pipeline([_Constant("quiet", 0.0)])
    assert _projected_tick(pipeline) is None


def test_projected_tick_keeps_silent_detectors_so_wired_and_unwired_stay_distinct() -> None:
    pipeline = Pipeline([_Constant("loud", 0.5, "half"), _Constant("quiet", 0.0)])
    pipeline.process(ToolEvent(tool="read", ts=3.0))

    tick = _projected_tick(pipeline)
    assert tick is not None
    assert [(s.name, s.confidence) for s in tick.signals] == [("loud", 0.5), ("quiet", 0.0)]
    assert [s.name for s in tick.reacting_signals] == ["loud"]
    assert (tick.index, tick.ts) == (0, 3.0)
    assert tick.fused == pytest.approx(0.5)


def test_projected_tick_carries_no_reason_when_nothing_was_decided() -> None:
    # dominant_signal() names a detector even at confidence 0; attributing a
    # reason to a NONE tick would blame a detector that decided nothing.
    pipeline = Pipeline([_Constant("loop", 0.1, "one repeat")])
    pipeline.process(ToolEvent(tool="read"))

    tick = _projected_tick(pipeline)
    assert tick is not None
    assert tick.action is Action.NONE
    assert tick.reason == ""


def test_projected_tick_reports_the_driving_signal_when_an_action_fires() -> None:
    pipeline = Pipeline([_Constant("loop", 1.0, "5x repeat")])
    pipeline.process(ToolEvent(tool="read"))

    tick = _projected_tick(pipeline)
    assert tick is not None
    assert tick.action is Action.STOP
    assert tick.reason == "loop: 5x repeat"
    assert tick.dry_run is False


def test_projected_tick_is_deterministic_across_repeated_reads() -> None:
    pipeline = Pipeline([_Constant("quiet", 0.0)])
    pipeline.process(ToolEvent(tool="read"))
    assert _projected_tick(pipeline) == _projected_tick(pipeline)


def test_projected_tick_is_not_updated_by_an_ignored_post_stop_event() -> None:
    # A halted pipeline runs no detector, so claiming a reading for the events
    # that follow would be an invention.
    pipeline = Pipeline([_Constant("loop", 1.0, "5x repeat")])
    pipeline.process(ToolEvent(tool="read", ts=0.0))
    stop_tick = _projected_tick(pipeline)

    assert pipeline.process(ToolEvent(tool="edit", ts=1.0)) is Action.STOP
    assert _projected_tick(pipeline) == stop_tick


def test_projected_tick_marks_a_dry_run_decision_as_never_emitted() -> None:
    pipeline = Pipeline([_Constant("loop", 1.0, "5x repeat")], dry_run=True)
    assert pipeline.process(ToolEvent(tool="read")) is Action.NONE

    tick = _projected_tick(pipeline)
    assert tick is not None
    assert tick.action is Action.STOP
    assert tick.dry_run is True


def test_projected_tick_survives_a_raising_detector_untouched() -> None:
    class _Boom(Detector):
        name = "boom"

        def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
            if len(history) > 1:
                raise RuntimeError("detector exploded")
            return Signal(self.name, confidence=0.0)

    pipeline = Pipeline([_Boom()])
    pipeline.process(ToolEvent(tool="read", ts=0.0))
    first = _projected_tick(pipeline)

    with pytest.raises(RuntimeError, match="exploded"):
        pipeline.process(ToolEvent(tool="edit", ts=1.0))

    assert _projected_tick(pipeline) == first


def test_tick_record_round_trips_through_its_dict_form() -> None:
    # The shape a durable journal persists: a tick must survive a store-and-reload
    # verbatim.
    tick = TickRecord(
        index=4,
        ts=4.0,
        signals=(SignalObservation("loop", 0.94, "5x repeat"), SignalObservation("retry", 0.0)),
        fused=0.94,
        action=Action.STOP,
        reason="loop: 5x repeat",
        dry_run=False,
    )
    assert TickRecord.from_dict(tick.to_dict()) == tick
    assert tick.to_dict()["action"] == "STOP"
    assert tick.to_dict()["signals"][0]["confidence"] == 0.94


def test_tick_record_from_dict_defaults_absent_fields() -> None:
    tick = TickRecord.from_dict({"index": 1, "ts": 2.0})
    # An absent fused score reloads as unknown, never as a measured 0.0.
    assert (tick.signals, tick.fused, tick.action, tick.reason) == ((), None, Action.NONE, "")
    assert TickRecord.from_dict({"index": 1, "ts": 2.0, "fused": 0.0}).fused == 0.0
    assert SignalObservation.from_dict({"name": "loop", "confidence": 0.5}).reason == ""


# --------------------------------------------------------------------------- #
# build_session_report: pure aggregation
# --------------------------------------------------------------------------- #


def test_healthy_run_is_completed_with_no_action() -> None:
    ticks = [_tick(i, signals=(SignalObservation("loop", 0.0),)) for i in range(3)]
    report = build_session_report(
        run_id="run-clean", status="completed", ticks=ticks, events_recorded=3
    )

    data = report.to_dict()
    assert data["outcome"] == "completed"
    assert data["actions"] == []
    assert data["events_analyzed"] == 3
    assert "never left 0.00" in report.format()


def test_zero_fused_override_stop_does_not_claim_nothing_fired() -> None:
    # A detector with a fusion weight of 0 can still drive an action through an
    # override rule (see policies/base.py's override_signal): fused stays 0.0
    # even though a STOP was emitted. The report must not describe that as "no
    # actionable signal ever fired" next to an action it just listed.
    ticks = [_tick(0, action=Action.STOP, reason="budget: override at 0.98")]
    report = build_session_report(
        run_id="run-zero-fused-override", status="completed", ticks=ticks, events_recorded=1
    )

    formatted = report.format()
    assert "no actionable signal ever fired" not in formatted
    assert "never left 0.00" in formatted
    assert len(report.emitted_actions) == 1


def test_stopped_run_reports_the_stop_and_its_reason() -> None:
    ticks = [
        _tick(0, signals=(SignalObservation("loop", 0.4),), fused=0.4),
        _tick(
            1,
            signals=(SignalObservation("loop", 0.95, "5x repeat"),),
            fused=0.95,
            action=Action.STOP,
            reason="loop: 5x repeat",
        ),
    ]
    report = build_session_report(
        run_id="run-stop", status="completed", ticks=ticks, events_recorded=4, stopped=True
    )

    data = report.to_dict()
    assert data["outcome"] == "stopped"
    assert data["stop_reason"] == "loop: 5x repeat"
    assert data["actions"] == [
        {"action": "STOP", "index": 1, "ts": 1.0, "reason": "loop: 5x repeat", "emitted": True}
    ]
    text = report.format()
    assert "outcome: stopped" in text
    assert "stop reason: loop: 5x repeat" in text
    # 4 events recorded but only 2 analyzed: a passive STOP ends analysis, not
    # the run, and the report must not silently drop the difference.
    assert "4 recorded, 2 analyzed" in text


def test_escalate_without_stop_is_reported_as_escalated() -> None:
    ticks = [
        _tick(
            0,
            signals=(SignalObservation("retry", 0.6),),
            fused=0.6,
            action=Action.ESCALATE,
            reason="retry: 3 failures",
        )
    ]
    report = build_session_report(
        run_id="run-esc", status="completed", ticks=ticks, events_recorded=1
    )

    assert report.outcome is SessionOutcome.ESCALATED
    assert report.severest_emitted is Action.ESCALATE


def test_a_completed_run_that_emitted_a_warn_never_claims_it_emitted_nothing() -> None:
    # Regression: `completed` was described as "Pathlayer emitted nothing", right
    # above the actions section listing the WARN. Only ESCALATE and STOP change
    # the outcome, so a run warned into completion lands here.
    ticks = [
        _tick(
            0,
            signals=(SignalObservation("loop", 0.5, "2x repeat"),),
            fused=0.5,
            action=Action.WARN,
            reason="loop: 2x repeat",
        )
    ]
    report = build_session_report(
        run_id="run-warned", status="completed", ticks=ticks, events_recorded=1
    )

    assert report.outcome is SessionOutcome.COMPLETED
    assert report.severest_emitted is Action.WARN
    text = report.format()
    assert "emitted nothing" not in text
    assert "never halted it" in text
    assert "event 0 (t=0.0): WARN -- loop: 2x repeat" in text


def test_a_run_nobody_saw_finish_is_never_reported_as_completed() -> None:
    report = build_session_report(
        run_id="run-cut", status="interrupted", ticks=[_tick(0)], events_recorded=1
    )
    assert report.outcome is SessionOutcome.INTERRUPTED
    assert "never reached its end" in report.format()

    # A run still in flight is not finished either: `started` must not be filed
    # as a clean completion just because no end was recorded.
    live = build_session_report(run_id="run-live", status="started", ticks=[], events_recorded=0)
    assert live.outcome is SessionOutcome.INTERRUPTED


def test_a_stop_recorded_only_in_the_manifest_is_still_reported() -> None:
    # The journal was truncated before the STOP tick was written; the run's own
    # record still says it stopped, so the report says so too.
    report = build_session_report(
        run_id="run-truncated", status="completed", ticks=[], events_recorded=7, stopped=True
    )
    assert report.outcome is SessionOutcome.STOPPED
    assert report.stop_reason is None


def test_high_confidence_that_never_acted_is_the_headline() -> None:
    ticks = [
        _tick(0, signals=(SignalObservation("loop", 0.3),), fused=0.3),
        _tick(1, signals=(SignalObservation("loop", 0.85, "4x repeat"),), fused=0.85),
        _tick(2, signals=(SignalObservation("loop", 0.5),), fused=0.5),
    ]
    report = build_session_report(
        run_id="run-near-miss", status="completed", ticks=ticks, events_recorded=3
    )

    data = report.to_dict()
    assert data["peak_never_acted"] is True
    assert (data["peak_fused"], data["peak_fused_index"]) == (0.85, 1)
    assert data["detectors"][0]["peak_confidence"] == 0.85
    assert data["detectors"][0]["peak_index"] == 1
    text = report.format()
    assert "peaked at 0.85 on event 1" in text
    assert "never reached an action tier" in text


def test_a_sustained_action_is_one_entry_not_one_per_event() -> None:
    ticks = [_tick(i, fused=0.95, action=Action.STOP, reason="loop: repeat") for i in range(3)]
    report = build_session_report(
        run_id="run-sustained", status="completed", ticks=ticks, events_recorded=3, stopped=True
    )
    assert [action.index for action in report.actions] == [0]


def test_a_dry_run_decision_is_reported_as_never_applied() -> None:
    ticks = [
        _tick(
            0,
            signals=(SignalObservation("loop", 1.0, "5x repeat"),),
            fused=1.0,
            action=Action.STOP,
            reason="loop: 5x repeat",
            dry_run=True,
        )
    ]
    report = build_session_report(
        run_id="run-dry", status="completed", ticks=ticks, events_recorded=1
    )

    # Nothing was emitted, so the run really did complete -- but the would-be
    # STOP is visible, and it is not mistaken for an intervention.
    assert report.outcome is SessionOutcome.COMPLETED
    assert report.emitted_actions == ()
    assert report.stop_reason is None
    assert "dry-run: never applied" in report.format()


def test_a_journal_without_detector_data_is_unknown_not_silent() -> None:
    report = build_session_report(
        run_id="run-legacy", status="completed", ticks=[], events_recorded=5
    )
    text = report.format()
    assert "no tool call was analyzed" in text
    assert "fused score: none recorded" in text

    partial = build_session_report(
        run_id="run-legacy",
        status="completed",
        ticks=[TickRecord(index=0, ts=0.0)],
        events_recorded=5,
    )
    assert "none recorded" in partial.format()


def test_detectors_are_ordered_by_peak_then_name() -> None:
    ticks = [
        _tick(
            0,
            signals=(
                SignalObservation("retry", 0.0),
                SignalObservation("budget", 0.2),
                SignalObservation("loop", 0.9),
                SignalObservation("no_progress", 0.0),
            ),
        )
    ]
    report = build_session_report(run_id="r", status="completed", ticks=ticks, events_recorded=1)
    assert [activity.name for activity in report.detectors] == [
        "loop",
        "budget",
        "no_progress",
        "retry",
    ]


def test_a_small_reading_never_prints_as_silent() -> None:
    # 0.004 rounded to two decimals is 0.00, which would contradict the very
    # next words ("non-zero on 1 of 1 tick(s)").
    ticks = [_tick(0, signals=(SignalObservation("budget", 0.004, "1% of budget"),), fused=0.004)]
    report = build_session_report(run_id="r", status="completed", ticks=ticks, events_recorded=1)
    text = report.format()
    assert "peak <0.01" in text
    assert "1 wired, 1 reacted" in text


def test_an_unusable_confidence_is_reported_as_misbehaving_not_as_silence() -> None:
    # Regression: `NaN > 0.0` is false, so a detector emitting NaN was counted as
    # inactive and printed as "silent on all N tick(s)" -- the report claiming a
    # reading it never got, about the exact case it exists to surface.
    ticks = [_tick(0, signals=(SignalObservation("custom", float("nan")),))]
    report = build_session_report(run_id="r", status="completed", ticks=ticks, events_recorded=1)

    activity = report.detectors[0]
    assert (activity.invalid_ticks, activity.active_ticks) == (1, 0)
    assert report.to_dict()["detectors"][0]["invalid_ticks"] == 1
    text = report.format()
    assert "silent on all" not in text
    assert "1 unusable reading(s) (NaN or out of range) over 1 tick(s)" in text


def test_an_unusable_confidence_does_not_hide_the_peak_that_followed_it() -> None:
    # Regression: NaN loses every comparison, so the peak stayed pinned on the
    # first unusable reading and the run's real high point disappeared.
    ticks = [
        _tick(0, signals=(SignalObservation("custom", float("nan")),)),
        _tick(1, signals=(SignalObservation("custom", 0.94, "5x repeat"),)),
    ]
    report = build_session_report(run_id="r", status="completed", ticks=ticks, events_recorded=2)

    activity = report.detectors[0]
    assert activity.peak_confidence == pytest.approx(0.94)
    assert (activity.peak_index, activity.peak_reason) == (1, "5x repeat")
    assert (activity.invalid_ticks, activity.active_ticks) == (1, 1)


def test_ticks_are_aggregated_in_index_order_whatever_their_arrival_order() -> None:
    ticks = [_tick(2, fused=0.2), _tick(0, fused=0.9, action=Action.STOP), _tick(1, fused=0.1)]
    report = build_session_report(
        run_id="r", status="completed", ticks=ticks, events_recorded=3, stopped=True
    )
    assert [action.index for action in report.actions] == [0]
    assert report.peak_fused_index == 0


# --------------------------------------------------------------------------- #
# CostBreakdown: an absent measurement is never a $0.00 result
# --------------------------------------------------------------------------- #


def test_an_unmeasured_cost_is_reported_as_unmeasured() -> None:
    cost = CostBreakdown(captured_usd=0.0, priced_events=0, source="no cost source is wired")
    assert cost.measured is False
    assert cost.total_usd is None
    assert "not measured" in cost.describe()
    assert "$0.00" not in cost.describe()

    report = build_session_report(
        run_id="r", status="completed", ticks=[], events_recorded=2, cost=cost
    )
    assert report.to_dict()["cost"]["total_usd"] is None
    assert "cost: not measured" in report.format()


def test_a_measured_cost_shows_every_component_and_its_provenance() -> None:
    cost = CostBreakdown(
        captured_usd=1.5,
        priced_events=3,
        estimated_usd=0.25,
        baseline_usd=2.0,
        source="captured from LLM token usage",
    )
    assert cost.measured is True
    assert cost.total_usd == pytest.approx(3.75)
    text = cost.describe()
    assert "$3.7500 effective" in text
    assert "$1.5000 captured over 3 priced event(s)" in text
    assert "$0.2500 estimated" in text
    assert "$2.0000 configured baseline" in text


def test_every_component_of_the_printed_total_is_printed_with_it() -> None:
    # Regression: the captured line was conditioned on `priced_events`, so a
    # captured figure recorded without a count was silently left out of the sum
    # printed beside the total it is part of.
    cost = CostBreakdown(captured_usd=1.0, estimated_usd=0.5)
    assert cost.total_usd == pytest.approx(1.5)

    text = cost.describe()
    assert "$1.5000 effective" in text
    assert "$1.0000 captured" in text
    assert "priced event(s)" not in text


def test_cross_run_spend_is_never_added_to_the_effective_total() -> None:
    cost = CostBreakdown(captured_usd=1.0, priced_events=1, carried_over_usd=None)
    assert cost.to_dict()["carried_over_usd"] is None
    # A store that restores a run by replaying its journal recounts those dollars
    # into the captured/estimated components, so the reserved field must stay out
    # of the effective total.
    assert CostBreakdown(
        captured_usd=1.0, priced_events=1, carried_over_usd=99.0
    ).total_usd == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Recorded runs: the durable chronology in decisions.jsonl
# --------------------------------------------------------------------------- #


def test_every_analyzed_decision_carries_its_tick_even_without_a_stop(tmp_path: Path) -> None:
    # Before OBS-2 the signals only appeared once a post-mortem existed, so a run
    # that never stopped left no detector evidence at all.
    _record_run(
        tmp_path,
        "run-quiet",
        [_tool(name, "run-quiet", name, path=f"{name}.py") for name in ("read", "edit", "test")],
    )

    decisions = _decisions(_run(tmp_path, "run-quiet"))
    assert len(decisions) == 3
    for decision in decisions:
        assert decision["analyzed"] is True
        assert decision["arbitrated_action"] == "NONE"
        assert decision["fused"] == 0.0
        assert decision["dry_run"] is False
        assert {signal["name"] for signal in decision["signals"]} == {
            "loop",
            "retry",
            "no_progress",
        }
    # Existing columns are preserved byte for byte.
    assert [decision["event_index"] for decision in decisions] == [0, 1, 2]
    assert decisions[0]["tool"] == "read"
    assert decisions[0]["action"] == "NONE"


def test_events_past_a_passive_stop_claim_no_detector_reading(tmp_path: Path) -> None:
    _record_run(
        tmp_path,
        "run-stop",
        [*_looping("run-stop"), _tool("edit", "run-stop", "late", path="b.py")],
    )

    decisions = _decisions(_run(tmp_path, "run-stop"))
    stop_at = next(i for i, d in enumerate(decisions) if d["action"] == "STOP")
    assert decisions[stop_at]["signals"]
    assert decisions[stop_at]["reason"].startswith("loop")
    for decision in decisions[stop_at + 1 :]:
        assert decision["analyzed"] is False
        assert "signals" not in decision
        assert "fused" not in decision


def test_tick_columns_do_not_overwrite_the_emitted_action() -> None:
    # In dry-run the arbitrated action was never emitted; `action` must keep
    # meaning "what was applied" and the arbitrated one gets its own column.
    tick = TickRecord(index=0, ts=0.0, action=Action.STOP, dry_run=True)
    columns = tick_columns(tick)
    assert "action" not in columns
    assert columns["arbitrated_action"] == "STOP"
    assert columns["dry_run"] is True


# --------------------------------------------------------------------------- #
# report.json + reconstruction from the run directory
# --------------------------------------------------------------------------- #


def test_a_completed_run_gets_a_report_matching_its_directory(tmp_path: Path) -> None:
    _record_run(tmp_path, "run-clean", [_tool("read", "run-clean", "c1")])

    run = _run(tmp_path, "run-clean")
    stored = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert stored == build_run_report(run).to_dict()
    assert stored["outcome"] == "completed"
    assert stored["run_id"] == "run-clean"
    assert stored["session_key"] == "sess-1"
    assert stored["duration_seconds"] == pytest.approx(4.2)


def test_a_stopped_run_report_names_the_detector_that_halted_it(tmp_path: Path) -> None:
    _record_run(tmp_path, "run-stop", _looping("run-stop"))

    report = build_run_report(_run(tmp_path, "run-stop"))
    assert report.outcome is SessionOutcome.STOPPED
    assert report.stop_reason is not None
    assert report.stop_reason.startswith("loop")
    assert [action.action for action in report.actions] == [Action.STOP]
    peaks = {activity.name: activity.peak_confidence for activity in report.detectors}
    assert peaks["loop"] > 0.9
    assert peaks["retry"] == 0.0


def test_an_interrupted_run_is_reported_at_the_startup_sweep(tmp_path: Path) -> None:
    _record_run(tmp_path, "run-cut", [_tool("read", "run-cut", "c1")], finish=False)
    run = _run(tmp_path, "run-cut")
    assert not run.report_path.exists()

    OpenClawRunRecorder(tmp_path).startup()

    stored = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert stored["outcome"] == "interrupted"
    assert stored["status"] == "interrupted"
    assert any("never reached agent_end" in note for note in stored["notes"])
    assert stored["cost"]["total_usd"] is None


def test_an_unreportable_run_does_not_abort_the_startup_sweep(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The report only summarises data that is already durable, so one unreadable
    # journal must degrade to a logged warning rather than take down the sweep --
    # and the other unfinished run must still be swept and reported.
    _record_run(tmp_path, "run-torn", [_tool("read", "run-torn", "c1")], finish=False)
    _record_run(tmp_path, "run-ok", [_tool("read", "run-ok", "c1")], finish=False)
    torn = _run(tmp_path, "run-torn")
    with torn.decisions_path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")

    with caplog.at_level(logging.WARNING):
        OpenClawRunRecorder(tmp_path).startup()

    assert torn.read_manifest()["status"] == "interrupted"
    assert not torn.report_path.exists()
    assert "could not write the session report" in caplog.text
    assert json.loads(_run(tmp_path, "run-ok").report_path.read_text(encoding="utf-8"))


def test_structurally_invalid_signals_do_not_abort_the_startup_sweep(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _record_run(tmp_path, "run-bad-shape", [_tool("read", "run-bad-shape", "c1")], finish=False)
    _record_run(tmp_path, "run-good-shape", [_tool("read", "run-good-shape", "c1")], finish=False)
    malformed = _run(tmp_path, "run-bad-shape")
    records = _decisions(malformed)
    records[0]["signals"] = [1]
    malformed.decisions_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING):
        OpenClawRunRecorder(tmp_path).startup()

    assert malformed.read_manifest()["status"] == "interrupted"
    assert not malformed.report_path.exists()
    assert "could not write the session report" in caplog.text
    assert _run(tmp_path, "run-good-shape").report_path.is_file()


def test_a_run_with_no_cost_source_says_so_instead_of_zero(tmp_path: Path) -> None:
    _record_run(tmp_path, "run-free", [_tool("read", "run-free", "c1")])

    run = _run(tmp_path, "run-free")
    assert run.read_manifest()[COST_BUDGET_STATUS_KEY] == COST_BUDGET_DISABLED
    report = build_run_report(run)
    assert report.cost.measured is False
    assert report.cost.total_usd is None
    # None, not 0.0: the trace was readable and priced nothing, which is an
    # absence of measurement rather than a measured zero.
    assert report.cost.captured_usd is None
    assert report.cost.priced_events == 0
    assert "$0.00" not in report.format()
    assert "total-token usage" in report.cost.source
    assert "budget detector was not enabled" in report.cost.source
    assert "cost: not measured" in report.format()


def test_bud3_estimates_are_reported_as_estimates_not_as_captured_cost(tmp_path: Path) -> None:
    config = PathlayerConfig(
        detectors=DetectorsConfig(
            budget=BudgetConfig(
                enabled=True, max_cost_usd=50.0, baseline_usd=1.0, default_cost_usd=0.01
            )
        )
    )
    _record_run(
        tmp_path, "run-est", [_tool("read", "run-est", f"c{i}") for i in range(3)], config=config
    )

    run = _run(tmp_path, "run-est")
    assert run.read_manifest()[COST_BUDGET_STATUS_KEY] == COST_BUDGET_ENABLED
    report = build_run_report(run)
    assert report.cost.captured_usd is None
    assert report.cost.priced_events == 0
    assert report.cost.estimated_usd == pytest.approx(0.03)
    assert report.cost.baseline_usd == pytest.approx(1.0)
    assert report.cost.total_usd == pytest.approx(1.03)
    # A measured total must not drag a fictitious captured component along with
    # it. (The `budget` detector's own reason line does mention its internal
    # $0.00 captured component; that is the detector's diagnostic, not the
    # report's claim about what was measured.)
    cost_line = next(line for line in report.format().splitlines() if line.startswith("cost:"))
    assert "estimated" in cost_line
    assert "captured" not in cost_line


def test_custom_pipeline_factory_keeps_budget_provenance_unknown(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(
        tmp_path,
        pipeline_factory=lambda: Pipeline(
            [BudgetDetector(max_cost_usd=10.0, default_cost_usd=0.25)]
        ),
    )
    recorder.process_payload(_before("run-custom-budget"))
    recorder.process_payload(_tool("read", "run-custom-budget", "c1"))
    recorder.process_payload(_end("run-custom-budget"))

    run = _run(tmp_path, "run-custom-budget")
    assert run.read_manifest()[COST_BUDGET_STATUS_KEY] == COST_BUDGET_UNKNOWN
    report = build_run_report(run)
    assert report.cost.estimated_usd is None
    assert "custom pipeline factory" in report.cost.source
    assert "was not enabled" not in report.cost.source


def test_captured_event_costs_are_summed_back_from_the_trace(tmp_path: Path) -> None:
    # The captured component is recomputed from trace.json, so it works for any
    # adapter that does price its events -- not only for the OpenClaw path.
    store = RunStore(tmp_path)
    run = store.open_or_create(
        "run-priced",
        manifest={"schema_version": 1, "run_id": "run-priced", "status": "completed"},
    )
    atomic_write_json(
        run.manifest_path,
        {
            "schema_version": 1,
            "run_id": "run-priced",
            "dir_name": run.dir_name,
            "status": "completed",
            "started_at": "2026-08-10T09:00:00+00:00",
            "ended_at": "2026-08-10T09:00:30+00:00",
            "stopped": False,
        },
    )
    atomic_write_json(
        run.trace_path,
        {
            "label": "unknown",
            "events": [
                {"tool": "gpt-4o", "kind": EventKind.LLM_CALL.value, "cost_usd": 0.25},
                {"tool": "read"},
            ],
        },
    )

    report = build_run_report(run)
    assert report.cost.captured_usd == pytest.approx(0.25)
    assert report.cost.priced_events == 1
    assert report.cost.total_usd == pytest.approx(0.25)
    assert report.duration_seconds == pytest.approx(30.0)


def test_a_budget_estimate_of_zero_is_a_measured_zero_not_an_absence(tmp_path: Path) -> None:
    # Regression: `measured` used to require every component to be non-zero, so a
    # budget detector that ran over tools with no configured price -- a perfectly
    # ordinary configuration -- had its computed $0.00 reported as "not measured".
    # Refusing to report a zero someone computed is the mirror image of inventing
    # one nobody did.
    config = PathlayerConfig(
        detectors=DetectorsConfig(budget=BudgetConfig(enabled=True, max_cost_usd=50.0))
    )
    _record_run(tmp_path, "run-zero", [_tool("read", "run-zero", "c1")], config=config)

    run = _run(tmp_path, "run-zero")
    assert run.read_manifest()["cost_estimated_usd"] == 0.0
    report = build_run_report(run)
    assert report.cost.estimated_usd == 0.0
    assert report.cost.measured is True
    assert report.cost.total_usd == 0.0
    assert "not measured" not in report.cost.describe()
    # And the measured zero is explained, so it cannot be read as "free".
    assert "computed $0.00" in report.cost.source
    assert "$0.0000 effective" in report.format()


def test_an_interrupted_run_keeps_the_cost_it_had_already_accumulated(tmp_path: Path) -> None:
    # Regression: the cost components were only persisted at `agent_end`, so a run
    # cut before it left a manifest with no cost keys at all -- and its report then
    # asserted the budget detector had not been enabled, which was simply false.
    config = PathlayerConfig(
        detectors=DetectorsConfig(
            budget=BudgetConfig(
                enabled=True, max_cost_usd=50.0, baseline_usd=1.0, default_cost_usd=0.01
            )
        )
    )
    _record_run(
        tmp_path,
        "run-cut-cost",
        [_tool("read", "run-cut-cost", f"c{i}") for i in range(3)],
        config=config,
        finish=False,
    )
    run = _run(tmp_path, "run-cut-cost")
    assert run.read_manifest()["cost_estimated_usd"] == pytest.approx(0.03)

    OpenClawRunRecorder(tmp_path).startup()

    report = build_run_report(_run(tmp_path, "run-cut-cost"))
    assert report.outcome is SessionOutcome.INTERRUPTED
    assert report.cost.estimated_usd == pytest.approx(0.03)
    assert report.cost.baseline_usd == pytest.approx(1.0)
    assert report.cost.total_usd == pytest.approx(1.03)
    assert "was not enabled" not in report.cost.source
    stored = json.loads(_run(tmp_path, "run-cut-cost").report_path.read_text(encoding="utf-8"))
    assert stored["cost"]["estimated_usd"] == pytest.approx(0.03)


def test_the_running_accumulator_matches_a_single_batch_fold(tmp_path: Path) -> None:
    # The estimate is advanced event by event so it is durable before the end;
    # that incremental path must equal the batch fold it replaced.
    config = PathlayerConfig(
        detectors=DetectorsConfig(
            budget=BudgetConfig(enabled=True, max_cost_usd=50.0, default_cost_usd=0.02)
        )
    )
    tools = [_tool("read", "run-fold", f"c{i}") for i in range(4)]
    _record_run(tmp_path, "run-fold", tools, config=config)

    run = _run(tmp_path, "run-fold")
    events = list(load_trace(run.trace_path).events)
    detector = BudgetDetector(max_cost_usd=50.0, default_cost_usd=0.02)
    assert run.read_manifest()["cost_estimated_usd"] == pytest.approx(
        detector.accumulate(events).estimated_usd
    )


def test_an_early_cut_can_report_a_proven_disabled_budget_detector(tmp_path: Path) -> None:
    # The recorder now persists budget provenance before the first tool call, so
    # an early cut can distinguish a proven disabled detector from missing data.
    _record_run(tmp_path, "run-early-cut", [], finish=False)
    OpenClawRunRecorder(tmp_path).startup()

    run = _run(tmp_path, "run-early-cut")
    assert run.read_manifest()[COST_BUDGET_STATUS_KEY] == COST_BUDGET_DISABLED
    report = build_run_report(run)
    assert report.outcome is SessionOutcome.INTERRUPTED
    assert report.cost.estimated_usd is None
    assert report.cost.measured is False
    assert "was not enabled" in report.cost.source
    assert "cost: not measured" in report.format()


def test_a_pre_obs2_journal_is_reported_as_unknown_not_as_silent(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run = store.open_or_create(
        "run-legacy",
        manifest={"schema_version": 1, "run_id": "run-legacy", "status": "completed"},
    )
    atomic_write_json(
        run.manifest_path,
        {
            "schema_version": 1,
            "run_id": "run-legacy",
            "dir_name": run.dir_name,
            "status": "completed",
            "stopped": False,
        },
    )
    append_jsonl(
        run.decisions_path,
        {
            "sequence": 2,
            "event_index": 0,
            "event_ts": 0.0,
            "tool": "read",
            "action": "NONE",
            "stopped": False,
            "analyzed": True,
        },
    )

    report = build_run_report(run)
    assert report.detectors == ()
    assert report.events_recorded == 1
    # The line was analyzed, so the report says the confidences are unknown --
    # not that no detector ever ran, which the journal does not support either.
    assert report.events_analyzed == 1
    assert any("no detector confidence" in note for note in report.notes)
    assert "unknown, not silent" in report.format()
    assert "no detector ever ran" not in report.format()


def test_a_dry_run_recording_reports_the_action_it_never_applied(tmp_path: Path) -> None:
    config = PathlayerConfig(
        pipeline=PipelineSection(dry_run=True), policy=PolicySection(stop_threshold=0.0)
    )
    _record_run(tmp_path, "run-dry", [_tool("read", "run-dry", "c1")], config=config)

    report = build_run_report(_run(tmp_path, "run-dry"))
    assert report.dry_run is True
    assert report.outcome is SessionOutcome.COMPLETED
    assert [action.emitted for action in report.actions] == [False]
    assert "dry-run: never applied" in report.format()


def test_an_escalating_run_is_reported_as_escalated(tmp_path: Path) -> None:
    config = PathlayerConfig(policy=PolicySection(tier_thresholds={"escalate": 0.0}))
    _record_run(
        tmp_path,
        "run-esc",
        [_tool(name, "run-esc", name, path=f"{name}.py") for name in ("read", "edit")],
        config=config,
    )

    report = build_run_report(_run(tmp_path, "run-esc"))
    assert report.outcome is SessionOutcome.ESCALATED
    assert report.severest_emitted is Action.ESCALATE
    assert "outcome: escalated" in report.format()


def test_a_near_miss_recording_shows_the_peak_that_never_acted(tmp_path: Path) -> None:
    # stop_threshold 1.0 is unreachable for the loop detector, so a genuinely
    # repetitive run climbs high and still emits nothing.
    config = PathlayerConfig(policy=PolicySection(stop_threshold=1.0))
    _record_run(tmp_path, "run-near", _looping("run-near"), config=config)

    report = build_run_report(_run(tmp_path, "run-near"))
    assert report.outcome is SessionOutcome.COMPLETED
    assert report.actions == ()
    assert report.peak_never_acted is True
    assert report.peak_fused is not None and report.peak_fused > 0.9
    assert "never reached an action tier" in report.format()


def test_write_run_report_uses_the_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[str] = []
    real_replace = os.replace

    def _spy(src: Any, dst: Any) -> None:
        replaced.append(Path(dst).name)
        real_replace(src, dst)

    _record_run(tmp_path, "run-atomic", [_tool("read", "run-atomic", "c1")])
    monkeypatch.setattr(os, "replace", _spy)
    write_run_report(_run(tmp_path, "run-atomic"))
    assert "report.json" in replaced


def test_a_corrupt_decision_journal_is_refused_loudly(tmp_path: Path) -> None:
    _record_run(tmp_path, "run-bad", [_tool("read", "run-bad", "c1")])
    run = _run(tmp_path, "run-bad")
    with run.decisions_path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")

    with pytest.raises(ValueError, match="invalid JSON"):
        build_run_report(run)


def test_a_journal_torn_mid_line_is_still_reported_with_a_note(tmp_path: Path) -> None:
    # Regression: `decisions.jsonl` is appended line by line (write + flush +
    # fsync), so a killed process typically leaves its *last* line half written.
    # Refusing the whole file made exactly the interrupted run this module exists
    # for unreportable -- and the startup sweep never revisits a terminal status,
    # so nothing would ever report it afterwards.
    _record_run(
        tmp_path,
        "run-torn",
        [_tool("read", "run-torn", f"c{i}") for i in range(2)],
        finish=False,
    )
    run = _run(tmp_path, "run-torn")
    with run.decisions_path.open("a", encoding="utf-8") as handle:
        handle.write('{"sequence": 9, "event_index": 2, "event_ts": 2.0, "too')

    report = build_run_report(run)
    assert report.events_analyzed == 2
    assert any("truncated" in note for note in report.notes)


def test_a_journal_corrupted_before_its_last_line_is_still_refused(tmp_path: Path) -> None:
    # A garbled line followed by a complete one is not a torn write: the process
    # kept writing afterwards, so the damage is corruption and summarising around
    # it would hide it.
    _record_run(tmp_path, "run-mid", [_tool("read", "run-mid", "c1")], finish=False)
    run = _run(tmp_path, "run-mid")
    with run.decisions_path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    append_jsonl(run.decisions_path, {"event_index": 2, "event_ts": 2.0, "analyzed": False})

    with pytest.raises(ValueError, match="invalid JSON"):
        build_run_report(run)


def test_an_interrupted_run_never_reports_the_listener_downtime_as_its_duration(
    tmp_path: Path,
) -> None:
    # Regression: `ended_at` on an interrupted run is stamped by the startup sweep
    # that noticed the interruption, which can be a day later. Subtracting the
    # timestamps reported that downtime as the run's own length.
    _record_run(tmp_path, "run-late", [_tool("read", "run-late", "c1")], finish=False)
    run = _run(tmp_path, "run-late")
    run.update_manifest(
        status="interrupted",
        started_at="2026-08-10T09:00:00+00:00",
        ended_at="2026-08-11T09:00:00+00:00",
    )

    report = build_run_report(run)
    assert report.duration_seconds is None
    assert any("when the interruption was noticed" in note for note in report.notes)
    text = report.format()
    assert "duration unknown" in text
    assert "86400" not in text


def test_an_analyzed_decision_line_without_its_position_is_refused(tmp_path: Path) -> None:
    # Defaulting the index and the timestamp to 0 / 0.0 fabricated a position for
    # a line that recorded none, and the report then printed it as a measurement.
    _record_run(tmp_path, "run-nopos", [_tool("read", "run-nopos", "c1")])
    run = _run(tmp_path, "run-nopos")
    append_jsonl(run.decisions_path, {"analyzed": True, "arbitrated_action": "NONE", "fused": 0.0})

    with pytest.raises(ValueError, match="event_index"):
        build_run_report(run)


def test_an_unknown_arbitrated_action_is_refused(tmp_path: Path) -> None:
    _record_run(tmp_path, "run-weird", [_tool("read", "run-weird", "c1")])
    run = _run(tmp_path, "run-weird")
    append_jsonl(
        run.decisions_path,
        {"event_index": 1, "event_ts": 1.0, "analyzed": True, "arbitrated_action": "TELEPORT"},
    )

    with pytest.raises(ValueError, match="unknown arbitrated action"):
        build_run_report(run)


# --------------------------------------------------------------------------- #
# Backward compatibility: journals written before OBS-2
# --------------------------------------------------------------------------- #


def _legacy_run(runs_root: Path, run_id: str) -> Any:
    """A run directory in the pre-OBS-2 journal format, as ADP-2 wrote it.

    The previous recorder wrote ``signals`` and ``reason`` too, copied from the
    post-mortem, but only from the STOP onward -- and never ``fused`` or
    ``arbitrated_action``. Those two columns are therefore the only reliable way
    to tell a legacy line from an OBS-2 one.
    """
    store = RunStore(runs_root)
    run = store.open_or_create(run_id, manifest={"schema_version": 1, "run_id": run_id})
    atomic_write_json(
        run.manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "dir_name": run.dir_name,
            "status": "completed",
            "started_at": "2026-08-10T09:00:00+00:00",
            "ended_at": "2026-08-10T09:00:30+00:00",
            "stopped": True,
        },
    )
    for index, (tool, action) in enumerate(
        [("read", "NONE"), ("read", "WARN"), ("read", "STOP"), ("read", "STOP")]
    ):
        analyzed = action != "STOP" or index == 2
        line: dict[str, Any] = {
            "sequence": index + 2,
            "received_at": "2026-08-10T09:00:10+00:00",
            "event_index": index,
            "event_ts": float(index),
            "tool": tool,
            "action": action,
            "stopped": index >= 2,
            "analyzed": analyzed,
        }
        if index == 2:
            line["signals"] = [
                {"name": "loop", "confidence": 0.94, "reason": "5x repeat of 1-action block"},
                {"name": "retry", "confidence": 0.0, "reason": ""},
            ]
            line["reason"] = "loop: 5x repeat of 1-action block"
        append_jsonl(run.decisions_path, line)
    atomic_write_json(
        run.trace_path,
        {"label": "unknown", "events": [{"tool": "read"} for _ in range(4)]},
    )
    return run


def test_a_legacy_journal_keeps_its_stop_its_reason_and_its_actions(tmp_path: Path) -> None:
    # Regression: a legacy line carries `signals` but no `fused` and no
    # `arbitrated_action`. Reading it as a complete OBS-2 tick flattened its
    # action to NONE, so the STOP, its reason and the WARN before it all
    # disappeared from the report of a run that is already on disk.
    report = build_run_report(_legacy_run(tmp_path, "run-legacy"))

    assert report.outcome is SessionOutcome.STOPPED
    assert [(action.action, action.index) for action in report.actions] == [
        (Action.WARN, 1),
        (Action.STOP, 2),
    ]
    assert report.stop_reason == "loop: 5x repeat of 1-action block"
    assert report.severest_emitted is Action.STOP
    text = report.format()
    assert "stop reason: loop: 5x repeat of 1-action block" in text
    assert "event 2 (t=2.0): STOP" in text


def test_a_legacy_journal_reports_an_unknown_fused_score_not_a_zero(tmp_path: Path) -> None:
    # Regression: an absent fused column used to be read back as 0.0, which both
    # understated the run's real peak and printed a fabricated measurement.
    report = build_run_report(_legacy_run(tmp_path, "run-legacy"))

    assert report.peak_fused is None
    assert report.to_dict()["peak_fused"] is None
    assert report.peak_never_acted is False
    text = report.format()
    assert "fused score: none recorded for this run" in text
    assert "peaked at 0.00" not in text
    assert "peak n/a" in report.summary()
    # The detector confidences the legacy journal does carry are still reported,
    # and the gap is named rather than passed off as detector silence.
    peaks = {activity.name: activity.peak_confidence for activity in report.detectors}
    assert peaks["loop"] == pytest.approx(0.94)
    assert any("journaled before OBS-2" in note for note in report.notes)
    assert any("unknown, not silent" in note for note in report.notes)
    # The old listener never persisted cost components either, so the absence of
    # an estimate here says nothing about whether the budget detector was on.
    assert "was not enabled" not in report.cost.source
    assert "indeterminate" in report.cost.source


def test_a_legacy_journal_does_not_claim_its_analyzed_events_were_never_analyzed(
    tmp_path: Path,
) -> None:
    # A legacy analyzed line still proves the event went through the detectors.
    # Reporting 0 analyzed events for it would deny a fact the journal records.
    report = build_run_report(_legacy_run(tmp_path, "run-legacy"))
    assert report.events_analyzed == 3
    assert report.events_recorded == 4
    assert report.ticks_with_signals == 1
    assert "3 of 3 analyzed line(s) were journaled before OBS-2" in report.notes[0]


def test_an_obs2_journal_is_not_mistaken_for_a_legacy_one(tmp_path: Path) -> None:
    _record_run(tmp_path, "run-modern", _looping("run-modern"))

    report = build_run_report(_run(tmp_path, "run-modern"))
    assert report.notes == () or all("before OBS-2" not in note for note in report.notes)
    assert report.peak_fused is not None and report.peak_fused > 0.9


# --------------------------------------------------------------------------- #
# pathlayer-report CLI
# --------------------------------------------------------------------------- #


def _three_runs(tmp_path: Path) -> None:
    _record_run(tmp_path, "run-a", [_tool("read", "run-a", "c1")])
    _record_run(tmp_path, "run-b", _looping("run-b"))
    _record_run(tmp_path, "run-c", [_tool("edit", "run-c", "c1", path="b.py")])
    # Deterministic chronological order, independent of filesystem listing.
    for index, run_id in enumerate(("run-a", "run-b", "run-c")):
        _run(tmp_path, run_id).update_manifest(started_at=f"2026-08-1{index}T09:00:00+00:00")


def test_cli_shows_one_named_run_in_full(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _three_runs(tmp_path)
    assert main(["run-b", "--runs-root", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "run run-b" in out
    assert "outcome: stopped" in out
    assert "stop reason: loop" in out
    assert "run-a" not in out


def test_cli_last_selects_the_most_recently_started_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _three_runs(tmp_path)
    assert main(["--last", "--runs-root", str(tmp_path)]) == 0
    assert "run run-c" in capsys.readouterr().out


def test_cli_today_lists_only_todays_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _record_run(tmp_path, "run-now", [_tool("read", "run-now", "c1")])
    _record_run(tmp_path, "run-old", [_tool("read", "run-old", "c1")])
    _run(tmp_path, "run-old").update_manifest(started_at="2020-01-01T09:00:00+00:00")

    assert main(["--today", "--runs-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "run-now" in out
    assert "run-old" not in out


def test_cli_all_lists_every_run_one_line_each_in_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _three_runs(tmp_path)
    assert main(["--all", "--runs-root", str(tmp_path)]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split()[1] for line in lines] == ["run-a", "run-b", "run-c"]
    assert "stopped" in lines[1]


def test_cli_json_mode_emits_the_structured_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _three_runs(tmp_path)
    assert main(["run-b", "--runs-root", str(tmp_path), "--json"]) == 0
    single = json.loads(capsys.readouterr().out)
    assert single["run_id"] == "run-b"
    assert single["outcome"] == "stopped"

    assert main(["--all", "--runs-root", str(tmp_path), "--json"]) == 0
    every = json.loads(capsys.readouterr().out)
    assert [entry["run_id"] for entry in every] == ["run-a", "run-b", "run-c"]


def test_cli_reads_the_runs_root_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _record_run(tmp_path, "run-env", [_tool("read", "run-env", "c1")])
    monkeypatch.setenv("PATHLAYER_RUNS_DIR", str(tmp_path))

    assert main(["run-env"]) == 0
    assert "run run-env" in capsys.readouterr().out


def test_cli_refuses_an_unknown_run_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["missing", "--runs-root", str(tmp_path)]) == 1
    assert "no recorded run" in capsys.readouterr().err


def test_cli_refuses_an_ambiguous_run_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = RunStore(tmp_path)
    for suffix in ("a", "b"):
        directory = store.open_or_create(
            f"same-id-{suffix}",
            manifest={"schema_version": 1, "run_id": f"same-id-{suffix}", "status": "completed"},
        )
        atomic_write_json(
            directory.manifest_path,
            {"schema_version": 1, "run_id": "same-id", "dir_name": directory.dir_name},
        )

    assert main(["same-id", "--runs-root", str(tmp_path)]) == 1
    assert "ambiguous" in capsys.readouterr().err


def test_cli_reports_no_run_at_all_as_an_error_for_last(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--last", "--runs-root", str(tmp_path)]) == 1
    assert "no recorded run" in capsys.readouterr().err


def test_cli_treats_an_empty_but_valid_selection_as_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--today", "--runs-root", str(tmp_path)]) == 0
    assert "no run matched" in capsys.readouterr().out


def test_cli_requires_exactly_one_selection_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--runs-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["run-a", "--all", "--runs-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["--all", "--last", "--runs-root", str(tmp_path)])


def test_cli_surfaces_a_broken_run_directory_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _record_run(tmp_path, "run-bad", [_tool("read", "run-bad", "c1")])
    run = _run(tmp_path, "run-bad")
    with run.decisions_path.open("a", encoding="utf-8") as handle:
        handle.write("{oops\n")

    assert main(["run-bad", "--runs-root", str(tmp_path)]) == 1
    assert "invalid JSON" in capsys.readouterr().err


def test_reports_survive_a_policy_that_acts_without_any_detector() -> None:
    # A custom policy can act with no detector wired at all; the report must
    # still name the action instead of asserting a detector drove it.
    class _AlwaysStop(Policy):
        def step(self, state: Any, signals: Any, evidence: Any = None) -> Any:
            return state, Action.STOP

    pipeline = Pipeline([], _AlwaysStop(PolicyConfig()))
    pipeline.process(ToolEvent(tool="read"))
    tick = _projected_tick(pipeline)
    assert tick is not None
    assert tick.action is Action.STOP
    assert tick.reason == "stop"
    assert tick.signals == ()

    report = build_session_report(
        run_id="r", status="completed", ticks=[tick], events_recorded=1, stopped=True
    )
    assert report.detectors == ()
    assert "stop reason: stop" in report.format()


@pytest.mark.parametrize("amount", [0.0, 0.25])
@pytest.mark.parametrize("manifest_mode", ["new", "legacy", "disabled"])
def test_adapter_estimates_are_reported_once(amount: float, manifest_mode: str) -> None:
    from pathlayer.adapters.openclaw_report import _build_cost, cost_manifest_fields
    from pathlayer.detectors.budget import BudgetDetector

    events = [
        ToolEvent(
            tool="model", cost_usd=amount, metadata={"cost_measurement": "estimated_total_tokens"}
        )
    ]
    detector = BudgetDetector(default_cost_usd=10.0)
    if manifest_mode == "new":
        manifest = cost_manifest_fields(detector, detector.accumulate(events))
    elif manifest_mode == "legacy":
        manifest = {"cost_estimated_usd": 0.1}
    else:
        manifest = {"cost_budget_status": "disabled"}
    cost = _build_cost(manifest, events, True, completed=True, legacy_journal=False)
    assert cost.captured_usd is None
    assert cost.priced_events == 0
    assert cost.estimated_usd == pytest.approx(amount + (0.1 if manifest_mode == "legacy" else 0))
    assert "estimated from total tokens" in cost.source
    assert "estimated" in cost.describe()


@pytest.mark.parametrize("enabled", [True, False])
def test_recorded_llm_estimate_report_retains_provenance(tmp_path: Path, enabled: bool) -> None:
    from pathlayer.config import PricingSection
    from pathlayer.pricing import ModelPrice

    config = PathlayerConfig(
        detectors=DetectorsConfig(budget=BudgetConfig(enabled=enabled, max_cost_usd=100.0)),
        pricing=PricingSection(models={"model": ModelPrice(1.0, 2.0)}),
    )
    payload = {
        "hook": "llm_output",
        "event": {"runId": "llm", "model": "model", "usage": {"total": 1000}},
        "ctx": _ctx("llm"),
    }
    _record_run(tmp_path, "llm", [payload], config=config)
    report = build_run_report(_run(tmp_path, "llm"))
    assert report.cost.captured_usd is None
    assert report.cost.estimated_usd == pytest.approx(0.0012)
    assert report.cost.total_usd == pytest.approx(0.0012)
    assert "estimated" in report.cost.describe()


def test_coding_diagnostic_names_progress_resource_and_intervention() -> None:
    spec = ProofSpec(
        schema_version=1,
        task_id="fix-login",
        proof_id="login-tests",
        scope_id="workspace",
        kind="checkpoint_set",
        required=("tests/test_login.py::test_login",),
    )
    spec_hash = compute_spec_hash(spec)
    state = {
        "schema_version": 1,
        "outcome": "changed",
        "records": [
            {
                "schema_version": 1,
                "tool_event_ref": "run-1",
                "phase": "before",
                "resource_type": "filesystem",
                "resource_id": "src/auth.py",
                "status": "observed",
                "raw_fingerprint": {"sha256": "before"},
                "canonical_fingerprint": {"sha256": "before"},
                "delta": None,
                "collector_version": "filesystem/v1",
            },
            {
                "schema_version": 1,
                "tool_event_ref": "run-1",
                "phase": "after",
                "resource_type": "filesystem",
                "resource_id": "src/auth.py",
                "status": "observed",
                "raw_fingerprint": {"sha256": "after"},
                "canonical_fingerprint": {"sha256": "after"},
                "delta": {"changed": True},
                "collector_version": "filesystem/v1",
            },
        ],
    }
    events = [
        ToolEvent(
            tool="edit",
            run_id="run-1",
            metadata={
                "proof_spec": {
                    "schema_version": spec.schema_version,
                    "task_id": spec.task_id,
                    "proof_id": spec.proof_id,
                    "scope_id": spec.scope_id,
                    "kind": spec.kind,
                    "required": list(spec.required),
                },
                "progress_proof": {
                    "spec_hash": spec_hash,
                    "sequence": 1,
                    "checkpoints": {"tests/test_login.py::test_login": True},
                },
                "pathlayer_state_evidence": state,
            },
        ),
        ToolEvent(
            tool="edit",
            run_id="run-2",
            metadata={
                "progress_proof": {
                    "spec_hash": spec_hash,
                    "sequence": 2,
                    "checkpoints": {"tests/test_login.py::test_login": True},
                }
            },
        ),
    ]

    diagnostic = build_coding_diagnostic(
        events,
        [_tick(0), _tick(1, action=Action.STOP, reason="no_progress: contract stalled")],
    )

    assert diagnostic.to_dict() == {
        "contract_state": "reached",
        "last_progress": "tests/test_login.py::test_login",
        "last_progress_event": 0,
        "state_resources": ["src/auth.py"],
        "state_outcome": "changed",
        "state_event": 0,
        "calls_since_progress": 1,
        "intervention": "no_progress",
        "intervention_event": 1,
        "intervention_action": "stop",
        "intervention_reason": "no_progress: contract stalled",
        "would_have_action": None,
        "would_have_reason": None,
        "would_have_event": None,
    }
    assert "sha256" not in diagnostic.format()
    assert "src/auth.py is changed" in diagnostic.format()


def test_coding_diagnostic_keeps_unavailable_distinct_from_unchanged() -> None:
    event = ToolEvent(
        tool="edit",
        run_id="run-1",
        metadata={
            "pathlayer_state_evidence": {
                "schema_version": 1,
                "outcome": "unavailable",
                "records": [
                    {
                        "schema_version": 1,
                        "tool_event_ref": "run-1",
                        "phase": phase,
                        "resource_type": "filesystem",
                        "resource_id": "src/auth.py",
                        "status": "unavailable",
                        "raw_fingerprint": None,
                        "canonical_fingerprint": None,
                        "delta": None,
                        "collector_version": "filesystem/v1",
                    }
                    for phase in ("before", "after")
                ],
            }
        },
    )

    diagnostic = build_coding_diagnostic([event], [_tick(0)])

    assert diagnostic.state_outcome == "unavailable"
    assert "unavailable" in diagnostic.format()
