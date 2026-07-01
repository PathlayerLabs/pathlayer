"""Dry-run mode (OBS-3): observe detectors and arbitration without intervening.

In dry-run all detectors fire and arbitration runs, but no action is emitted
(every returned action is ``NONE``) and the run never halts. The actions that
*would* have been taken are exposed via ``Pipeline.would_be_actions`` and logged
with a distinct ``would_be_action`` kind, recorded once per transition so a
sustained STOP does not flood the record.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from pathlayer import Pipeline, WouldBeAction
from pathlayer.detectors.base import Signal
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import ToolEvent
from pathlayer.observability import get_logger
from pathlayer.policies import Action, HysteresisState, Policy


def _loop_events(n: int) -> list[ToolEvent]:
    """``n`` identical tool calls -> the LoopDetector escalates to STOP."""
    return [ToolEvent(tool="read", args={"path": "a.py"}, ts=float(i)) for i in range(n)]


def _records(caplog: pytest.LogCaptureFixture, kind: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "pathlayer", {}).get("kind") == kind]


def test_dry_run_emits_nothing_and_never_halts() -> None:
    pipeline = Pipeline([LoopDetector()], dry_run=True)
    actions = pipeline.run(_loop_events(20))

    assert pipeline.dry_run is True
    # The whole run plays out (no early halt) and nothing reaches the agent.
    assert pipeline.stopped is False
    assert len(actions) == 20
    assert all(action is Action.NONE for action in actions)
    assert all(action is Action.NONE for action in pipeline.actions)
    assert pipeline.post_mortem is None


def test_dry_run_detectors_still_fire_and_record_would_be_stop() -> None:
    # A non-dry-run run STOPs at the first crossing; dry-run must reach the same
    # decision and record it as a would-be action, without halting.
    enforced = Pipeline([LoopDetector()])
    enforced.run(_loop_events(20))
    assert enforced.stopped is True
    first_stop_index = enforced.post_mortem.stopped_at_index  # type: ignore[union-attr]

    pipeline = Pipeline([LoopDetector()], dry_run=True)
    pipeline.run(_loop_events(20))

    (would_be,) = pipeline.would_be_actions
    assert would_be.action is Action.STOP
    assert would_be.index == first_stop_index
    assert would_be.ts == float(first_stop_index)
    assert would_be.reason.startswith("loop")
    assert would_be.signals  # the firing signals are captured for the report


def test_dry_run_records_one_entry_per_transition() -> None:
    # The STOP decision holds for every event after the crossing, but a sustained
    # action is recorded once (on transition), not once per event.
    pipeline = Pipeline([LoopDetector()], dry_run=True)
    pipeline.run(_loop_events(20))

    assert len(pipeline.would_be_actions) == 1
    assert isinstance(pipeline.would_be_actions[0], WouldBeAction)


def test_dry_run_logs_would_be_action_not_action(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("pathlayer.test.dryrun")
    pipeline = Pipeline([LoopDetector()], logger=logger, dry_run=True)

    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline.run(_loop_events(20))

    # Signals and decisions are still logged every event; only the suppressed
    # action surfaces as would_be_action, and no real "action" is ever emitted.
    assert _records(caplog, "signal")
    assert _records(caplog, "decision")
    assert _records(caplog, "action") == []
    would_be = _records(caplog, "would_be_action")
    assert len(would_be) == 1
    assert would_be[0].pathlayer["action"] == "STOP"


def test_enforcement_mode_is_unchanged() -> None:
    # Regression: default (dry_run=False) still halts and writes a post-mortem.
    pipeline = Pipeline([LoopDetector()])
    pipeline.run(_loop_events(20))

    assert pipeline.dry_run is False
    assert pipeline.stopped is True
    assert pipeline.post_mortem is not None
    assert pipeline.would_be_actions == ()
    assert Action.STOP in pipeline.actions


class _AlwaysStopPolicy(Policy):
    def step(
        self, state: HysteresisState, signals: Sequence[Signal]
    ) -> tuple[HysteresisState, Action]:
        return HysteresisState(Action.STOP), Action.STOP


class _ScriptedPolicy(Policy):
    """Returns a pre-set action per call, to drive arbitrary transitions.

    The pipeline drives the policy through :meth:`Policy.step` (ARB-3), so the
    test double overrides ``step`` directly to script the committed action.
    """

    def __init__(self, script: list[Action]) -> None:
        super().__init__()
        self._script = script
        self._i = 0

    def step(
        self, state: HysteresisState, signals: Sequence[Signal]
    ) -> tuple[HysteresisState, Action]:
        action = self._script[self._i]
        self._i += 1
        return HysteresisState(action), action


def test_dry_run_records_each_transition_and_recovery() -> None:
    # NONE -> STOP -> NONE: the STOP is recorded on its transition; the recovery
    # back to NONE is a transition too but records nothing (NONE is not an
    # action), and never crashes or halts.
    policy = _ScriptedPolicy([Action.NONE, Action.STOP, Action.NONE])
    pipeline = Pipeline([], policy=policy, dry_run=True)
    pipeline.run(_loop_events(3))

    (would_be,) = pipeline.would_be_actions
    assert would_be.action is Action.STOP
    assert would_be.index == 1
    assert pipeline.stopped is False
    assert all(action is Action.NONE for action in pipeline.actions)


def test_dry_run_records_escalation_between_non_none_actions() -> None:
    # Forward-compat with the graded ladder (ARB): a WARN -> STOP escalation is
    # two distinct transitions, and a re-stop after recovery is recorded again.
    # NONE, WARN, STOP, NONE, STOP -> entries: WARN@1, STOP@2, STOP@4.
    policy = _ScriptedPolicy([Action.NONE, Action.WARN, Action.STOP, Action.NONE, Action.STOP])
    pipeline = Pipeline([], policy=policy, dry_run=True)
    pipeline.run(_loop_events(5))

    recorded = [(w.index, w.action) for w in pipeline.would_be_actions]
    assert recorded == [(1, Action.WARN), (2, Action.STOP), (4, Action.STOP)]
    assert pipeline.stopped is False
    # With no detectors, the reason falls back to the action's own name -- a
    # would-be WARN reads "warn", not a misleading hard-coded "stop".
    assert pipeline.would_be_actions[0].reason == "warn"
    assert pipeline.would_be_actions[1].reason == "stop"


def test_dry_run_would_be_reason_handles_empty_signals() -> None:
    # A custom policy can STOP with no detectors/signals; the would-be record
    # falls back to the "stop" reason, mirroring the post-mortem path.
    pipeline = Pipeline([], policy=_AlwaysStopPolicy(), dry_run=True)
    pipeline.process(ToolEvent(tool="read"))

    (would_be,) = pipeline.would_be_actions
    assert would_be.action is Action.STOP
    assert would_be.signals == ()
    assert would_be.reason == "stop"
    assert pipeline.stopped is False
