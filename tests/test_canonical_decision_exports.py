"""OBS-4: public decision exports use action names, never IntEnum numbers."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from pathlayer.adapters.openclaw_report import _tick_from_decision
from pathlayer.detectors.base import Signal
from pathlayer.pipeline import PipelineDecision
from pathlayer.policies import Action
from pathlayer.policies.actions import action_from_export, action_name
from pathlayer.postmortem import TickRecord


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(Action.NONE, Action.NONE), (0, Action.NONE), ("0", Action.NONE), ("WARN", Action.WARN)],
)
def test_legacy_numeric_or_named_action_reads_as_the_same_action(
    raw: object, expected: Action
) -> None:
    assert action_from_export(raw) is expected
    assert action_name(raw) == expected.name


@pytest.mark.parametrize("raw", [True, -1, 6, "unknown", "6", " warn ", "Warn", "\u0665", "\uff15"])
def test_invalid_exported_action_is_rejected(raw: object) -> None:
    with pytest.raises(ValueError, match="invalid action"):
        action_from_export(raw)


def test_tick_record_normalizes_a_legacy_numeric_action() -> None:
    tick = TickRecord.from_dict({"index": 4, "ts": 2.0, "action": 0})

    assert tick.action is Action.NONE
    assert tick.to_dict()["action"] == "NONE"


def test_openclaw_replay_normalizes_the_same_numeric_action_as_live_export() -> None:
    live = TickRecord(index=4, ts=2.0, action=Action.STOP)
    replay = _tick_from_decision(
        {
            "analyzed": True,
            "event_index": 4,
            "event_ts": 2.0,
            "signals": [],
            "fused": 1.0,
            "arbitrated_action": 5,
            "reason": "stop",
            "dry_run": False,
        }
    )

    assert replay is not None
    assert replay.to_dict()["action"] == live.to_dict()["action"] == "STOP"


def test_pipeline_decision_exports_actions_canonically_on_the_generic_alpha_path() -> None:
    decision = PipelineDecision(
        index=3,
        ts=1.5,
        signals=(Signal("loop", confidence=0.9, reason="repeat"),),
        evidence={"progress": 0.0},
        fused=0.9,
        reason="loop: repeat",
        arbitrated_action=Action.STOP,
        emitted_action=Action.STOP,
        dry_run=False,
    )

    exported = asdict(decision)

    assert exported["arbitrated_action"] == "STOP"
    assert exported["emitted_action"] == "STOP"
    serialized = json.loads(json.dumps(exported))
    assert serialized["arbitrated_action"] == "STOP"
    assert serialized["emitted_action"] == "STOP"
