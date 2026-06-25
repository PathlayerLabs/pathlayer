"""VAL-3: tests for the replay harness + metrics."""

from __future__ import annotations

from pathlib import Path

from pathlayer.events import ToolEvent
from pathlayer.policies import Policy
from pathlayer.policies.base import PolicyConfig
from pathlayer.validation import (
    Report,
    Trace,
    load_traces,
    replay_trace,
    score,
)
from pathlayer.validation.replay import main

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def _loop(label: str = "loop", *, repeats: int = 8, cost: float = 1.0) -> Trace:
    """A single action repeated `repeats` times; each event costs `cost`."""
    events = [ToolEvent(tool="poll", args={"job": "x"}, cost_usd=cost) for _ in range(repeats)]
    return Trace(label=label, events=events)


def _healthy() -> Trace:
    tools = ["list_dir", "read_file", "edit_file", "run_tests", "git_commit"]
    events = [ToolEvent(tool=t, args={"i": i}, cost_usd=0.1) for i, t in enumerate(tools)]
    return Trace(label="healthy", events=events)


# --------------------------------------------------------------------------- #
# replay_trace
# --------------------------------------------------------------------------- #
def test_replay_loop_intervenes_and_avoids_work() -> None:
    result = replay_trace(_loop(repeats=8, cost=1.0))
    assert result.intervened
    # LoopDetector(k=2) reaches stop_threshold 0.9 at the 5th repeat (index 4).
    assert result.detected_at_index == 4
    assert result.events_processed == 5
    assert result.events_avoided == 3
    assert result.cost_avoided == 3.0


def test_replay_healthy_does_not_intervene() -> None:
    result = replay_trace(_healthy())
    assert not result.intervened
    assert result.detected_at_index is None
    assert result.events_avoided == 0
    assert result.cost_avoided == 0.0


def test_replay_threshold_is_tunable() -> None:
    short_loop = _loop(repeats=3)  # only 3 reps -> max confidence 0.75
    assert not replay_trace(short_loop).intervened
    lenient = Policy(PolicyConfig(stop_threshold=0.5))
    assert replay_trace(short_loop, policy=lenient).intervened


def test_replay_with_no_detectors_never_intervenes() -> None:
    result = replay_trace(_loop(), detectors=[])
    assert not result.intervened
    assert result.events_avoided == 0


# --------------------------------------------------------------------------- #
# score / Report
# --------------------------------------------------------------------------- #
def test_score_aggregates_confusion_matrix() -> None:
    traces = [
        _loop(),  # pathological, detected -> TP
        _healthy(),  # healthy, silent -> TN
        _loop(label="healthy"),  # healthy that loops -> FP
        Trace(label="no_progress", events=_healthy().events),  # patho, silent -> FN
    ]
    report = score(traces)
    assert report.true_positives == 1
    assert report.true_negatives == 1
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.healthy_total == 2
    assert report.pathological_total == 2
    assert report.fp_rate == 0.5
    assert report.fn_rate == 0.5
    assert report.mean_detection_latency == 4.0


def test_report_per_class_and_to_dict() -> None:
    report = score([_loop(), _healthy()])
    per_class = report.per_class()
    assert set(per_class) == {"loop", "healthy"}
    assert per_class["loop"]["intervened"] == 1
    assert per_class["healthy"]["intervened"] == 0
    payload = report.to_dict()
    assert payload["traces"] == 2
    assert payload["true_positives"] == 1
    assert "per_class" in payload


def test_empty_report_has_zero_rates() -> None:
    report = Report([])
    assert report.fp_rate == 0.0
    assert report.fn_rate == 0.0
    assert report.mean_detection_latency == 0.0
    assert report.events_avoided == 0
    assert report.cost_avoided == 0.0


def test_report_format_is_human_readable() -> None:
    text = score([_loop(), _healthy()]).format()
    assert "FP=" in text
    assert "per class:" in text


# --------------------------------------------------------------------------- #
# load_traces + golden corpus integration
# --------------------------------------------------------------------------- #
def test_load_traces_single_file_and_dir() -> None:
    one = load_traces(_TRACES_DIR / "loop" / "volatile_ids.json")
    assert len(one) == 1
    many = load_traces(_TRACES_DIR / "loop")
    assert len(many) >= 4


def test_golden_loops_are_all_caught() -> None:
    report = score(load_traces(_TRACES_DIR / "loop"))
    assert report.false_negatives == 0
    assert report.true_positives == report.pathological_total


def test_golden_healthy_have_no_false_positives() -> None:
    report = score(load_traces(_TRACES_DIR / "healthy"))
    assert report.false_positives == 0


def test_golden_anchored_no_progress_caught_under_default_policy() -> None:
    # Since NOP-3, no_progress is actionable by default. The one anchored stall in
    # the corpus (ground-truth tests stay red) is caught without a custom policy.
    # The unanchored no_progress traces (no test / exit-code signal) stay out of
    # NoProgressDetector's reach and remain intentional false negatives until a
    # later anchor (NOP-4 git diff) or another detector (budget) covers them.
    churn = load_traces(_TRACES_DIR / "no_progress" / "failing_edit_churn.json")
    assert replay_trace(churn[0]).intervened
    report = score(load_traces(_TRACES_DIR / "no_progress"))
    assert report.true_positives >= 1


def test_active_policy_catches_anchored_no_progress() -> None:
    # Pins the behavior explicitly, independent of the default: with the advisory
    # set empty, the anchored stall is caught and no healthy trace turns into a
    # false positive (the success-suppression guardrail). Equivalent to the
    # default policy since NOP-3, kept as a regression anchor.
    active = Policy(PolicyConfig(advisory_signals=frozenset()))
    churn = load_traces(_TRACES_DIR / "no_progress" / "failing_edit_churn.json")
    assert replay_trace(churn[0], policy=active).intervened
    healthy = score(load_traces(_TRACES_DIR / "healthy"), policy=active)
    assert healthy.false_positives == 0


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def test_main_runs_on_directory(capsys) -> None:  # type: ignore[no-untyped-def]
    code = main([str(_TRACES_DIR)])
    assert code == 0
    out = capsys.readouterr().out
    assert "traces:" in out


def test_main_accepts_stop_threshold(capsys) -> None:  # type: ignore[no-untyped-def]
    code = main([str(_TRACES_DIR / "loop"), "--stop-threshold", "0.5"])
    assert code == 0
    assert "FP=" in capsys.readouterr().out
