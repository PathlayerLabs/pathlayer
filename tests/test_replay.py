"""VAL-3: tests for the replay harness + metrics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pathlayer.config import build_pipeline
from pathlayer.detectors.canonical import ToolMask
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import ToolEvent
from pathlayer.pipeline import PROGRESS_EVIDENCE_KEY
from pathlayer.policies import Policy, SuppressionRule
from pathlayer.policies.base import PolicyConfig
from pathlayer.validation import (
    UNKNOWN_LABEL,
    Report,
    Trace,
    load_trace,
    load_traces,
    replay_trace,
    score,
)
from pathlayer.validation.replay import main

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


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
    assert result.intervention_reason is not None
    assert result.intervention_reason.startswith("loop")
    assert {signal.name for signal in result.intervention_signals} == {"loop", "no_progress"}


def test_replay_healthy_does_not_intervene() -> None:
    result = replay_trace(_healthy())
    assert not result.intervened
    assert result.detected_at_index is None
    assert result.events_avoided == 0
    assert result.cost_avoided == 0.0
    assert result.intervention_reason is None
    assert result.intervention_signals == ()


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


def test_unknown_trace_is_reported_but_excluded_from_confusion_metrics() -> None:
    report = score([Trace(label=UNKNOWN_LABEL), _loop(), _healthy()])

    assert len(report.results) == 3
    assert report.healthy_total == 1
    assert report.pathological_total == 1
    assert report.unscored_total == 1
    assert report.true_positives == 1
    assert report.true_negatives == 1
    assert report.false_positives == 0
    assert report.false_negatives == 0
    assert report.to_dict()["unscored"] == 1
    assert "unscored=1" in report.format()


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


@pytest.mark.parametrize("subdir", ["nop4_experiment", "val7"])
def test_load_traces_skips_non_pipeline_subdirs_when_scanning_root(subdir: str) -> None:
    # Scanning traces/ must exclude the anchor-only experiment, synthetic
    # generated corpora (audit finding M-1, 2026-06-29) and VAL-7 session-only
    # traces, which need session_stitching enabled, so an aggregate pipeline
    # score is not polluted.
    def keys(traces: list[Trace]) -> set[tuple[str, str]]:
        return {(t.label, t.description) for t in traces}

    scanned = keys(load_traces(_TRACES_DIR))
    excluded = keys(load_traces(_TRACES_DIR / subdir))
    assert excluded, f"expected {subdir} fixtures to exist"
    # None of the excluded traces leak into a full-tree scan.
    assert excluded.isdisjoint(scanned)


def test_load_traces_loads_excluded_dir_when_targeted_directly() -> None:
    # The exclusion is nested-discovery only: pointing straight at the subdir
    # still loads it (the NOP-4 experiment rig depends on this).
    direct = load_traces(_TRACES_DIR / "nop4_experiment")
    assert len(direct) >= 1


def test_full_corpus_scan_has_no_false_positives() -> None:
    # The whole curated tree scores clean: excluding non-pipeline subdirs, no
    # healthy trace is wrongly stopped (the FP=3 from nop4_experiment is gone).
    report = score(load_traces(_TRACES_DIR))
    assert report.false_positives == 0


def test_production_pipeline_has_no_false_positives_on_corpus() -> None:
    # RET-2 garde-fou: the default replay detectors deliberately omit the
    # RetryDetector, so this scores the curated corpus through the *production*
    # detector set (build_pipeline: loop + retry + no_progress) and pins FP=0.
    # This is what proves adding tool-dominance did not start stopping healthy
    # runs; the dominance component is capped below stop_threshold and gated on
    # failures, so legitimate monotool bursts (exploration, bounded retries)
    # stay quiet.
    pipeline = build_pipeline()
    report = score(
        load_traces(_TRACES_DIR),
        detectors=pipeline._detectors,
        policy=pipeline._policy,
    )
    assert report.false_positives == 0


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
    # false positive (the success-suppression guardrail, plus the LOOP-2
    # precision loop<-progress suppression -- an explicit custom policy must
    # opt back into it, same as build_pipeline does). Equivalent to the default
    # policy since NOP-3, kept as a regression anchor.
    active = Policy(
        PolicyConfig(
            advisory_signals=frozenset(),
            suppressions={
                LoopDetector.name: SuppressionRule(
                    evidence=PROGRESS_EVIDENCE_KEY, min_confidence=1.0
                )
            },
        )
    )
    churn = load_traces(_TRACES_DIR / "no_progress" / "failing_edit_churn.json")
    assert replay_trace(churn[0], policy=active).intervened
    healthy = score(load_traces(_TRACES_DIR / "healthy"), policy=active)
    assert healthy.false_positives == 0


# --------------------------------------------------------------------------- #
# CAN-3: UUID-valued id false merge, fixed by a per-tool keep config
# --------------------------------------------------------------------------- #
def test_can3_uuid_id_false_loop_is_fixed_by_keep_config() -> None:
    trace = load_trace(_FIXTURES_DIR / "can3_uuid_reads.json")
    # Default canonicalization masks the UUID-valued file_id by value shape, so
    # genuinely distinct reads collapse into one token -> false loop.
    assert replay_trace(trace, detectors=[LoopDetector()]).intervened
    # Declaring file_id as kept for read_file restores distinctness -> no loop.
    masks = {"read_file": ToolMask(keep=frozenset({"file_id"}))}
    assert not replay_trace(trace, detectors=[LoopDetector(tool_masks=masks)]).intervened


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


def test_installed_pathlayer_replay_command_prints_report_without_warning() -> None:
    # `python -m pathlayer.validation.replay` warns from runpy, because the
    # package imports the module first; the installed command does not.
    name = "pathlayer-replay.exe" if os.name == "nt" else "pathlayer-replay"
    executable = Path(sys.executable).with_name(name)
    assert executable.is_file(), f"missing {executable}: reinstall with pip install -e ."
    completed = subprocess.run(
        [str(executable), str(_TRACES_DIR / "loop_trivial.json")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "FP=0" in completed.stdout
    assert "TP=1" in completed.stdout
    assert "Warning" not in completed.stderr
