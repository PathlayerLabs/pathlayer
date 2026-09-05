"""ANC-3: sealed before/after proof-producer measurement on the labeled corpus."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import anc3_corpus_benchmark
from scripts.anc3_corpus_benchmark import BenchmarkComparison, BenchmarkError

_ROOT = Path(__file__).resolve().parents[1]
_TRACES = _ROOT / "traces"
_SCENARIOS = _ROOT / "scripts" / "anc3_progress_scenarios.json"


def _trace_hashes() -> dict[str, str]:
    return {
        path.relative_to(_TRACES).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_TRACES.rglob("*.json"))
    }


@pytest.fixture(scope="module")
def comparison() -> Iterator[BenchmarkComparison]:
    before_hashes = _trace_hashes()
    result = anc3_corpus_benchmark.run_benchmark()
    assert _trace_hashes() == before_hashes
    yield result


def test_sealed_scenarios_cover_exactly_healthy_and_no_progress() -> None:
    raw = json.loads(_SCENARIOS.read_text(encoding="utf-8"))
    scenario_paths = set(raw["traces"])
    expected = {"healthy_trivial.json", "no_progress_trivial.json"}
    expected.update(f"healthy/{path.name}" for path in (_TRACES / "healthy").glob("*.json"))
    expected.update(f"no_progress/{path.name}" for path in (_TRACES / "no_progress").glob("*.json"))

    assert scenario_paths == expected
    assert len(scenario_paths) == 28
    assert not any(
        path.startswith("loop/") or path == "loop_trivial.json" for path in scenario_paths
    )


def test_before_after_confusion_matrix_and_latency(comparison: BenchmarkComparison) -> None:
    before = comparison.before
    after = comparison.after

    assert (before.healthy_total, before.pathological_total) == (20, 8)
    assert (before.false_positives, before.false_negatives) == (0, 4)
    assert (before.true_positives, before.true_negatives) == (4, 20)
    assert before.mean_detection_latency == 11.0

    assert (after.healthy_total, after.pathological_total) == (20, 8)
    assert (after.false_positives, after.false_negatives) == (0, 1)
    assert (after.true_positives, after.true_negatives) == (7, 20)
    assert after.mean_detection_latency == pytest.approx(10.57, abs=0.005)
    assert after.events_avoided - before.events_avoided == 3
    assert after.cost_avoided - before.cost_avoided == pytest.approx(0.47)


def test_three_new_stops_are_no_progress_only_at_index_ten(
    comparison: BenchmarkComparison,
) -> None:
    newly_stopped = {
        trace.path: trace
        for trace in comparison.traces
        if not trace.before.intervened and trace.after.intervened
    }

    assert set(newly_stopped) == {
        "no_progress/replan_churn.json",
        "no_progress/research_rabbit_hole.json",
        "no_progress/slow_burn_47k.json",
    }
    for trace in newly_stopped.values():
        assert trace.after.detected_at_index == 10
        assert trace.after.intervention_reason is not None
        assert trace.after.intervention_reason.startswith("no_progress")
        positive = [
            signal.name for signal in trace.after.intervention_signals if signal.confidence > 0.0
        ]
        assert positive == ["no_progress"]


def test_healthy_and_previously_detected_verdicts_do_not_regress(
    comparison: BenchmarkComparison,
) -> None:
    for trace in comparison.traces:
        if trace.before.label == "healthy":
            assert not trace.before.intervened
            assert not trace.after.intervened
        elif trace.before.intervened:
            assert trace.after.intervened
            assert trace.after.detected_at_index == trace.before.detected_at_index


def test_short_trivial_trace_remains_the_one_documented_false_negative(
    comparison: BenchmarkComparison,
) -> None:
    trace = next(item for item in comparison.traces if item.path == "no_progress_trivial.json")
    assert trace.before.events_total == 10
    assert not trace.before.intervened
    assert not trace.after.intervened
    assert comparison.after.false_negatives == 1


def test_scenario_mismatch_fails_before_enrichment(tmp_path: Path) -> None:
    raw = json.loads(_SCENARIOS.read_text(encoding="utf-8"))
    raw["traces"].pop("healthy_trivial.json")
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BenchmarkError, match="scenario/corpus mismatch"):
        anc3_corpus_benchmark.run_benchmark(scenarios_path=scenarios)


def test_benchmark_cli_json_uses_reproducible_result(
    comparison: BenchmarkComparison,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(anc3_corpus_benchmark, "run_benchmark", lambda **_: comparison)

    assert anc3_corpus_benchmark.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["before"]["false_negatives"] == 4
    assert payload["after"]["false_negatives"] == 1
