"""ANC-3 reproducible before/after progress-proof corpus benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.proof import ProofProgressAnchor
from pathlayer.detectors.proof_producer import build_producer
from pathlayer.proof_scaffold import write_progress_proof_scaffold
from pathlayer.validation.replay import ReplayResult, Report, load_traces, score
from pathlayer.validation.traces import Trace

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TRACES = _REPO_ROOT / "traces"
_DEFAULT_SCENARIOS = Path(__file__).with_name("anc3_progress_scenarios.json")
_DEFAULT_PROBE = Path(__file__).with_name("anc3_fixture_probe.py")


class BenchmarkError(ValueError):
    """Raised when the sealed ANC-3 inputs or results violate the benchmark contract."""


@dataclass(frozen=True, slots=True)
class Scenario:
    objective: str
    checkpoints: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True, slots=True)
class TraceComparison:
    path: str
    before: ReplayResult
    after: ReplayResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "label": self.before.label,
            "before_intervened": self.before.intervened,
            "after_intervened": self.after.intervened,
            "before_index": self.before.detected_at_index,
            "after_index": self.after.detected_at_index,
            "after_reason": self.after.intervention_reason,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    before: Report
    after: Report
    traces: tuple[TraceComparison, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
        }

    def format(self) -> str:
        lines = ["ANC-3 progress-proof corpus benchmark", "", "before:"]
        lines.extend(f"  {line}" for line in self.before.format().splitlines())
        lines.extend(["", "after:"])
        lines.extend(f"  {line}" for line in self.after.format().splitlines())
        lines.extend(["", "per trace:"])
        for trace in self.traces:
            before = _outcome(trace.before)
            after = _outcome(trace.after)
            reason = (
                f" ({trace.after.intervention_reason})" if trace.after.intervention_reason else ""
            )
            lines.append(f"  {trace.path}: {before} -> {after}{reason}")
        return "\n".join(lines)


def _outcome(result: ReplayResult) -> str:
    if not result.intervened:
        return "silent"
    return f"STOP@index-{result.detected_at_index}"


def _trace_paths(traces_root: Path) -> list[Path]:
    paths = [traces_root / "healthy_trivial.json", traces_root / "no_progress_trivial.json"]
    paths.extend(sorted((traces_root / "healthy").glob("*.json")))
    paths.extend(sorted((traces_root / "no_progress").glob("*.json")))
    if any(not path.is_file() for path in paths):
        raise BenchmarkError("ANC-3 corpus is missing a required trivial trace")
    return paths


def _load_scenarios(path: Path) -> dict[str, Scenario]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot load ANC-3 scenarios {path}: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "traces"}:
        raise BenchmarkError("ANC-3 scenario root must contain schema_version and traces")
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise BenchmarkError("unsupported ANC-3 scenario schema_version")
    raw_traces = raw["traces"]
    if not isinstance(raw_traces, dict):
        raise BenchmarkError("ANC-3 scenario traces must be an object")

    scenarios: dict[str, Scenario] = {}
    for trace_path, raw_scenario in raw_traces.items():
        if not isinstance(trace_path, str) or not isinstance(raw_scenario, dict):
            raise BenchmarkError("every ANC-3 trace scenario must be a named object")
        if set(raw_scenario) != {"objective", "checkpoints"}:
            raise BenchmarkError(f"invalid ANC-3 scenario fields for {trace_path}")
        objective = raw_scenario["objective"]
        raw_checkpoints = raw_scenario["checkpoints"]
        if not isinstance(objective, str) or not objective:
            raise BenchmarkError(f"scenario objective must be non-empty for {trace_path}")
        if not isinstance(raw_checkpoints, dict) or not raw_checkpoints:
            raise BenchmarkError(f"scenario checkpoints must be non-empty for {trace_path}")
        checkpoints: list[tuple[str, int | None]] = []
        for checkpoint_id, true_at in raw_checkpoints.items():
            if not isinstance(checkpoint_id, str) or not checkpoint_id:
                raise BenchmarkError(f"invalid checkpoint id for {trace_path}")
            if true_at is not None and (
                not isinstance(true_at, int) or isinstance(true_at, bool) or true_at < 0
            ):
                raise BenchmarkError(f"invalid true_at for {trace_path}:{checkpoint_id}")
            checkpoints.append((checkpoint_id, true_at))
        scenarios[trace_path] = Scenario(objective, tuple(sorted(checkpoints)))
    return scenarios


def _load_corpus(paths: Sequence[Path]) -> list[Trace]:
    return [load_traces(path)[0] for path in paths]


def _write_scenario(path: Path, scenario: Scenario) -> None:
    path.write_text(
        json.dumps(
            {"checkpoints": dict(scenario.checkpoints)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _enrich_trace(
    trace: Trace,
    *,
    relative_path: str,
    scenario: Scenario,
    work_dir: Path,
    probe_script: Path,
) -> None:
    out_of_range = [
        checkpoint_id
        for checkpoint_id, true_at in scenario.checkpoints
        if true_at is not None and true_at >= len(trace.events)
    ]
    if out_of_range:
        raise BenchmarkError(
            f"scenario checkpoint index is outside {relative_path}: {out_of_range}"
        )
    bundle_dir = work_dir / relative_path.replace("/", "__").removesuffix(".json")
    paths = write_progress_proof_scaffold(
        output_dir=bundle_dir,
        task_id=f"anc3:{relative_path}",
        name=f"ANC-3 corpus proof for {relative_path}",
        objective=scenario.objective,
        checkpoints=[checkpoint_id for checkpoint_id, _ in scenario.checkpoints],
    )
    scenario_path = bundle_dir / "scenario.json"
    counter_path = bundle_dir / "probe-counter.txt"
    _write_scenario(scenario_path, scenario)
    producer = build_producer(
        str(paths.manifest),
        (sys.executable, str(probe_script), str(scenario_path), str(counter_path)),
        timeout_seconds=5.0,
    )
    for event in trace.events:
        if producer.attach(event) is None:
            raise BenchmarkError(f"representative probe failed for {relative_path}")


def _assert_new_stop_attribution(comparisons: Sequence[TraceComparison]) -> None:
    for comparison in comparisons:
        if comparison.before.intervened or not comparison.after.intervened:
            continue
        if not (comparison.after.intervention_reason or "").startswith("no_progress"):
            raise BenchmarkError(f"new STOP was not attributed to no_progress: {comparison.path}")
        positive_other = [
            signal.name
            for signal in comparison.after.intervention_signals
            if signal.name != "no_progress" and signal.confidence > 0.0
        ]
        if positive_other:
            raise BenchmarkError(
                f"new STOP had other positive signals for {comparison.path}: {positive_other}"
            )


def run_benchmark(
    *,
    traces_root: Path = _DEFAULT_TRACES,
    scenarios_path: Path = _DEFAULT_SCENARIOS,
    probe_script: Path = _DEFAULT_PROBE,
) -> BenchmarkComparison:
    """Run the sealed ANC-3 baseline and representative-producer comparison."""
    trace_paths = _trace_paths(traces_root)
    relative_paths = [path.relative_to(traces_root).as_posix() for path in trace_paths]
    scenarios = _load_scenarios(scenarios_path)
    if set(scenarios) != set(relative_paths):
        missing = sorted(set(relative_paths) - set(scenarios))
        extra = sorted(set(scenarios) - set(relative_paths))
        raise BenchmarkError(f"scenario/corpus mismatch: missing={missing}, extra={extra}")
    if not probe_script.is_file():
        raise BenchmarkError(f"representative probe script not found: {probe_script}")

    before = score(_load_corpus(trace_paths))
    enriched = _load_corpus(trace_paths)
    with tempfile.TemporaryDirectory(prefix="pathlayer-anc3-") as temp_name:
        work_dir = Path(temp_name)
        for relative_path, trace in zip(relative_paths, enriched, strict=True):
            _enrich_trace(
                trace,
                relative_path=relative_path,
                scenario=scenarios[relative_path],
                work_dir=work_dir,
                probe_script=probe_script,
            )
        detectors = [
            LoopDetector(),
            NoProgressDetector([*coding_anchors(), ProofProgressAnchor()]),
        ]
        after = score(enriched, detectors=detectors)

    comparisons = tuple(
        TraceComparison(path, old, new)
        for path, old, new in zip(relative_paths, before.results, after.results, strict=True)
    )
    _assert_new_stop_attribution(comparisons)
    return BenchmarkComparison(before=before, after=after, traces=comparisons)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=_DEFAULT_TRACES)
    parser.add_argument("--scenarios", type=Path, default=_DEFAULT_SCENARIOS)
    parser.add_argument("--probe", type=Path, default=_DEFAULT_PROBE)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    comparison = run_benchmark(
        traces_root=args.traces,
        scenarios_path=args.scenarios,
        probe_script=args.probe,
    )
    if args.as_json:
        print(json.dumps(comparison.to_dict(), indent=2, sort_keys=True))
    else:
        print(comparison.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
