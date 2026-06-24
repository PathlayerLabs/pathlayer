"""VAL-3: offline replay harness + detector-quality metrics.

Replays recorded traces (VAL-1 format) through the existing
:class:`~pathlayer.pipeline.Pipeline` -- no live agent, no LLM -- and scores how
well the detectors behave:

- **false positives**: a ``healthy`` trace that triggers an intervention;
- **false negatives**: a ``loop`` / ``no_progress`` trace that is never caught;
- **detection latency**: how many steps until the first intervention;
- **work avoided**: events (and their cost) skipped after a ``STOP``.

This is the rig that makes thresholds tunable against data: build a custom
``Policy`` and feed it to :func:`score`.

``LoopDetector`` and ``NoProgressDetector`` (CUSUM) are wired by default. NOP-2
ships in observe mode -- ``no_progress`` is advisory in the default policy, so it
is computed and measured but does not STOP. To measure its quality, replay with
a policy that lifts the advisory exemption (``advisory_signals=frozenset()``).
Retry / budget detectors are still stubs (confidence 0), so traces that depend
on them surface as false negatives; the harness reports that honestly.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathlayer.detectors.base import Detector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.policies.base import PolicyConfig
from pathlayer.validation.traces import Trace, load_trace

HEALTHY_LABEL = "healthy"


def _default_detectors() -> list[Detector]:
    # LoopDetector intervenes; NoProgressDetector (CUSUM) ships in observe mode --
    # its no_progress signal is advisory in the default policy, so it is computed
    # and measured but does not STOP unless the policy lifts the exemption. The
    # remaining retry / budget detectors are still stubs (confidence 0).
    return [LoopDetector(), NoProgressDetector(coding_anchors())]


@dataclass(slots=True)
class ReplayResult:
    """Outcome of replaying a single trace through the pipeline."""

    label: str
    intervened: bool
    detected_at_index: int | None
    events_processed: int
    events_total: int
    cost_avoided: float

    @property
    def events_avoided(self) -> int:
        return self.events_total - self.events_processed


def replay_trace(
    trace: Trace,
    *,
    detectors: Sequence[Detector] | None = None,
    policy: Policy | None = None,
) -> ReplayResult:
    """Replay one trace through a fresh pipeline and summarize the outcome."""
    chosen = list(detectors) if detectors is not None else _default_detectors()
    pipeline = Pipeline(chosen, policy)
    pipeline.run(trace.events)

    actions = pipeline.actions
    detected_at_index = next(
        (i for i, action in enumerate(actions) if action is not Action.NONE), None
    )
    processed = len(pipeline.history)
    total = len(trace.events)
    cost_avoided = sum(event.cost_usd for event in trace.events[processed:])
    return ReplayResult(
        label=trace.label,
        intervened=detected_at_index is not None,
        detected_at_index=detected_at_index,
        events_processed=processed,
        events_total=total,
        cost_avoided=cost_avoided,
    )


@dataclass(slots=True)
class Report:
    """Aggregate detector-quality metrics over a set of replayed traces."""

    results: list[ReplayResult]

    def _count(self, *, healthy: bool, intervened: bool) -> int:
        return sum(
            1
            for r in self.results
            if (r.label == HEALTHY_LABEL) is healthy and r.intervened is intervened
        )

    @property
    def healthy_total(self) -> int:
        return sum(1 for r in self.results if r.label == HEALTHY_LABEL)

    @property
    def pathological_total(self) -> int:
        return len(self.results) - self.healthy_total

    @property
    def false_positives(self) -> int:
        return self._count(healthy=True, intervened=True)

    @property
    def true_negatives(self) -> int:
        return self._count(healthy=True, intervened=False)

    @property
    def false_negatives(self) -> int:
        return self._count(healthy=False, intervened=False)

    @property
    def true_positives(self) -> int:
        return self._count(healthy=False, intervened=True)

    @property
    def fp_rate(self) -> float:
        return self.false_positives / self.healthy_total if self.healthy_total else 0.0

    @property
    def fn_rate(self) -> float:
        total = self.pathological_total
        return self.false_negatives / total if total else 0.0

    @property
    def mean_detection_latency(self) -> float:
        latencies = [
            r.detected_at_index
            for r in self.results
            if r.label != HEALTHY_LABEL and r.detected_at_index is not None
        ]
        return sum(latencies) / len(latencies) if latencies else 0.0

    @property
    def events_avoided(self) -> int:
        return sum(r.events_avoided for r in self.results)

    @property
    def cost_avoided(self) -> float:
        return sum(r.cost_avoided for r in self.results)

    def per_class(self) -> dict[str, dict[str, Any]]:
        labels = sorted({r.label for r in self.results})
        out: dict[str, dict[str, Any]] = {}
        for label in labels:
            group = [r for r in self.results if r.label == label]
            out[label] = {
                "total": len(group),
                "intervened": sum(1 for r in group if r.intervened),
                "events_avoided": sum(r.events_avoided for r in group),
                "cost_avoided": sum(r.cost_avoided for r in group),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "traces": len(self.results),
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "fp_rate": self.fp_rate,
            "fn_rate": self.fn_rate,
            "mean_detection_latency": self.mean_detection_latency,
            "events_avoided": self.events_avoided,
            "cost_avoided": self.cost_avoided,
            "per_class": self.per_class(),
        }

    def format(self) -> str:
        lines = [
            f"traces: {len(self.results)} "
            f"(healthy={self.healthy_total}, pathological={self.pathological_total})",
            f"FP={self.false_positives} (rate {self.fp_rate:.2f})  "
            f"FN={self.false_negatives} (rate {self.fn_rate:.2f})  "
            f"TP={self.true_positives}  TN={self.true_negatives}",
            f"mean detection latency: {self.mean_detection_latency:.2f} steps",
            f"work avoided: {self.events_avoided} events, ${self.cost_avoided:.2f}",
            "per class:",
        ]
        for label, stats in self.per_class().items():
            lines.append(
                f"  {label}: {stats['intervened']}/{stats['total']} intervened, "
                f"{stats['events_avoided']} events avoided, ${stats['cost_avoided']:.2f}"
            )
        return "\n".join(lines)


def score(
    traces: Iterable[Trace],
    *,
    detectors: Sequence[Detector] | None = None,
    policy: Policy | None = None,
) -> Report:
    """Replay every trace and aggregate the metrics into a :class:`Report`."""
    results = [replay_trace(trace, detectors=detectors, policy=policy) for trace in traces]
    return Report(results)


def load_traces(path: str | Path) -> list[Trace]:
    """Load a single trace file or every ``*.json`` trace under a directory."""
    root = Path(path)
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    return [load_trace(p) for p in paths]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pathlayer-replay",
        description="Replay labeled traces through the pipeline and report detector quality.",
    )
    parser.add_argument("path", help="trace file or directory of traces")
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=None,
        help="override the policy stop_threshold (confidence in [0, 1])",
    )
    args = parser.parse_args(argv)

    policy: Policy | None = None
    if args.stop_threshold is not None:
        policy = Policy(PolicyConfig(stop_threshold=args.stop_threshold))

    report = score(load_traces(args.path), policy=policy)
    print(report.format())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
