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

``LoopDetector`` and ``NoProgressDetector`` (CUSUM) are wired by default. Since
NOP-3 the ``no_progress`` signal is actionable in the default policy, so an
anchored stall STOPs without any custom policy. The ``BudgetDetector`` and
``RetryDetector`` are implemented but not wired into the default detector set
here, so traces that depend on them surface as false negatives; the harness
reports that honestly.
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
from pathlayer.detectors.progress import ProgressAnchor, coding_anchors, loop_suppression_anchors
from pathlayer.pipeline import PROGRESS_EVIDENCE_KEY, Pipeline
from pathlayer.policies import Action, Policy, SuppressionRule
from pathlayer.policies.base import PolicyConfig
from pathlayer.validation.traces import Trace, load_trace

HEALTHY_LABEL = "healthy"


def _default_detectors() -> list[Detector]:
    # LoopDetector and NoProgressDetector (CUSUM) both intervene by default: since
    # NOP-3 no_progress is actionable, so an anchored stall STOPs without a custom
    # policy. RetryDetector and BudgetDetector are implemented but intentionally
    # not wired here, so traces that depend on them surface as false negatives.
    return [LoopDetector(), NoProgressDetector(coding_anchors())]


def _default_policy() -> Policy:
    """Mirrors ``build_pipeline``'s auto-wired ``loop<-progress`` suppression.

    Both the golden-trace regression tests and ad-hoc replay calls should
    reflect the same LOOP-2 precision fix as production, not an unfixed
    baseline -- otherwise the F1a fixture (``traces/healthy/
    repetitive_but_passing.json``) would stay a false positive here even after
    the bug is fixed in ``build_pipeline``.
    """
    return Policy(
        PolicyConfig(
            suppressions={
                LoopDetector.name: SuppressionRule(
                    evidence=PROGRESS_EVIDENCE_KEY, min_confidence=1.0
                )
            }
        )
    )


def _default_progress_anchors() -> list[ProgressAnchor]:
    """Anchors backing the default policy's suppression rule (see :func:`_default_policy`)."""
    return loop_suppression_anchors()


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
    progress_anchors: Sequence[ProgressAnchor] | None = None,
) -> ReplayResult:
    """Replay one trace through a fresh pipeline and summarize the outcome.

    ``policy``/``progress_anchors`` default to the same loop<-progress
    suppression :func:`build_pipeline <pathlayer.config.build_pipeline>` wires
    in production (see :func:`_default_policy`), so an explicit caller has to
    opt *out* of the fix (pass a bare ``Policy()``), not opt into it.
    """
    chosen = list(detectors) if detectors is not None else _default_detectors()
    chosen_policy = policy if policy is not None else _default_policy()
    anchors = progress_anchors if progress_anchors is not None else _default_progress_anchors()
    pipeline = Pipeline(chosen, chosen_policy, progress_anchors=anchors)
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


# Subdirectories that are NOT part of the curated pipeline corpus and must not
# pollute an aggregate score of ``traces/``:
# - ``nop4_experiment``: anchor-only fixtures (see test_nop4_git_experiment.py);
#   some are 16x-identical edits that legitimately trip the LoopDetector in the
#   full pipeline, inflating the FP rate with traces never meant for it.
# - ``generated``: synthetic VAL-5 output, not hand-labeled golden traces.
# They are skipped only when discovered *beneath* the scan root, so pointing
# load_traces directly at one of them still loads it (the experiment test does).
_NON_PIPELINE_SUBDIRS = frozenset({"nop4_experiment", "generated"})


def load_traces(path: str | Path) -> list[Trace]:
    """Load a single trace file or every ``*.json`` trace under a directory.

    When scanning a directory, non-pipeline subdirectories
    (:data:`_NON_PIPELINE_SUBDIRS`) are skipped so an aggregate score reflects
    only the curated pipeline corpus. Targeting such a subdirectory directly
    still loads it -- the exclusion only applies to nested discovery.
    """
    root = Path(path)
    if root.is_file():
        return [load_trace(root)]
    paths = [
        p
        for p in sorted(root.rglob("*.json"))
        if not _NON_PIPELINE_SUBDIRS.intersection(p.relative_to(root).parts[:-1])
    ]
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
