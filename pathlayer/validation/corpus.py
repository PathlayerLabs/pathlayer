"""VAL-5: reproducible, non-interactive corpus builder.

Sweeps seeded healthy-trace generators and the VAL-4 mutations to write a mixed
corpus (~50% healthy / ~50% pathological) under an output directory, in the
VAL-1 on-disk format. Deterministic for a given ``--seed`` and idempotent (the
output directory is cleaned before writing), so it is safe to run in CI.

The output lives in its own directory (default ``traces/generated/``) and is
organised by *category* -- ``healthy`` plus the four fault kinds (``loop``,
``retry_storm``, ``no_progress``, ``cost_spike``). Category is read from the
mutation metadata, not from the VAL-1 label, because ``inject_cost_spike``
deliberately keeps the base label. The curated golden set
(``traces/{loop,no_progress,healthy}/``) is never touched.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pathlayer.events import ToolEvent
from pathlayer.validation.mutations import (
    inject_cost_spike,
    inject_loop,
    inject_no_progress,
    inject_retry_storm,
)
from pathlayer.validation.traces import Trace, dump_trace

DEFAULT_OUT_DIR = Path("traces/generated")

_TOOLS = (
    "list_dir",
    "read_file",
    "edit_file",
    "run_tests",
    "git_commit",
    "search",
    "grep",
    "write_file",
)
_FAULTS = ("loop", "retry_storm", "no_progress", "cost_spike")


def _healthy_trace(rng: random.Random, index: int) -> Trace:
    """A varied, advancing run. The per-step ``step`` arg keeps events distinct
    so a healthy trace never accidentally looks like a loop."""
    length = rng.randint(6, 14)
    events = [
        ToolEvent(
            tool=rng.choice(_TOOLS),
            args={"path": f"src/mod_{rng.randint(0, 99)}.py", "step": step},
            cost_usd=round(rng.uniform(0.01, 0.3), 4),
            ts=float(step),
        )
        for step in range(length)
    ]
    return Trace(
        label="healthy",
        description=f"generated healthy run #{index}",
        events=events,
        metadata={"generated": True, "index": index},
    )


def _pathological_trace(rng: random.Random, index: int) -> Trace:
    """Generate a healthy base and inject one randomly chosen fault."""
    base = _healthy_trace(rng, index)
    n = len(base.events)
    fault = rng.choice(_FAULTS)
    if fault == "loop":
        start = rng.randint(0, n - 2)
        seq_len = rng.randint(1, min(3, n - start))
        return inject_loop(
            base, start_idx=start, sequence_length=seq_len, n_repeats=rng.randint(5, 9)
        )
    if fault == "retry_storm":
        return inject_retry_storm(
            base,
            start_idx=rng.randint(0, n),
            n_events=rng.randint(5, 9),
            tool="api_call",
            error="429",
        )
    if fault == "no_progress":
        return inject_no_progress(base, start_idx=rng.randint(0, n - 1))
    return inject_cost_spike(
        base, event_idx=rng.randint(0, n - 1), multiplier=rng.uniform(10.0, 60.0)
    )


def _category(trace: Trace) -> str:
    mutation = trace.metadata.get("mutation")
    if mutation is None:
        return "healthy"
    category: str = mutation["type"]
    return category


@dataclass(slots=True)
class CorpusSummary:
    """What was written: per-category counts, the healthy/pathological split and
    the total number of events."""

    counts: dict[str, int]
    healthy: int
    pathological: int
    total_events: int

    def format(self) -> str:
        lines = [
            f"corpus: {self.healthy + self.pathological} traces "
            f"(healthy={self.healthy}, pathological={self.pathological}), "
            f"{self.total_events} events",
            "per category:",
        ]
        for category in sorted(self.counts):
            lines.append(f"  {category}: {self.counts[category]}")
        return "\n".join(lines)


def build_corpus(
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    n: int = 20,
    seed: int = 0,
    clean: bool = True,
) -> CorpusSummary:
    """Write ``n`` traces (alternating healthy / pathological) under ``out_dir``."""
    if n < 0:
        raise ValueError("n must be >= 0")

    rng = random.Random(seed)
    out = Path(out_dir)
    if clean and out.exists():
        shutil.rmtree(out)

    counts: Counter[str] = Counter()
    total_events = 0
    for index in range(n):
        trace = _healthy_trace(rng, index) if index % 2 == 0 else _pathological_trace(rng, index)
        category = _category(trace)
        counts[category] += 1
        total_events += len(trace.events)
        path = out / category / f"trace_{index:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_trace(trace, path)

    healthy = counts.get("healthy", 0)
    return CorpusSummary(
        counts=dict(counts),
        healthy=healthy,
        pathological=n - healthy,
        total_events=total_events,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pathlayer-build-corpus",
        description="Build a reproducible mixed trace corpus (non-interactive).",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output directory")
    parser.add_argument("--n", type=int, default=20, help="number of traces to generate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="overwrite the output directory without asking (required if it exists)",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.yes:
        print(f"refusing to overwrite existing {out} without --yes")
        return 1

    summary = build_corpus(out, n=args.n, seed=args.seed)
    print(summary.format())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
