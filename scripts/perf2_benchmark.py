"""PERF-2 (#76) reproducible before/after throughput benchmark.

Reproduces the issue #76 measurement: ``build_pipeline(PathlayerConfig())`` in
``dry_run`` mode, fed ``N`` distinct ``TOOL_CALL`` events each carrying
``exit_code`` evidence (so the coding anchors -- and therefore
``NoProgressDetector`` -- are actually applicable every tick, not silently
abstaining), reporting throughput by 500-event block, the first/last-block
ratio, and the total time.

Not a pytest test on purpose (see ``perf-2-plan-durci`` -- "benchmark
reproductible hors assertion stricte"): CI must not depend on wall-clock
numbers, and pre-PERF-2 the 20,000-event run does not finish in a reasonable
time. Run it directly, pointing ``PYTHONPATH``/the active checkout at either
side of the change to compare:

    python scripts/perf2_benchmark.py 4000 20000

Scope note: this measures the **default** pipeline (loop, retry, no_progress;
budget disabled). PERF-2 makes ``NoProgressDetector`` O(1) amortized per event,
but ``LoopDetector``/``RetryDetector`` still use the batch
``evaluate(history)`` fallback (unchanged from before PERF-2, out of this
issue's scope) -- so the *default* pipeline's remaining per-event cost is
whatever those two already cost, not O(1). If ``BudgetDetector`` or another
O(n)-per-call opt-in detector is enabled and becomes the dominant cost, that is
a new bottleneck for a follow-up issue (PERF-3), not a PERF-2 regression.
"""

from __future__ import annotations

import sys
import time

from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.events import EventKind, ToolEvent


def make_event(i: int) -> ToolEvent:
    return ToolEvent(
        kind=EventKind.TOOL_CALL,
        tool=f"tool_{i}",
        args={"i": i},
        result={"exit_code": 0 if i % 3 else 1},
        metadata={},
        ts=float(i),
        cost_usd=0.0,
    )


def run(n: int, block: int = 500) -> None:
    if n <= 0:
        print(f"n={n}: skipping, nothing to benchmark")
        return

    cfg = PathlayerConfig()
    cfg.pipeline.dry_run = True
    pipeline = build_pipeline(cfg)
    sink = pipeline.sink

    times: list[float] = []
    t_total_start = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        sink(make_event(i))
        times.append(time.perf_counter() - t0)
    t_total = time.perf_counter() - t_total_start

    if len(pipeline.history) != n:
        raise AssertionError(f"expected {n} events processed, got {len(pipeline.history)}")

    print(f"n={n} total={t_total:.3f}s avg_rate={n / t_total:.1f} ev/s")
    n_blocks = max(1, n // block)
    for b in range(n_blocks):
        chunk = times[b * block : (b + 1) * block]
        if not chunk:
            continue
        rate = len(chunk) / sum(chunk)
        print(f"  block {b:>3} [{b * block:>6}:{(b + 1) * block:>6}] rate={rate:>9.1f} ev/s")
    first = times[:block]
    last = times[-block:]
    first_rate = len(first) / sum(first)
    last_rate = len(last) / sum(last)
    print(
        f"  first-{block} rate={first_rate:.1f} ev/s, last-{block} rate={last_rate:.1f} ev/s, "
        f"ratio(first/last)={first_rate / last_rate:.2f}x"
    )


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [4000, 20000]
    for n in sizes:
        run(n)
        print()
