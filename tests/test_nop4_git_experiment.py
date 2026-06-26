"""NOP-4 Phase 0 - validation-first experiment (no production code).

This module is the *measurement rig* for the NOP-4 decision: does a git-diff
``ProgressAnchor`` help (remove a false stop) more than it hurts (mask a real
stall, or false-stop on its additive bias)? It keeps a *prototype* anchor here,
isolated -- this rig never imports the production anchor and never relies on
``coding_anchors()`` -- so it measures git against the pre-NOP-4 baseline even
now that the gate said GO and the production anchor ships.

The gate said GO: ``GitProgressAnchor`` now ships in production (in
``coding_anchors()``) with the ``size`` scalar and test-deferring shape this rig
selected. The rig is kept as the regression that pins *why* -- it still uses its
own isolated prototype anchor and an explicit tests+exit baseline, so it measures
git against the pre-NOP-4 behavior independently of what ``coding_anchors()`` now
ships.

It replays the labeled traces in ``traces/nop4_experiment/`` through
``NoProgressDetector`` under several anchor configurations and asserts the
confusion-matrix outcome of each. The configs:

- ``baseline``   : ``coding_anchors()`` (tests + exit-code), no git.
- ``+git_net``   : baseline + git anchor, level = insertions - deletions.
- ``+git_size``  : baseline + git anchor, level = insertions + deletions.
- ``+git_net_cond`` / ``+git_size_cond`` : the same git anchors, but
  *test-deferring* (abstain whenever the history carries any test evidence), so
  git can never mask an applicable test anchor.

Only ``NoProgressDetector`` is wired (no ``LoopDetector``) so the numbers reflect
the anchors alone, not exact-repeat detection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pathlayer.detectors.base import Detector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    ProgressAnchor,
    TestProgressAnchor,
)
from pathlayer.events import ToolEvent
from pathlayer.validation import Trace, load_traces, replay_trace

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "traces" / "nop4_experiment"

_TEST_KEYS = ("passed", "failed")
_INSERTIONS = "insertions"
_DELETIONS = "deletions"
_NET_LINES = "net_lines"


def _source(event: ToolEvent, keys: tuple[str, ...]) -> Mapping[str, object] | None:
    result = event.result
    if isinstance(result, Mapping) and any(key in result for key in keys):
        return result
    if any(key in event.metadata for key in keys):
        return event.metadata
    return None


def _has_test_evidence(history: Sequence[ToolEvent]) -> bool:
    return any(_source(event, _TEST_KEYS) is not None for event in history)


def _latest_git_source(history: Sequence[ToolEvent]) -> Mapping[str, object] | None:
    """Most recent git evidence mapping carried by the history, or None."""
    latest: Mapping[str, object] | None = None
    for event in history:
        source = _source(event, (_INSERTIONS, _DELETIONS, _NET_LINES))
        if source is not None:
            latest = source
    return latest


class _GitProgressAnchor(ProgressAnchor):
    """Prototype git progress anchor (experiment only, not production).

    ``scalar`` selects the level: ``"net"`` = insertions - deletions (rewards
    net additions, signed), ``"size"`` = insertions + deletions (magnitude of
    change, so in-place edits and deletions count). ``defer_to_tests`` makes it
    abstain whenever the history carries test evidence, so it can never mask an
    applicable test anchor (the "conditional" integration).

    Mirrors the production anchor on the degraded ``net_lines`` shortcut: ``net``
    keeps its sign, ``size`` takes the magnitude (never negative).
    """

    def __init__(self, scalar: str, *, defer_to_tests: bool = False) -> None:
        self._scalar = scalar
        self._defer = defer_to_tests
        self.name = f"git_{scalar}" + ("_cond" if defer_to_tests else "")

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        if self._defer and _has_test_evidence(history):
            return None
        source = _latest_git_source(history)
        if source is None:
            return None
        if _INSERTIONS in source or _DELETIONS in source:
            ins = int(source.get(_INSERTIONS, 0))  # type: ignore[arg-type]
            dele = int(source.get(_DELETIONS, 0))  # type: ignore[arg-type]
            return float(ins - dele) if self._scalar == "net" else float(ins + dele)
        net_lines = int(source.get(_NET_LINES, 0))  # type: ignore[arg-type]
        return float(net_lines) if self._scalar == "net" else float(abs(net_lines))


def _base() -> list[ProgressAnchor]:
    # The pre-NOP-4 coding anchors (tests + exit code), pinned here so this rig
    # measures git against that baseline even though coding_anchors() now ships
    # the production git anchor.
    return [TestProgressAnchor(), ExitCodeProgressAnchor()]


def _configs() -> dict[str, list[Detector]]:
    return {
        "baseline": [NoProgressDetector(_base())],
        "+git_net": [NoProgressDetector([*_base(), _GitProgressAnchor("net")])],
        "+git_size": [NoProgressDetector([*_base(), _GitProgressAnchor("size")])],
        "+git_net_cond": [
            NoProgressDetector([*_base(), _GitProgressAnchor("net", defer_to_tests=True)])
        ],
        "+git_size_cond": [
            NoProgressDetector([*_base(), _GitProgressAnchor("size", defer_to_tests=True)])
        ],
    }


def _outcome(trace: Trace, detectors: list[Detector]) -> str:
    intervened = replay_trace(trace, detectors=detectors).intervened
    healthy = trace.label == "healthy"
    if healthy:
        return "FP" if intervened else "TN"
    return "TP" if intervened else "FN"


def _matrix() -> tuple[list[Trace], dict[str, list[Detector]], dict[str, dict[str, str]]]:
    traces = sorted(load_traces(_EXPERIMENT_DIR), key=lambda t: t.description[:40])
    configs = _configs()
    table: dict[str, dict[str, str]] = {}
    for trace in traces:
        name = _trace_name(trace)
        table[name] = {cfg: _outcome(trace, dets) for cfg, dets in configs.items()}
    return traces, configs, table


def _trace_name(trace: Trace) -> str:
    # Recover a short name from the first src/<x>.py path touched.
    for event in trace.events:
        path = event.args.get("path")
        if isinstance(path, str):
            return f"{trace.label}:{path}"
    return trace.label


def _totals(
    table: dict[str, dict[str, str]],
    cfg: str,
    predicate: str,
    names: set[str] | None = None,
) -> int:
    rows = table.values() if names is None else (table[name] for name in names)
    return sum(1 for row in rows if row[cfg] == predicate)


def test_print_experiment_matrix() -> None:
    """Print the confusion-matrix table per (trace, config). Run with -s to read."""
    traces, configs, table = _matrix()
    cfg_names = list(configs)
    header = f"{'trace':<28} " + " ".join(f"{c:>14}" for c in cfg_names)
    lines = ["", header, "-" * len(header)]
    for name, row in table.items():
        lines.append(f"{name:<28} " + " ".join(f"{row[c]:>14}" for c in cfg_names))
    lines.append("-" * len(header))
    for label in ("FP", "FN"):
        totals = {c: _totals(table, c, label) for c in cfg_names}
        lines.append(f"{label + ' total':<28} " + " ".join(f"{totals[c]:>14}" for c in cfg_names))
    print("\n".join(lines))
    # Non-empty is enough here; the matrix test pins the specific traces. Adding
    # new experiment traces later must not break this print-only rig.
    assert traces


def test_experiment_confusion_matrix_is_stable() -> None:
    """Lock the measured outcome so the NOP-4 finding is a regression, not a memory.

    Key reads:
    - naive git (net or size) MASKS ``useless_growth`` (b) -> FN, despite tests
      catching it -> violates "no masking when tests present".
    - the ``net`` scalar also false-stops the in-place bug fix (c) -> FP; ``size``
      does not (additive-bias fix).
    - the *test-deferring* ``size`` anchor strictly dominates the baseline: it
      catches the no-test churn (d) the baseline misses, adds no FP, masks nothing.
    """
    _, _, table = _matrix()
    expected = {
        "no_progress:src/b.py": {
            "baseline": "TP",
            "+git_net": "FN",
            "+git_size": "FN",
            "+git_net_cond": "TP",
            "+git_size_cond": "TP",
        },
        "no_progress:src/d.py": {
            "baseline": "FN",
            "+git_net": "TP",
            "+git_size": "TP",
            "+git_net_cond": "TP",
            "+git_size_cond": "TP",
        },
        "healthy:src/c.py": {
            "baseline": "TN",
            "+git_net": "FP",
            "+git_size": "TN",
            "+git_net_cond": "FP",
            "+git_size_cond": "TN",
        },
        "healthy:src/a.py": {
            "baseline": "FP",
            "+git_net": "TN",
            "+git_size": "TN",
            "+git_net_cond": "FP",
            "+git_size_cond": "FP",
        },
        "healthy:src/e.py": {
            "baseline": "FP",
            "+git_net": "TN",
            "+git_size": "TN",
            "+git_net_cond": "FP",
            "+git_size_cond": "FP",
        },
    }
    # Subset check: pin the 5 decisive traces without forbidding the directory
    # from growing later (load_traces picks up every *.json).
    pinned = set(expected)
    for name, row in expected.items():
        assert table[name] == row, name

    # Naive git masks a test-caught stall -> rejected.
    assert table["no_progress:src/b.py"]["+git_net"] == "FN"
    assert table["no_progress:src/b.py"]["+git_size"] == "FN"
    # Test-deferring git never masks tests (over the pinned traces).
    assert _totals(table, "+git_size_cond", "FN", pinned) == 0
    # Test-deferring size dominates baseline: catches d, no new FP vs baseline.
    assert table["no_progress:src/d.py"]["+git_size_cond"] == "TP"
    assert _totals(table, "+git_size_cond", "FP", pinned) == _totals(
        table, "baseline", "FP", pinned
    )
