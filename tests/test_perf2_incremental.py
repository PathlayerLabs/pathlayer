"""PERF-2: equivalence, rollback and complexity tests for the incremental fold.

``evaluate``/``measure``/``is_satisfied``/``is_advancing`` stay the pure batch
specification (untouched by PERF-2, see the module docstrings in
``pathlayer.detectors.progress``, ``pathlayer.detectors.no_progress`` and
``pathlayer.pipeline``). Everything here verifies that the *new* incremental
fold (``create_state``/``step``) is a faster, byte-for-byte equivalent
implementation of that same specification -- prefix by prefix, not just on the
final result -- plus the transactional/rollback contract that makes it safe to
wire into ``Pipeline``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import pytest

from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    GitProgressAnchor,
    ProgressAnchor,
    TestProgressAnchor,
    coding_anchors,
    is_advancing,
    loop_suppression_anchors,
)
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.policies.base import PolicyConfig, SuppressionRule
from pathlayer.validation import Trace, load_traces

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"
_PROGRESS_EVIDENCE_KEY = "progress"


def _all_traces() -> list[Trace]:
    traces: list[Trace] = []
    for directory in ("loop", "no_progress", "healthy", "nop4_experiment"):
        path = _TRACES_DIR / directory
        if path.is_dir():
            traces.extend(load_traces(path))
    return traces


_ALL_TRACES = _all_traces()
_ALL_TRACE_IDS = [f"{t.label}:{t.description[:30]}" for t in _ALL_TRACES]


# --------------------------------------------------------------------------- #
# Generated corpus: fixed seeds, deliberately adversarial mixes
# --------------------------------------------------------------------------- #
def _random_events(
    seed: int,
    n: int,
    *,
    targets: Sequence[str] = ("a", "b"),
    malformed: bool = False,
    duplicates: bool = False,
    out_of_order_ts: bool = False,
    interleave_llm: bool = True,
    high_cardinality: bool = False,
) -> list[ToolEvent]:
    """A deliberately adversarial mix of test/exit_code/git/edit/LLM events."""
    rng = random.Random(seed)
    events: list[ToolEvent] = []
    last_event: ToolEvent | None = None
    for i in range(n):
        if duplicates and last_event is not None and rng.random() < 0.1:
            events.append(last_event)
            continue
        target = f"suite_{i}" if high_cardinality else rng.choice(targets)
        ts = float(n - i) if out_of_order_ts else float(i)
        r = rng.random()
        if r < 0.3:
            passed = "oops" if malformed and rng.random() < 0.1 else rng.randint(0, 6)
            failed = rng.randint(0, 4)
            event = ToolEvent(
                tool="run_tests",
                args={"target": target},
                result={"passed": passed, "failed": failed},
                ts=ts,
            )
        elif r < 0.5:
            exit_code = "boom" if malformed and rng.random() < 0.1 else rng.choice([0, 0, 1])
            event = ToolEvent(tool="run_cmd", result={"exit_code": exit_code}, ts=ts)
        elif r < 0.65:
            ins = rng.randint(0, 12)
            dele = rng.randint(0, 12)
            event = ToolEvent(tool="git_diff", result={"insertions": ins, "deletions": dele}, ts=ts)
        elif interleave_llm and r < 0.8:
            event = ToolEvent(tool="llm", kind=EventKind.LLM_CALL, ts=ts, cost_usd=0.01)
        else:
            event = ToolEvent(tool="edit_file", args={"path": f"f{i}.py"}, ts=ts)
        events.append(event)
        last_event = event
    return events


_SEEDS = list(range(12))
_GENERATED_CORPUS: list[list[ToolEvent]] = [
    _random_events(seed, 260, malformed=True, duplicates=True, high_cardinality=(seed % 4 == 0))
    for seed in _SEEDS
] + [
    _random_events(0, 260, out_of_order_ts=True),
    _random_events(1, 40, targets=("only_one",)),  # minimal-window-ish, few targets
]


def _anchors() -> list[ProgressAnchor]:
    return [TestProgressAnchor(), ExitCodeProgressAnchor(), GitProgressAnchor()]


# --------------------------------------------------------------------------- #
# A. ProgressAnchor.step vs measure/is_satisfied, prefix by prefix
# --------------------------------------------------------------------------- #
def _assert_anchor_step_matches_batch(anchor_factory, events: Sequence[ToolEvent]) -> None:
    anchor = anchor_factory()
    state = anchor.create_state()
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, snapshot = anchor.step(state, event)
        expected_level = anchor.measure(history)
        assert snapshot.level == expected_level, (len(history), snapshot.level, expected_level)
        expected_satisfied = anchor.is_satisfied(history) if expected_level is not None else False
        assert snapshot.satisfied == expected_satisfied or (
            expected_level is None and snapshot.satisfied == anchor.is_satisfied(history)
        )


@pytest.mark.parametrize(
    "events", _GENERATED_CORPUS, ids=[f"seed{i}" for i in range(len(_GENERATED_CORPUS))]
)
@pytest.mark.parametrize(
    "anchor_factory", [TestProgressAnchor, ExitCodeProgressAnchor, GitProgressAnchor]
)
def test_anchor_step_matches_batch_on_generated_corpus(anchor_factory, events) -> None:
    _assert_anchor_step_matches_batch(anchor_factory, events)


@pytest.mark.parametrize("trace", _ALL_TRACES, ids=_ALL_TRACE_IDS)
@pytest.mark.parametrize(
    "anchor_factory", [TestProgressAnchor, ExitCodeProgressAnchor, GitProgressAnchor]
)
def test_anchor_step_matches_batch_on_golden_traces(anchor_factory, trace: Trace) -> None:
    _assert_anchor_step_matches_batch(anchor_factory, trace.events)


def test_anchor_step_matches_batch_on_empty_history() -> None:
    for anchor_factory in (TestProgressAnchor, ExitCodeProgressAnchor, GitProgressAnchor):
        anchor = anchor_factory()
        assert anchor.measure([]) is None


class _CustomAnchor(ProgressAnchor):
    """A custom anchor that only implements measure/is_satisfied (no PERF-2 override).

    Stands in for the NOP-4 experiment's prototype anchor: proves the default
    batch-rebuild fallback keeps a third-party anchor correct without any
    changes on its part.
    """

    name = "custom_net_lines"

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        total = 0
        seen = False
        for event in history:
            if "net_lines" in event.result if isinstance(event.result, dict) else False:
                seen = True
                total += event.result["net_lines"]
        return float(total) if seen else None


class _LegacyDetector(Detector):
    """A detector predating PERF-2: implements only ``evaluate``."""

    name = "legacy"

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        return Signal(self.name, confidence=1.0 if len(history) >= 3 else 0.0)


def test_detector_base_default_create_state_and_step_fallback() -> None:
    # Direct unit coverage of Detector.create_state()/step() (PERF-2's default
    # fallback for any detector that never heard of the incremental fold): the
    # state is a growing tuple, and step(state, event) == evaluate((*state, event)).
    detector = _LegacyDetector()
    state = detector.create_state()
    assert state == ()

    events = [ToolEvent(tool="a"), ToolEvent(tool="b"), ToolEvent(tool="c")]
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, signal = detector.step(state, event)
        assert state == tuple(history)
        assert signal.confidence == detector.evaluate(tuple(history)).confidence


def test_custom_anchor_without_perf2_override_uses_batch_fallback() -> None:
    events = [
        ToolEvent(tool="edit", result={"net_lines": 3}),
        ToolEvent(tool="edit", result={"net_lines": 5}),
        ToolEvent(tool="noop", args={}),
        ToolEvent(tool="edit", result={"net_lines": -2}),
    ]
    _assert_anchor_step_matches_batch(_CustomAnchor, events)
    # Sanity: create_state()/step() really are the generic tuple-history fallback.
    anchor = _CustomAnchor()
    assert anchor.create_state() == ()
    state, snapshot = anchor.step((), events[0])
    assert state == (events[0],)
    assert snapshot.level == 3.0


# --------------------------------------------------------------------------- #
# B. NoProgressDetector.step vs evaluate, prefix by prefix
# --------------------------------------------------------------------------- #
def _assert_detector_step_matches_batch(
    detector: NoProgressDetector, events: Sequence[ToolEvent]
) -> None:
    state = detector.create_state()
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, step_signal = detector.step(state, event)
        batch_signal = detector.evaluate(tuple(history))
        assert step_signal.confidence == pytest.approx(batch_signal.confidence, abs=1e-9), (
            len(history),
            step_signal,
            batch_signal,
        )
        assert step_signal.reason == batch_signal.reason, (len(history), step_signal, batch_signal)


@pytest.mark.parametrize(
    "events", _GENERATED_CORPUS, ids=[f"seed{i}" for i in range(len(_GENERATED_CORPUS))]
)
def test_no_progress_detector_step_matches_batch_on_generated_corpus(events) -> None:
    detector = NoProgressDetector(_anchors(), window=11, target_rate=0.2, threshold=2.0)
    _assert_detector_step_matches_batch(detector, events)


@pytest.mark.parametrize("trace", _ALL_TRACES, ids=_ALL_TRACE_IDS)
def test_no_progress_detector_step_matches_batch_on_golden_traces(trace: Trace) -> None:
    detector = NoProgressDetector(coding_anchors())
    _assert_detector_step_matches_batch(detector, trace.events)


@pytest.mark.parametrize("window", [11, 12, 64])
def test_no_progress_detector_step_matches_batch_across_windows(window: int) -> None:
    detector = NoProgressDetector(_anchors(), window=window, target_rate=0.2, threshold=2.0)
    _assert_detector_step_matches_batch(detector, _GENERATED_CORPUS[0])


def test_no_progress_detector_step_matches_batch_with_evicted_tie_at_max() -> None:
    # Exercise the monotone deque's tie-preservation and front-eviction together:
    # a high value, then a run of ties, then enough steps to evict the original
    # maximum out of the window while ties remain inside it.
    events = [
        ToolEvent(tool="run_tests", result={"passed": 20, "failed": 0}),  # level 20 (record)
    ]
    events += [
        ToolEvent(tool="run_tests", result={"passed": 15, "failed": 0}) for _ in range(15)
    ]  # level 15, tied repeatedly (never beats 20)
    detector = NoProgressDetector([TestProgressAnchor()], window=11, target_rate=0.2, threshold=2.0)
    _assert_detector_step_matches_batch(detector, events)


def test_no_progress_detector_step_matches_batch_first_evidence_mid_window() -> None:
    # Several non-coding events before the first applicable evidence appears
    # inside a window -- the deque must not count the leading gap as stagnation.
    events = [ToolEvent(tool="noop", args={"i": i}) for i in range(5)]
    events += [ToolEvent(tool="run_tests", result={"passed": 5, "failed": 0}) for _ in range(8)]
    detector = NoProgressDetector([TestProgressAnchor()], window=11, target_rate=0.2, threshold=2.0)
    _assert_detector_step_matches_batch(detector, events)


def test_no_progress_detector_step_matches_batch_with_non_finite_levels() -> None:
    # NaN is normalized to a gap (like None) in both `_cusum` and `_push_level`
    # -- required, not cosmetic: NaN fails every float comparison, so appending
    # it into the monotone deque unconditionally would corrupt its pop-while
    # invariant (a later, larger level would no longer evict it). +inf/-inf
    # behave like any other float (total order holds), so only NaN needs the
    # special case; this exercises both.
    class _NonFiniteAnchor(ProgressAnchor):
        name = "non_finite"

        def __init__(self, levels: Sequence[float]) -> None:
            self._levels = list(levels)

        def measure(self, history: Sequence[ToolEvent]) -> float | None:
            n = len(history)
            return self._levels[n - 1] if 0 < n <= len(self._levels) else None

    levels = [
        1.0,
        float("nan"),
        float("nan"),
        float("inf"),
        float("inf"),
        2.0,
        float("-inf"),
        float("-inf"),
        3.0,
    ]
    events = [ToolEvent(tool="t", ts=float(i)) for i in range(len(levels))]
    detector = NoProgressDetector(
        [_NonFiniteAnchor(levels)], window=11, target_rate=0.2, threshold=2.0
    )
    _assert_detector_step_matches_batch(detector, events)


def test_no_progress_detector_step_matches_batch_with_mixed_fast_and_fallback_anchors() -> None:
    # A detector wiring one PERF-2 anchor (TestProgressAnchor, O(1) amortized)
    # alongside a custom anchor stuck on the default batch fallback
    # (_CustomAnchor, O(n) per event for itself only) must still be exactly
    # equivalent to evaluate() -- the two anchor kinds are combined by the same
    # `min(applicable_cusums)` aggregation regardless of how each folds.
    events = [
        ToolEvent(tool="run_tests", result={"passed": 5, "failed": 0}),
        ToolEvent(tool="edit", result={"net_lines": 3}),
        ToolEvent(tool="run_tests", result={"passed": 5, "failed": 0}),
        ToolEvent(tool="edit", result={"net_lines": 3}),
        ToolEvent(tool="run_tests", result={"passed": 5, "failed": 0}),
        ToolEvent(tool="edit", result={"net_lines": 5}),
        ToolEvent(tool="run_tests", result={"passed": 5, "failed": 0}),
        ToolEvent(tool="edit", result={"net_lines": 3}),
        ToolEvent(tool="run_tests", result={"passed": 5, "failed": 0}),
        ToolEvent(tool="edit", result={"net_lines": 3}),
        ToolEvent(tool="run_tests", result={"passed": 5, "failed": 0}),
        ToolEvent(tool="edit", result={"net_lines": 3}),
    ]
    detector = NoProgressDetector(
        [TestProgressAnchor(), _CustomAnchor()], window=11, target_rate=0.2, threshold=2.0
    )
    _assert_detector_step_matches_batch(detector, events)


def test_no_progress_detector_step_matches_batch_on_test_appearing_mid_run() -> None:
    # ExitCode/Git go from applicable to permanently deferred the moment a test
    # event appears anywhere in the history -- the sharpest edge in the fold.
    events = [ToolEvent(tool="run_cmd", result={"exit_code": 0}) for _ in range(6)]
    events += [ToolEvent(tool="run_tests", result={"passed": 3, "failed": 1})]
    events += [ToolEvent(tool="run_cmd", result={"exit_code": 0}) for _ in range(6)]
    detector = NoProgressDetector(_anchors(), window=11, target_rate=0.2, threshold=2.0)
    _assert_detector_step_matches_batch(detector, events)


# --------------------------------------------------------------------------- #
# C. Pipeline-level equivalence: incremental vs a legacy-forced NoProgressDetector
# --------------------------------------------------------------------------- #
class _BatchOnlyWrapper(Detector):
    """Wraps a detector's ``evaluate`` without exposing an overridden ``step``.

    ``Pipeline`` detects whether a detector overrides ``step`` via
    ``type(detector).step is Detector.step``; wrapping forces the *legacy*
    dispatch path (shared ``tuple(history)`` rebuild, ``evaluate`` called every
    tick) for a detector that would otherwise take the O(1)-amortized path, so
    a test can diff the two dispatch strategies against each other using the
    real ``Pipeline.process`` for both sides -- not a reimplementation of it.
    """

    def __init__(self, inner: Detector) -> None:
        self._inner = inner
        self.name = inner.name
        self.event_kinds = inner.event_kinds

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        return self._inner.evaluate(history)


def _wired_pipeline(*, legacy_no_progress: bool, dry_run: bool = True) -> Pipeline:
    from pathlayer.detectors.loop import LoopDetector
    from pathlayer.detectors.retry import RetryDetector

    no_progress = NoProgressDetector(coding_anchors())
    detectors: list[Detector] = [
        LoopDetector(),
        RetryDetector(),
        _BatchOnlyWrapper(no_progress) if legacy_no_progress else no_progress,
    ]
    suppression = SuppressionRule(evidence=_PROGRESS_EVIDENCE_KEY, min_confidence=1.0)
    policy = Policy(PolicyConfig(suppressions={"loop": suppression}))
    return Pipeline(
        detectors,
        policy,
        dry_run=dry_run,
        progress_anchors=loop_suppression_anchors(),
    )


def _run_and_capture(pipeline: Pipeline, events: Sequence[ToolEvent]) -> tuple:
    actions = pipeline.run(events)
    would_be = tuple((w.index, w.action, w.reason) for w in pipeline.would_be_actions)
    return tuple(actions), would_be


@pytest.mark.parametrize(
    "events", _GENERATED_CORPUS, ids=[f"seed{i}" for i in range(len(_GENERATED_CORPUS))]
)
def test_pipeline_incremental_matches_legacy_dispatch_on_generated_corpus(events) -> None:
    fast = _run_and_capture(_wired_pipeline(legacy_no_progress=False), events)
    slow = _run_and_capture(_wired_pipeline(legacy_no_progress=True), events)
    assert fast == slow


@pytest.mark.parametrize("trace", _ALL_TRACES, ids=_ALL_TRACE_IDS)
def test_pipeline_incremental_matches_legacy_dispatch_on_golden_traces(trace: Trace) -> None:
    fast = _run_and_capture(_wired_pipeline(legacy_no_progress=False), trace.events)
    slow = _run_and_capture(_wired_pipeline(legacy_no_progress=True), trace.events)
    assert fast == slow


def test_pipeline_enforcement_mode_stops_identically() -> None:
    # Same comparison, but in enforcement mode (not dry-run) on a trace known to
    # stop, so the STOP-then-halt path is covered too.
    trace = load_traces(_TRACES_DIR / "no_progress")[0]
    for t in load_traces(_TRACES_DIR / "no_progress"):
        fast = _run_and_capture(_wired_pipeline(legacy_no_progress=False, dry_run=False), t.events)
        slow = _run_and_capture(_wired_pipeline(legacy_no_progress=True, dry_run=False), t.events)
        assert fast == slow
    del trace


def test_build_pipeline_default_config_is_unaffected() -> None:
    # build_pipeline() itself is untouched by PERF-2 (Pipeline wires the
    # incremental path internally) -- a smoke test that the public entry point
    # still produces a working, non-crashing pipeline end to end.
    pipeline = build_pipeline(PathlayerConfig())
    events = _GENERATED_CORPUS[0]
    actions = pipeline.run(events)
    assert len(actions) <= len(events)


# --------------------------------------------------------------------------- #
# D. Transactional rollback: exceptions must leave every prior state untouched
# --------------------------------------------------------------------------- #
class _BoomOnNthAnchor(ProgressAnchor):
    """Raises from ``step`` on the Nth call; batch ``measure`` never raises."""

    name = "boom_anchor"

    def __init__(self, boom_on: int) -> None:
        self._boom_on = boom_on
        self._calls = 0

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        return float(len(history))

    def create_state(self) -> int:
        return 0

    def step(self, state: int, event: ToolEvent):
        self._calls += 1
        if self._calls == self._boom_on:
            raise RuntimeError("anchor boom")
        from pathlayer.detectors.progress import AnchorSnapshot

        return state + 1, AnchorSnapshot(level=float(state + 1), satisfied=False)


def test_anchor_exception_rolls_back_detector_state_and_history() -> None:
    anchor = _BoomOnNthAnchor(boom_on=2)
    detector = NoProgressDetector([anchor], window=11, target_rate=0.2, threshold=2.0)
    pipeline = Pipeline([detector])

    pipeline.process(ToolEvent(tool="a"))
    state_before = pipeline._detector_states[0]  # white-box rollback check

    with pytest.raises(RuntimeError, match="anchor boom"):
        pipeline.process(ToolEvent(tool="b"))

    assert pipeline._detector_states[0] == state_before
    assert len(pipeline.history) == 1


class _BoomDetector(Detector):
    name = "boom_detector"

    def __init__(self, fail_on: int) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("boom")
        return Signal(self.name, confidence=0.0)


def test_later_detector_exception_rolls_back_earlier_incremental_detector_state() -> None:
    # detector A (incremental) must succeed and compute a next state on the
    # failing tick; detector B (raising) fails afterwards. A's *committed*
    # state must still be the OLD one -- Pipeline must not partially commit.
    no_progress = NoProgressDetector(coding_anchors())
    boom = _BoomDetector(fail_on=2)
    pipeline = Pipeline([no_progress, boom])

    pipeline.process(ToolEvent(tool="run_tests", result={"passed": 1, "failed": 0}))
    state_before = pipeline._detector_states[0]

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.process(ToolEvent(tool="run_tests", result={"passed": 2, "failed": 0}))

    assert pipeline._detector_states[0] == state_before
    assert len(pipeline.history) == 1


def test_policy_exception_rolls_back_all_detector_and_evidence_state() -> None:
    class _BoomPolicy(Policy):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def step(self, state, signals, evidence=None):  # type: ignore[override]
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("policy boom")
            return super().step(state, signals, evidence or {})

    no_progress = NoProgressDetector(coding_anchors())
    pipeline = Pipeline(
        [no_progress],
        _BoomPolicy(),
        progress_anchors=loop_suppression_anchors(),
    )
    pipeline.process(ToolEvent(tool="run_tests", result={"passed": 1, "failed": 0}))
    detector_state_before = pipeline._detector_states[0]
    evidence_state_before = pipeline._evidence_state

    with pytest.raises(RuntimeError, match="policy boom"):
        pipeline.process(ToolEvent(tool="run_tests", result={"passed": 2, "failed": 0}))

    assert pipeline._detector_states[0] == detector_state_before
    assert pipeline._evidence_state == evidence_state_before
    assert len(pipeline.history) == 1


def test_evidence_anchor_exception_rolls_back_evidence_state_in_isolation() -> None:
    # Same rollback contract as test_anchor_exception_rolls_back_detector_state,
    # but exercised on the LOOP-2 progress-evidence tracker specifically (no
    # NoProgressDetector in the mix): a raising progress_anchors anchor must
    # leave _evidence_state exactly as it was, with no detector involved at all.
    anchor = _BoomOnNthAnchor(boom_on=2)
    pipeline = Pipeline([], progress_anchors=[anchor])

    pipeline.process(ToolEvent(tool="a"))
    evidence_state_before = pipeline._evidence_state

    with pytest.raises(RuntimeError, match="anchor boom"):
        pipeline.process(ToolEvent(tool="b"))

    assert pipeline._evidence_state == evidence_state_before
    assert len(pipeline.history) == 1


def test_retry_after_failure_reproduces_the_same_next_state() -> None:
    # After a rollback, retrying with the *same* event must be indistinguishable
    # from the failure never having happened (determinism across a retry).
    anchor = _BoomOnNthAnchor(boom_on=2)
    detector = NoProgressDetector([anchor], window=11, target_rate=0.2, threshold=2.0)
    pipeline = Pipeline([detector])
    pipeline.process(ToolEvent(tool="a"))

    with pytest.raises(RuntimeError):
        pipeline.process(ToolEvent(tool="b"))

    # A fresh, non-raising anchor call this time (Nth-call counter already
    # tripped and won't trip again since boom_on=2 already happened).
    action = pipeline.process(ToolEvent(tool="b"))
    assert action is Action.NONE
    assert len(pipeline.history) == 2


def test_two_pipelines_sharing_detector_instances_do_not_cross_talk() -> None:
    # Incremental state lives in Pipeline, never on the shared Detector/anchor
    # instance (architecture guardrail): two pipelines reusing the same
    # NoProgressDetector must not interfere with each other's state.
    shared_detector = NoProgressDetector(coding_anchors())
    pipeline_a = Pipeline([shared_detector])
    pipeline_b = Pipeline([shared_detector])

    flat = [ToolEvent(tool="run_tests", result={"passed": 5, "failed": 3}) for _ in range(20)]
    actions_a = pipeline_a.run(flat)  # stops early (run() halts on the first STOP)
    # pipeline_b starts completely fresh despite sharing the exact same
    # Detector instance that pipeline_a just drove to a high confidence.
    first_action_b = pipeline_b.process(flat[0])
    assert first_action_b is Action.NONE
    assert actions_a[-1] is Action.STOP
    assert len(pipeline_a.history) < len(flat)  # halted before exhausting the trace
    assert len(pipeline_b.history) == 1


def test_state_frozen_after_stop() -> None:
    no_progress = NoProgressDetector(coding_anchors())
    pipeline = Pipeline([no_progress])
    flat = [ToolEvent(tool="run_tests", result={"passed": 5, "failed": 3}) for _ in range(20)]
    pipeline.run(flat)
    assert pipeline.stopped
    stopped_at = len(pipeline.history)
    state_after_stop = pipeline._detector_states[0]

    # Further events are ignored entirely once stopped -- no detector.step call.
    assert pipeline.process(ToolEvent(tool="anything")) is Action.STOP
    assert pipeline._detector_states[0] == state_after_stop
    assert len(pipeline.history) == stopped_at


def test_long_dry_run_stays_consistent_with_batch() -> None:
    detector = NoProgressDetector(coding_anchors())
    events = _random_events(7, 600, malformed=True, duplicates=True)
    _assert_detector_step_matches_batch(detector, events)


# --------------------------------------------------------------------------- #
# E. Complexity: exactly one transition per anchor per relevant event, bounded
# --------------------------------------------------------------------------- #
class _CountingAnchor(ProgressAnchor):
    name = "counting"

    def __init__(self) -> None:
        self.step_calls = 0
        self._inner_state = TestProgressAnchor().create_state()

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        return TestProgressAnchor().measure(history)

    def is_satisfied(self, history: Sequence[ToolEvent]) -> bool:
        return TestProgressAnchor().is_satisfied(history)

    def create_state(self):
        return TestProgressAnchor().create_state()

    def step(self, state, event: ToolEvent):
        self.step_calls += 1
        return TestProgressAnchor().step(state, event)


def test_exactly_one_transition_per_anchor_per_relevant_event() -> None:
    anchor = _CountingAnchor()
    detector = NoProgressDetector([anchor], window=11, target_rate=0.2, threshold=2.0)
    pipeline = Pipeline([detector])
    n = 50
    events = [
        ToolEvent(tool="run_tests", result={"passed": i, "failed": 0}, ts=float(i))
        for i in range(n)
    ]
    pipeline.run(events)
    assert anchor.step_calls == n


def test_no_transitions_after_stop() -> None:
    anchor = _CountingAnchor()
    detector = NoProgressDetector([anchor], window=11, target_rate=0.2, threshold=2.0)
    pipeline = Pipeline([detector])
    flat = [ToolEvent(tool="run_tests", result={"passed": 5, "failed": 3}) for _ in range(20)]
    pipeline.run(flat)
    assert pipeline.stopped
    calls_at_stop = anchor.step_calls
    pipeline.process(ToolEvent(tool="more"))
    pipeline.process(ToolEvent(tool="more"))
    assert anchor.step_calls == calls_at_stop


def test_llm_events_do_not_advance_no_progress_anchor_state() -> None:
    # event_kinds filtering: an LLM_CALL event must not trigger a transition at
    # all (not even a "no-op" one) for a detector scoped to TOOL_CALL only.
    anchor = _CountingAnchor()
    detector = NoProgressDetector([anchor], window=11, target_rate=0.2, threshold=2.0)
    state = detector.create_state()
    llm_event = ToolEvent(tool="llm", kind=EventKind.LLM_CALL, cost_usd=0.01)
    state, signal = detector.step(state, llm_event)
    assert anchor.step_calls == 0
    assert signal.confidence == 0.0


def test_test_anchor_state_memory_grows_with_unique_targets_not_history_length() -> None:
    # Documented, not "fixed": _TestAnchorState.legacy_latest holds one entry per
    # distinct suite (_test_target) ever seen, so its size is O(unique targets)
    # -- independent of how many events were folded in for a *fixed* target set,
    # but unbounded if the agent keeps inventing new target names.
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    same_target_event = ToolEvent(tool="run_tests", result={"passed": 1, "failed": 0})
    for _ in range(500):
        state, _ = anchor.step(state, same_target_event)
    assert len(state.legacy_latest) == 1  # same target every time -> stays at 1 entry

    state = anchor.create_state()
    for i in range(500):
        event = ToolEvent(
            tool="run_tests", args={"target": f"suite_{i}"}, result={"passed": 1, "failed": 0}
        )
        state, _ = anchor.step(state, event)
    assert len(state.legacy_latest) == 500  # 500 distinct suites -> 500 entries


def test_no_progress_window_bounds_deque_size() -> None:
    detector = NoProgressDetector(coding_anchors(), window=11, target_rate=0.2, threshold=2.0)
    state = detector.create_state()
    events = [ToolEvent(tool="run_tests", result={"passed": i, "failed": 0}) for i in range(500)]
    for event in events:
        state, _ = detector.step(state, event)
    for runtime in state.runtimes:
        assert len(runtime.entries) <= 11


# --------------------------------------------------------------------------- #
# F. Progress-evidence tracker (LOOP-2 suppression) equivalence to is_advancing
# --------------------------------------------------------------------------- #
def _assert_evidence_matches_is_advancing(events: Sequence[ToolEvent]) -> None:
    from pathlayer.pipeline import _create_evidence_state, _step_evidence

    anchors = loop_suppression_anchors()
    state = _create_evidence_state(anchors)
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, advancing = _step_evidence(state, anchors, event)
        expected = is_advancing(anchors, history)
        assert advancing == expected, (len(history), advancing, expected)


@pytest.mark.parametrize(
    "events", _GENERATED_CORPUS, ids=[f"seed{i}" for i in range(len(_GENERATED_CORPUS))]
)
def test_progress_evidence_matches_is_advancing_on_generated_corpus(events) -> None:
    _assert_evidence_matches_is_advancing(events)


@pytest.mark.parametrize("trace", _ALL_TRACES, ids=_ALL_TRACE_IDS)
def test_progress_evidence_matches_is_advancing_on_golden_traces(trace: Trace) -> None:
    _assert_evidence_matches_is_advancing(trace.events)


# --------------------------------------------------------------------------- #
# G. Documented caller contract: mutating a past event is undefined behavior
# --------------------------------------------------------------------------- #
def test_mutating_a_past_event_is_undefined_behavior() -> None:
    """Characterization test for the ToolEvent immutability contract (PERF-2).

    Pins *why* retroactive mutation of an already-processed event is out of
    contract: the incremental fold extracted its scalars at fold time and
    never re-reads the event, so it does not see a later mutation, while
    `evaluate` re-derives from the live event on every call and does. This is
    documented (see `ToolEvent`'s docstring), not a bug to fix.
    """
    anchor = TestProgressAnchor()
    event = ToolEvent(tool="run_tests", result={"passed": 1, "failed": 0})
    state = anchor.create_state()
    state, snapshot = anchor.step(state, event)
    assert snapshot.level == 1.0

    # Retroactively mutate the *same* event object after it was folded in.
    event.result["passed"] = 100

    assert anchor.measure([event]) == 100.0  # batch path: sees the mutation
    assert snapshot.level == 1.0  # already-computed incremental snapshot: does not
