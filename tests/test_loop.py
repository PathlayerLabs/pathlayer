"""Tests for the LoopDetector (LOOP-1 exact repeat + LOOP-2 n-gram coverage).

Cover the two directions that matter: a trivial loop must produce high
confidence; a healthy, varied run must stay near zero. Also pin the K
threshold, determinism, and the synergy with canonicalization (a loop whose
arguments carry fresh volatile noise each turn is still caught).

LOOP-2 tests additionally prove: (1) a real recall gap that LOOP-1 alone
misses (an interrupted multi-action cycle), (2) LOOP-2 never boosts a loop
LOOP-1 already sees (non-regression by construction), (3) LOOP-2 stays capped
and off on healthy adversarial patterns, and (4) non-redundancy with RET-2
(RetryDetector), which has a different niche (single tool, uninterrupted run,
requires failures).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import pytest

from pathlayer.config import build_pipeline
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent
from pathlayer.policies import Action
from pathlayer.validation import load_trace

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _repeat(block: list[tuple[str, dict[str, object]]], times: int) -> list[ToolEvent]:
    return [ToolEvent(tool=tool, args=args) for _ in range(times) for tool, args in block]


def test_empty_history_is_zero() -> None:
    assert LoopDetector().evaluate([]).confidence == 0.0


def test_healthy_varied_run_is_zero() -> None:
    history = [
        ToolEvent(tool="read", args={"path": "a.py"}),
        ToolEvent(tool="edit", args={"path": "a.py"}),
        ToolEvent(tool="run_tests", args={}),
        ToolEvent(tool="read", args={"path": "b.py"}),
        ToolEvent(tool="edit", args={"path": "b.py"}),
    ]
    assert LoopDetector().evaluate(history).confidence == 0.0


def test_single_action_loop_detected() -> None:
    history = _repeat([("read", {"path": "a.py"})], times=6)
    signal = LoopDetector().evaluate(history)
    assert signal.confidence > 0.5


def test_two_action_loop_detected() -> None:
    history = _repeat([("list", {"p": "x"}), ("read", {"p": "y"})], times=6)
    signal = LoopDetector().evaluate(history)
    assert signal.confidence > 0.5


def test_confidence_increases_with_repetitions() -> None:
    few = LoopDetector().evaluate(_repeat([("read", {"p": "a"})], times=3))
    many = LoopDetector().evaluate(_repeat([("read", {"p": "a"})], times=8))
    assert many.confidence > few.confidence


def test_confidence_in_unit_interval() -> None:
    signal = LoopDetector().evaluate(_repeat([("read", {"p": "a"})], times=20))
    assert 0.0 <= signal.confidence <= 1.0


def test_k_threshold_blocks_short_loops() -> None:
    detector = LoopDetector(k=4)
    # Below k, LOOP-1 is silent -- but the same 3x repeat also forms a 2-gram
    # with hits=2 (below the default ngram_min_hits=3 ramp center), so LOOP-3's
    # continuous fallback now reports a small-but-nonzero score instead of
    # exactly 0.0 (see test_ngram_min_hits_blocks_short_pattern for the general
    # case). Still far below anything that could matter: nowhere near
    # stop_threshold, and dwarfed by the k=4 case below.
    below_k = detector.evaluate(_repeat([("read", {"p": "a"})], times=3)).confidence
    assert 0.0 <= below_k < 0.1
    # Repeated 4 times: at k -> LOOP-1 fires with full authority (saturating
    # map: reps == k -> exactly 0.5).
    at_k = detector.evaluate(_repeat([("read", {"p": "a"})], times=4)).confidence
    assert at_k >= 0.5
    assert at_k > below_k


def test_detects_loop_despite_volatile_args() -> None:
    # Same logical action each turn, but a fresh request_id every time.
    history = [
        ToolEvent(tool="fetch", args={"url": "/status", "request_id": f"r-{i}"}) for i in range(8)
    ]
    assert LoopDetector().evaluate(history).confidence > 0.5


def test_is_deterministic() -> None:
    history = _repeat([("a", {}), ("b", {})], times=5)
    detector = LoopDetector()
    assert detector.evaluate(history) == detector.evaluate(history)


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be"):
        LoopDetector(k=1)


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        LoopDetector(window=3)  # default k=2 needs window >= 4


def test_window_detects_recent_loop_in_long_history() -> None:
    # A long, varied prefix then a clear loop in the recent tail: still caught,
    # and per-step cost stays bounded by the window.
    prefix = [ToolEvent(tool=f"step_{i}", args={}) for i in range(50)]
    loop = _repeat([("read", {"p": "a"})], times=6)
    assert LoopDetector(window=12).evaluate(prefix + loop).confidence > 0.5


def test_window_ignores_loop_outside_window() -> None:
    # The loop has scrolled out of the window; only the varied tail remains.
    loop = _repeat([("read", {"p": "a"})], times=6)
    tail = [ToolEvent(tool=f"step_{i}", args={}) for i in range(20)]
    assert LoopDetector(window=10).evaluate(loop + tail).confidence == 0.0


def test_fixture_loop_has_high_confidence(trace: Callable[[str], list[ToolEvent]]) -> None:
    history = trace("loop_trivial.json")
    signal = LoopDetector().evaluate(history)
    assert signal.confidence >= 0.9
    assert signal.name == "loop"


# --------------------------------------------------------------------------- #
# LOOP-2: n-gram coverage fallback
# --------------------------------------------------------------------------- #


def _interrupted_cycle(cycles: int) -> list[ToolEvent]:
    """A 2-action cycle (a, b) interrupted by a distinct one-off action every
    single cycle. LOOP-1's trailing exact-repeat match resets on every
    interruption (never reaches 2 consecutive reps), so it stays silent
    however long the run goes on -- the proven LOOP-2 recall gap.
    """
    history: list[ToolEvent] = []
    for i in range(cycles):
        history.append(ToolEvent(tool="a", args={}))
        history.append(ToolEvent(tool="b", args={}))
        history.append(ToolEvent(tool=f"probe_{i}", args={"n": i}))
    return history


def test_loop1_alone_is_silent_on_interrupted_cycle() -> None:
    # Ground truth for the gap: with the n-gram fallback effectively disabled
    # (coverage can never reach 1.0), only LOOP-1 is left -- and it sees nothing.
    history = _interrupted_cycle(cycles=10)
    assert LoopDetector(ngram_min_coverage=1.0).evaluate(history).confidence == 0.0


def test_ngram_fallback_catches_interrupted_cycle() -> None:
    history = _interrupted_cycle(cycles=10)
    signal = LoopDetector().evaluate(history)
    assert signal.confidence > 0.0
    assert "recurring" in signal.reason
    # Capped below stop_threshold: an uncalibrated new path never solo-stops.
    assert signal.confidence <= 0.85


def test_ngram_fallback_confidence_stable_as_cycles_grow() -> None:
    # Coverage approaches 1/3 from above and never dips below it: an arbitrary
    # floor slightly under 1/3 stays robust regardless of run length.
    short = LoopDetector().evaluate(_interrupted_cycle(cycles=4))
    long_ = LoopDetector().evaluate(_interrupted_cycle(cycles=40))
    assert short.confidence > 0.0
    assert long_.confidence > 0.0
    assert short.confidence == pytest.approx(long_.confidence, abs=0.05)


def test_ngram_never_boosts_an_existing_exact_loop() -> None:
    # Non-regression by construction: whenever LOOP-1 already reports a nonzero
    # confidence, LOOP-2 must never run at all, however permissive its own
    # thresholds are. Same history, maximally lax ngram config vs strict.
    history = _repeat([("read", {"p": "a"})], times=6)
    strict = LoopDetector().evaluate(history)
    permissive = LoopDetector(
        ngram_min_hits=2, ngram_min_coverage=0.01, ngram_max_confidence=1.0
    ).evaluate(history)
    assert strict.confidence == permissive.confidence
    assert strict.reason == permissive.reason


def test_ngram_reports_the_single_best_size_not_a_sum() -> None:
    # Anti-double-counting invariant: a dominant 2-action cycle (40 reps,
    # interrupted every cycle) coexists with an unrelated, much rarer 3-action
    # sub-repeat (3 reps) elsewhere in the same window. Both sizes clear
    # ngram_min_hits, but only the *single best* size (2) is ever reported --
    # the weaker 3-gram candidate must not be summed in or otherwise inflate
    # the score.
    history: list[ToolEvent] = []
    for i in range(40):
        history.append(ToolEvent(tool="a", args={}))
        history.append(ToolEvent(tool="b", args={}))
        history.append(ToolEvent(tool=f"x_{i}", args={"n": i}))
    for i in range(3):
        history.append(ToolEvent(tool="p", args={}))
        history.append(ToolEvent(tool="q", args={}))
        history.append(ToolEvent(tool="r", args={}))
        history.append(ToolEvent(tool=f"z_{i}", args={"n": i}))
    signal = LoopDetector().evaluate(history)
    assert "2-action" in signal.reason
    assert "40/" in signal.reason


def test_ngram_size_guard_does_not_wrongly_exclude_short_windows() -> None:
    # Regression (Copilot review): the feasibility guard used to be
    # `size > n // 2`, which wrongly excluded n-gram sizes that could still
    # reach the hits floor on a short window. The only guard actually needed
    # is against a size that would leave zero grams (size > n); the hard floor
    # (_NGRAM_HITS_FLOOR=2) already rules out any size whose gram count cannot
    # reach it.
    detector = LoopDetector(ngram_min_hits=2, ngram_min_coverage=0.5)
    tokens = ["a", "b", "a", "b", "a"]  # n=5; size=3 has gram_count=3, hits=2
    confidence, reason = detector._ngram_coverage_confidence(tokens)
    # hits=2 == ngram_min_hits=2 here -> sigmoid centered exactly at the ramp
    # midpoint (factor 0.5), so confidence is half the raw coverage (2/3).
    assert confidence == pytest.approx((2 / 3) * 0.5)
    assert "3-action" in reason


def test_ngram_min_hits_blocks_short_pattern() -> None:
    # Only 2 cycles: hits=2 sits one below the default ngram_min_hits=3 ramp
    # center. LOOP-3 replaced the old hard cut (exactly 0.0) with a sigmoid, so
    # this is now merely small, not silent -- the hard floor is 2 occurrences
    # (_NGRAM_HITS_FLOOR), not 3. Still comfortably below anything actionable.
    history = _interrupted_cycle(cycles=2)
    confidence = LoopDetector().evaluate(history).confidence
    assert 0.0 < confidence < 0.05


def test_ngram_min_coverage_floor_blocks_low_coverage() -> None:
    history = _interrupted_cycle(cycles=10)
    # Real coverage is ~1/3; raising the floor above that silences it.
    assert LoopDetector(ngram_min_coverage=0.5).evaluate(history).confidence == 0.0


def test_ngram_max_confidence_caps_the_score() -> None:
    history = _interrupted_cycle(cycles=10)
    signal = LoopDetector(ngram_max_confidence=0.05).evaluate(history)
    assert signal.confidence == pytest.approx(0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ngram_sizes": ()},
        {"ngram_sizes": (1, 2)},
        {"ngram_min_hits": 1},
        {"ngram_min_coverage": 0.0},
        {"ngram_min_coverage": 1.5},
        {"ngram_max_confidence": 1.5},
        {"ngram_max_confidence": -0.1},
        {"ngram_steepness": 0.0},
        {"ngram_steepness": -1.0},
    ],
)
def test_invalid_ngram_params_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LoopDetector(**kwargs)  # type: ignore[arg-type]


def test_healthy_multi_file_refactor_stays_silent() -> None:
    # Adversarial: same read/edit/test shape repeats, but every occurrence
    # targets a genuinely different file, so no n-gram ever recurs.
    history: list[ToolEvent] = []
    for i in range(10):
        history.append(ToolEvent(tool="read_file", args={"path": f"mod_{i}.py"}))
        history.append(ToolEvent(tool="edit_file", args={"path": f"mod_{i}.py", "patch": f"p{i}"}))
        history.append(ToolEvent(tool="run_tests", args={}))
    assert LoopDetector().evaluate(history).confidence == 0.0


def test_healthy_repo_exploration_stays_silent() -> None:
    # Adversarial: alternating search/read, distinct query and file every time.
    history: list[ToolEvent] = []
    for i in range(10):
        history.append(ToolEvent(tool="rg", args={"pattern": f"query_{i}"}))
        history.append(ToolEvent(tool="read_file", args={"path": f"file_{i}.py"}))
    assert LoopDetector().evaluate(history).confidence == 0.0


def test_fixture_interrupted_cycle_fires_below_stop_via_ngram() -> None:
    # Lives under tests/fixtures/ (not traces/loop/) on purpose, like RET-2's
    # own storm fixture: the n-gram fallback is capped below stop_threshold, so
    # this trace cannot serve as a must-be-caught golden loop trace under
    # tests/test_replay.py::test_golden_loops_are_all_caught.
    trace = load_trace(_FIXTURES_DIR / "loop_ngram_interrupted_cycle.json")
    signal = LoopDetector().evaluate(trace.events)
    assert 0.0 < signal.confidence <= 0.85
    assert "recurring" in signal.reason
    # RET-2's niche is a different shape (single tool, uninterrupted run,
    # requires failures) -- it stays silent here, proving non-redundancy.
    assert RetryDetector().evaluate(trace.events).confidence == 0.0
    # All calls succeed with no test/exit-code evidence: NoProgressDetector
    # has nothing to anchor on either -- LOOP-2 is filling a real, distinct gap.
    assert NoProgressDetector(coding_anchors()).evaluate(trace.events).confidence == 0.0


def test_fixture_multi_file_refactor_is_zero(trace: Callable[[str], list[ToolEvent]]) -> None:
    history = trace("healthy/multi_file_refactor.json")
    assert LoopDetector().evaluate(history).confidence == 0.0


def test_fixture_repo_exploration_is_zero(trace: Callable[[str], list[ToolEvent]]) -> None:
    history = trace("healthy/repo_exploration.json")
    assert LoopDetector().evaluate(history).confidence == 0.0


def test_healthy_repeated_checkpoint_pair_never_solo_stops(
    trace: Callable[[str], list[ToolEvent]],
) -> None:
    # Known, documented limitation (see traces/README.md): a fixed routine
    # checkpoint pair (lint, run_tests) recurring after genuinely distinct real
    # work is structurally identical, from a pure repetition-frequency point of
    # view, to the proven LOOP-2 recall-gap shape with the roles reversed.
    # LoopDetector alone cannot tell "idle busywork interrupted by a one-off
    # check" apart from "real work followed by routine QA gates" -- that
    # distinction is NoProgressDetector's job. So the raw confidence here is
    # deliberately *not* zero, but it stays capped below stop_threshold and the
    # increasing `passed` count feeds the loop<-progress suppression, so the
    # product-level guarantee that actually matters -- the default pipeline
    # never emits a false STOP -- holds. This test pins that guarantee.
    history = trace("healthy/repeated_checkpoint_pair.json")
    signal = LoopDetector().evaluate(history)
    assert 0.0 < signal.confidence <= 0.85

    pipeline = build_pipeline()
    actions = {pipeline.process(event) for event in history}
    assert actions == {Action.NONE}
    assert pipeline.stopped is False


@pytest.mark.parametrize(
    "filename",
    [
        "loop/single_action_repeat.json",
        "loop/period3_cycle.json",
        "loop/volatile_ids.json",
        "loop/ping_pong.json",
        "loop/retry_same_error.json",
    ],
)
def test_existing_golden_loops_are_unaffected_by_ngram_fallback(
    filename: str, trace: Callable[[str], list[ToolEvent]]
) -> None:
    # Non-regression: every pre-existing golden `loop/` trace was already
    # caught by LOOP-1 alone. Adding LOOP-2 must not change their confidence
    # or reason at all -- the fallback never even runs for them (see
    # test_ngram_never_boosts_an_existing_exact_loop for the general proof).
    history = trace(filename)
    signal = LoopDetector().evaluate(history)
    assert signal.confidence > 0.0
    assert "repeat of" in signal.reason  # LOOP-1's reason shape, not LOOP-2's "recurring"


# --------------------------------------------------------------------------- #
# LOOP-3: windowed (continuous) scoring for the LOOP-2 fallback
# --------------------------------------------------------------------------- #


def test_ngram_confidence_rises_smoothly_with_hits() -> None:
    # The DoD for LOOP-3: confidence must rise smoothly as repetition increases,
    # not jump from 0.0 straight to the raw coverage value at a hard cutoff.
    # Build interrupted cycles with a growing number of repetitions. Raw
    # coverage itself is not perfectly monotonic (it approaches its asymptote
    # from above, see test_ngram_fallback_confidence_stable_as_cycles_grow),
    # so the assertion is on the shape of the curve -- no discontinuous jump,
    # small at the start, saturated by the end -- not strict monotonicity.
    confidences = [
        LoopDetector().evaluate(_interrupted_cycle(cycles=c)).confidence for c in range(2, 9)
    ]
    # No single step should jump by more than ~0.3 (a hard 0 -> raw-coverage
    # cutoff would jump by roughly the full coverage value, e.g. ~0.34).
    steps = [abs(b - a) for a, b in itertools.pairwise(confidences)]
    assert all(step < 0.3 for step in steps)
    # First point (cycles=2, hits=2) must be small but the ramp must have
    # visibly grown by cycles=8 (hits=8, well past the ngram_min_hits=3 center).
    assert 0.0 < confidences[0] < 0.05
    assert confidences[-1] > 0.3


def test_ngram_sigmoid_never_overflows_on_extreme_config() -> None:
    # Regression: _sigmoid used to clamp `x` before multiplying by `steepness`,
    # so a high ngram_min_hits combined with a high ngram_steepness could still
    # push the exponent past math.exp's range and raise OverflowError -- a
    # detector must never crash the host agent.
    detector = LoopDetector(ngram_min_hits=60, ngram_steepness=20.0, window=1000)
    signal = detector.evaluate(_interrupted_cycle(cycles=2))
    assert 0.0 <= signal.confidence <= 1.0
    # Symmetric case: hits far *above* an extreme center should saturate to
    # the coverage value (times ~1.0), not raise either.
    saturating = LoopDetector(ngram_min_hits=2, ngram_steepness=20.0)
    signal2 = saturating.evaluate(_interrupted_cycle(cycles=30))
    assert 0.0 <= signal2.confidence <= 1.0


def test_ngram_steepness_controls_ramp_sharpness() -> None:
    # Higher steepness -> sharper transition -> smaller leak just below the
    # ngram_min_hits center (protects adversarial fixtures like
    # retry_then_succeed.json more aggressively, at the cost of a less smooth
    # DoD curve). Verifies the parameter actually does something and in the
    # expected direction.
    history = _interrupted_cycle(cycles=2)  # hits=2, one below the default center=3
    soft = LoopDetector(ngram_steepness=1.0).evaluate(history).confidence
    sharp = LoopDetector(ngram_steepness=8.0).evaluate(history).confidence
    assert 0.0 < sharp < soft


def test_healthy_retry_then_succeed_stays_negligible(
    trace: Callable[[str], list[ToolEvent]],
) -> None:
    # Regression pinned from the LOOP-3 spike: this VAL-2
    # false-positive-boundary fixture (a transient failure retried twice then
    # resolved) has hits=2 for its best n-gram -- exactly the case the LOOP-3
    # ramp must keep negligible. A naive fix that simply lowered/removed
    # ngram_min_hits regressed this trace from 0.0 straight to 0.333; the
    # sigmoid ramp keeps it near-zero instead.
    history = trace("healthy/retry_then_succeed.json")
    signal = LoopDetector().evaluate(history)
    assert 0.0 <= signal.confidence < 0.05

    pipeline = build_pipeline()
    actions = {pipeline.process(event) for event in history}
    assert actions == {Action.NONE}
    assert pipeline.stopped is False
