"""Tests for arbitration: weighted noisy-OR fusion + single stop threshold.

Both directions matter: a strong (or converging) signal must stop the run, and
anything below the threshold must stay inert. ARB-2 shipped the single
``stop_threshold``; ARB-1 replaces the raw ``max`` with a weighted noisy-OR so
distinct detectors reinforce each other while a single maxed detector (default
weight 1.0) still stops. The threshold and the per-detector weights are
configurable.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    TestProgressAnchor,
    is_advancing,
    loop_suppression_anchors,
)
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, HysteresisState, OverrideRule, Policy, SuppressionRule
from pathlayer.policies.base import PolicyConfig


def test_no_signals_is_none() -> None:
    assert Policy().decide([]) is Action.NONE


def test_below_threshold_is_none() -> None:
    assert Policy().decide([Signal("loop", confidence=0.5)]) is Action.NONE


def test_at_threshold_stops() -> None:
    assert Policy().decide([Signal("loop", confidence=0.9)]) is Action.STOP


def test_above_threshold_stops() -> None:
    assert Policy().decide([Signal("loop", confidence=0.99)]) is Action.STOP


def test_strongest_single_signal_still_stops() -> None:
    # One detector above threshold stops, even amid weaker ones (default weights
    # 1.0): noisy-OR(0.1, 0.95, 0.0) = 0.955 >= 0.9. No regression vs raw max.
    signals = [
        Signal("retry", confidence=0.1),
        Signal("loop", confidence=0.95),
        Signal("budget", confidence=0.0),
    ]
    assert Policy().decide(signals) is Action.STOP


def test_threshold_is_configurable() -> None:
    lenient = Policy(config=PolicyConfig(stop_threshold=0.99))
    assert lenient.decide([Signal("loop", confidence=0.95)]) is Action.NONE

    strict = Policy(config=PolicyConfig(stop_threshold=0.5))
    assert strict.decide([Signal("loop", confidence=0.6)]) is Action.STOP


def test_default_stop_threshold() -> None:
    assert PolicyConfig().stop_threshold == 0.9


def test_boundary_thresholds_are_allowed() -> None:
    assert PolicyConfig(stop_threshold=0.0).stop_threshold == 0.0
    assert PolicyConfig(stop_threshold=1.0).stop_threshold == 1.0


def test_out_of_range_threshold_raises() -> None:
    with pytest.raises(ValueError, match="stop_threshold"):
        PolicyConfig(stop_threshold=1.5)
    with pytest.raises(ValueError, match="stop_threshold"):
        PolicyConfig(stop_threshold=-0.1)


# --------------------------------------------------------------------------- #
# ARB-1: weighted noisy-OR fusion
# --------------------------------------------------------------------------- #


def test_fuse_empty_is_zero() -> None:
    assert Policy().fuse([]) == 0.0


def test_fuse_single_signal_is_identity() -> None:
    # With weight 1.0, noisy-OR of one signal returns that confidence unchanged.
    assert Policy().fuse([Signal("loop", confidence=0.6)]) == pytest.approx(0.6)
    assert Policy().fuse([Signal("loop", confidence=0.3)]) == pytest.approx(0.3)


def test_fuse_is_bounded_by_one() -> None:
    # Two maxed signals must not overflow [0, 1].
    fused = Policy().fuse([Signal("loop", confidence=1.0), Signal("retry", confidence=1.0)])
    assert fused == pytest.approx(1.0)


def test_fuse_clamps_confidence_above_one() -> None:
    # A custom detector emitting confidence > 1 must not push fused past 1.0
    # (audit #104 M-1): clamp to 1.0, do not raise.
    assert Policy().fuse([Signal("custom", confidence=2.0)]) == pytest.approx(1.0)


def test_fuse_clamps_confidence_below_zero() -> None:
    # A negative confidence would make (1 - w*c) > 1 and invert the product,
    # yielding a negative fused score. Clamp to 0.0 so the signal is inert.
    assert Policy().fuse([Signal("custom", confidence=-1.0)]) == pytest.approx(0.0)


def test_fuse_mixed_out_of_range_stays_bounded() -> None:
    # One out-of-range signal alongside a normal one stays in [0, 1] and the
    # clamped-high signal behaves like a maxed (1.0) detector.
    fused = Policy().fuse([Signal("custom", confidence=1.5), Signal("loop", confidence=0.5)])
    assert 0.0 <= fused <= 1.0
    # 1.5 clamps to 1.0 -> noisy-OR with 0.5 saturates to 1.0.
    assert fused == pytest.approx(1.0)


def test_fuse_treats_nan_confidence_as_inert() -> None:
    # NaN fails every comparison, so a plain clamp would let it propagate and
    # make fused NaN (then `fused >= stop_threshold` silently reads False).
    # Map NaN to 0.0 so the signal is inert and the invariant holds.
    nan = float("nan")
    assert Policy().fuse([Signal("custom", confidence=nan)]) == pytest.approx(0.0)
    # A NaN signal must not swallow a real one riding alongside it.
    fused = Policy().fuse([Signal("custom", confidence=nan), Signal("loop", confidence=0.5)])
    assert fused == pytest.approx(0.5)


def test_convergence_stops_below_each_individual_threshold() -> None:
    # DoD: single-detector fire scores lower than multi-detector convergence.
    # Two signals at 0.7 -> 1 - 0.3*0.3 = 0.91 >= 0.9 -> STOP, yet each alone
    # (0.7 < 0.9) stays NONE.
    policy = Policy()
    loop = Signal("loop", confidence=0.7)
    retry = Signal("retry", confidence=0.7)
    assert policy.fuse([loop, retry]) == pytest.approx(0.91)
    assert policy.decide([loop, retry]) is Action.STOP
    assert policy.decide([loop]) is Action.NONE
    assert policy.decide([retry]) is Action.NONE


def test_three_detector_convergence_is_the_prod_set() -> None:
    # The production set (loop + retry + no_progress) converging: three at 0.55
    # -> 1 - 0.45^3 ~= 0.9089 >= 0.9 -> STOP, each alone NONE.
    policy = Policy()
    signals = [
        Signal("loop", confidence=0.55),
        Signal("retry", confidence=0.55),
        Signal("no_progress", confidence=0.55),
    ]
    assert policy.fuse(signals) == pytest.approx(1 - 0.45**3)
    assert policy.decide(signals) is Action.STOP
    for signal in signals:
        assert policy.decide([signal]) is Action.NONE


def test_weight_attenuates_contribution() -> None:
    # A weight scales a detector's pull on the fused score.
    policy = Policy(PolicyConfig(weights={"loop": 0.5}))
    assert policy.fuse([Signal("loop", confidence=0.8)]) == pytest.approx(0.4)


def test_weight_zero_neutralizes_detector() -> None:
    # weight 0 makes a detector inert: a maxed no_progress no longer stops.
    weighted = Policy(PolicyConfig(weights={"no_progress": 0.0}))
    signals = [Signal("loop", confidence=0.5), Signal("no_progress", confidence=1.0)]
    assert weighted.fuse(signals) == pytest.approx(0.5)
    assert weighted.decide(signals) is Action.NONE
    # Without the weight, the maxed no_progress alone stops.
    assert Policy().decide(signals) is Action.STOP


def test_weight_zero_matches_advisory_decision() -> None:
    # weight 0 and advisory exclusion both make a lone signal unable to STOP.
    signal = [Signal("loop", confidence=1.0)]
    by_weight = Policy(PolicyConfig(weights={"loop": 0.0}))
    by_advisory = Policy(PolicyConfig(advisory_signals=frozenset({"loop"})))
    assert by_weight.decide(signal) is Action.NONE
    assert by_advisory.decide(signal) is Action.NONE


def test_unknown_detector_name_defaults_to_full_weight() -> None:
    # fuse stays permissive: a name with no explicit weight gets 1.0, so a
    # custom/unconfigured detector still has full authority.
    policy = Policy(PolicyConfig(weights={"loop": 0.5}))
    assert policy.config.weight_for("retry") == 1.0
    assert policy.decide([Signal("retry", confidence=1.0)]) is Action.STOP


def test_advisory_excluded_before_fusion() -> None:
    policy = Policy(PolicyConfig(advisory_signals=frozenset({"no_progress"})))
    signals = [Signal("loop", confidence=0.5), Signal("no_progress", confidence=1.0)]
    assert policy.fuse(signals) == pytest.approx(0.5)
    assert policy.decide(signals) is Action.NONE


def test_out_of_range_weight_raises() -> None:
    with pytest.raises(ValueError, match="weight for 'loop'"):
        PolicyConfig(weights={"loop": 1.5})
    with pytest.raises(ValueError, match="weight for 'retry'"):
        PolicyConfig(weights={"retry": -0.1})


def test_weight_int_coerced_to_float() -> None:
    config = PolicyConfig(weights={"loop": 1})
    assert config.weight_for("loop") == 1.0
    assert isinstance(config.weight_for("loop"), float)


def test_weight_bool_rejected() -> None:
    # bool is an int subclass: True must not silently become weight 1.0.
    with pytest.raises(ValueError, match="weight for 'loop' must be a number"):
        PolicyConfig(weights={"loop": True})  # type: ignore[dict-item]


def test_weight_non_number_rejected() -> None:
    with pytest.raises(ValueError, match="weight for 'loop' must be a number"):
        PolicyConfig(weights={"loop": "0.5"})  # type: ignore[dict-item]


def test_default_weights_empty() -> None:
    assert PolicyConfig().weights == {}


def test_weights_are_defensively_copied() -> None:
    source = {"loop": 0.5}
    config = PolicyConfig(weights=source)
    source["loop"] = 0.0
    assert config.weight_for("loop") == 0.5


def test_dominant_signal_uses_weighted_contribution() -> None:
    # A loud-but-down-weighted signal must not outrank a fully-weighted one.
    policy = Policy(PolicyConfig(weights={"loop": 0.1}))
    loud = Signal("loop", confidence=0.9)  # contribution 0.09
    real = Signal("retry", confidence=0.5)  # contribution 0.50
    assert policy.dominant_signal([loud, real]) is real


def test_dominant_signal_none_without_actionable() -> None:
    policy = Policy(PolicyConfig(advisory_signals=frozenset({"loop"})))
    assert policy.dominant_signal([Signal("loop", confidence=1.0)]) is None
    assert policy.dominant_signal([]) is None


# --------------------------------------------------------------------------- #
# ARB-3: graded ladder (target_tier) + hysteresis (step)
# --------------------------------------------------------------------------- #

# A full intermediate ladder, each tier strictly below stop_threshold (0.9).
_LADDER: dict[Action, float] = {
    Action.WARN: 0.3,
    Action.SUMMARIZE: 0.5,
    Action.FALLBACK: 0.7,
    Action.ESCALATE: 0.8,
}


def _sig(confidence: float) -> list[Signal]:
    # One detector at full weight: fused confidence == this confidence.
    return [Signal("loop", confidence=confidence)]


def _run(policy: Policy, confidences: Sequence[float]) -> list[Action]:
    """Thread the hysteresis state across a sequence of single-signal ticks."""
    state = HysteresisState()
    actions: list[Action] = []
    for confidence in confidences:
        state, action = policy.step(state, _sig(confidence))
        actions.append(action)
    return actions


def test_target_tier_default_is_none_or_stop() -> None:
    # Zero-config: only STOP is active, so the ladder degenerates to ARB-1.
    policy = Policy()
    assert policy.target_tier(_sig(0.5)) is Action.NONE
    assert policy.target_tier(_sig(0.9)) is Action.STOP
    assert policy.target_tier(_sig(1.0)) is Action.STOP


def test_target_tier_maps_each_ladder_band() -> None:
    policy = Policy(PolicyConfig(tier_thresholds=_LADDER))
    assert policy.target_tier(_sig(0.1)) is Action.NONE
    assert policy.target_tier(_sig(0.3)) is Action.WARN
    assert policy.target_tier(_sig(0.5)) is Action.SUMMARIZE
    assert policy.target_tier(_sig(0.7)) is Action.FALLBACK
    assert policy.target_tier(_sig(0.85)) is Action.ESCALATE
    assert policy.target_tier(_sig(0.95)) is Action.STOP


def test_target_tier_none_without_actionable_signal() -> None:
    # Advisory exclusion can empty the actionable set: you cannot act on nothing.
    policy = Policy(PolicyConfig(tier_thresholds=_LADDER, advisory_signals=frozenset({"loop"})))
    assert policy.target_tier(_sig(1.0)) is Action.NONE
    assert policy.target_tier([]) is Action.NONE


def test_active_tiers_derives_stop_from_stop_threshold() -> None:
    policy = Policy(PolicyConfig(stop_threshold=0.95, tier_thresholds={Action.WARN: 0.3}))
    assert policy.config.active_tiers() == {Action.WARN: 0.3, Action.STOP: 0.95}


def test_default_ticks_follow_target_immediately() -> None:
    # escalate_ticks == deescalate_ticks == 1: no hysteresis, follow the target.
    policy = Policy(PolicyConfig(tier_thresholds=_LADDER))
    assert _run(policy, [0.0, 0.35, 0.0]) == [Action.NONE, Action.WARN, Action.NONE]


def test_default_policy_step_matches_arb1() -> None:
    # Regression: with the default policy, step is exactly decide (NONE/STOP).
    policy = Policy()
    assert _run(policy, [0.5, 0.9, 0.99]) == [Action.NONE, Action.STOP, Action.STOP]


def test_brief_spike_does_not_escalate() -> None:
    # A single tick above threshold then a drop must not flip the action up.
    policy = Policy(PolicyConfig(escalate_ticks=2))
    assert _run(policy, [0.0, 0.95, 0.0]) == [Action.NONE, Action.NONE, Action.NONE]


def test_sustained_breach_escalates() -> None:
    policy = Policy(PolicyConfig(escalate_ticks=2))
    assert _run(policy, [0.95, 0.95]) == [Action.NONE, Action.STOP]


def test_escalation_jumps_straight_to_target_tier() -> None:
    # H-1: a confidence that lands at the STOP band commits STOP after exactly
    # escalate_ticks ticks -- it does NOT crawl WARN->...->STOP one tier per N.
    policy = Policy(PolicyConfig(tier_thresholds=_LADDER, escalate_ticks=2))
    assert _run(policy, [0.95, 0.95]) == [Action.NONE, Action.STOP]


def test_rising_signal_commits_target_on_trip_tick() -> None:
    # Pins the documented semantics: the counter tracks pressure away from the
    # committed tier, not time at the destination. With escalate_ticks=2, a
    # WARN-level tick followed by a STOP-level tick commits STOP on the second
    # tick -- even though only one tick was at STOP level. A refactor toward
    # per-destination-tier counting would break this on purpose.
    policy = Policy(PolicyConfig(tier_thresholds=_LADDER, escalate_ticks=2))
    assert _run(policy, [0.35, 0.95]) == [Action.NONE, Action.STOP]


def test_oscillation_never_escalates() -> None:
    # Alternating above/below threshold keeps resetting the counter (anti-flap).
    policy = Policy(PolicyConfig(escalate_ticks=2))
    assert _run(policy, [0.95, 0.0, 0.95, 0.0]) == [Action.NONE] * 4


def test_smooth_deescalation_after_m_ticks() -> None:
    policy = Policy(PolicyConfig(tier_thresholds=_LADDER, deescalate_ticks=2))
    # Up to ESCALATE immediately (escalate_ticks=1), then two low ticks to drop.
    assert _run(policy, [0.85, 0.0, 0.0]) == [Action.ESCALATE, Action.ESCALATE, Action.NONE]


def test_deescalation_counter_resets_on_non_confirming_tick() -> None:
    # A tick back at the committed tier mid-descent resets the down counter, so
    # de-escalation needs deescalate_ticks *consecutive* low ticks.
    policy = Policy(PolicyConfig(tier_thresholds=_LADDER, deescalate_ticks=2))
    actions = _run(policy, [0.85, 0.0, 0.85, 0.0, 0.0])
    assert actions == [
        Action.ESCALATE,
        Action.ESCALATE,  # down_count 1
        Action.ESCALATE,  # back at ESCALATE -> reset
        Action.ESCALATE,  # down_count 1 again
        Action.NONE,  # down_count 2 -> drop
    ]


def test_asymmetric_escalate_and_deescalate() -> None:
    # Slow to escalate (3), fast to de-escalate (1).
    policy = Policy(PolicyConfig(tier_thresholds=_LADDER, escalate_ticks=3, deescalate_ticks=1))
    assert _run(policy, [0.35, 0.35, 0.35, 0.0]) == [
        Action.NONE,
        Action.NONE,
        Action.WARN,
        Action.NONE,
    ]


# --- PolicyConfig validation (ARB-3) --------------------------------------- #


def test_default_tier_thresholds_empty_and_ticks_one() -> None:
    config = PolicyConfig()
    assert config.tier_thresholds == {}
    assert config.escalate_ticks == 1
    assert config.deescalate_ticks == 1


def test_tier_thresholds_reject_stop_and_none_keys() -> None:
    with pytest.raises(ValueError, match="intermediate tiers"):
        PolicyConfig(tier_thresholds={Action.STOP: 0.5})
    with pytest.raises(ValueError, match="intermediate tiers"):
        PolicyConfig(tier_thresholds={Action.NONE: 0.5})


def test_tier_threshold_must_be_below_stop_threshold() -> None:
    with pytest.raises(ValueError, match="must be < stop_threshold"):
        PolicyConfig(stop_threshold=0.9, tier_thresholds={Action.ESCALATE: 0.95})


def test_tier_thresholds_must_be_non_decreasing() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        PolicyConfig(tier_thresholds={Action.WARN: 0.8, Action.SUMMARIZE: 0.5})


def test_tier_thresholds_with_gaps_are_allowed() -> None:
    # A sparse ladder (only WARN and ESCALATE) is valid and ordered.
    config = PolicyConfig(tier_thresholds={Action.WARN: 0.3, Action.ESCALATE: 0.8})
    assert config.tier_thresholds == {Action.WARN: 0.3, Action.ESCALATE: 0.8}


def test_tier_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        PolicyConfig(tier_thresholds={Action.WARN: 1.5})


def test_tier_threshold_bool_rejected() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        PolicyConfig(tier_thresholds={Action.WARN: True})  # type: ignore[dict-item]


def test_tier_threshold_non_number_rejected() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        PolicyConfig(tier_thresholds={Action.WARN: "0.3"})  # type: ignore[dict-item]


def test_tier_thresholds_defensively_copied() -> None:
    source = {Action.WARN: 0.3}
    config = PolicyConfig(tier_thresholds=source)
    source[Action.WARN] = 0.0
    assert config.tier_thresholds == {Action.WARN: 0.3}


def test_tier_threshold_int_coerced_to_float() -> None:
    config = PolicyConfig(stop_threshold=1.0, tier_thresholds={Action.WARN: 0})
    assert config.tier_thresholds[Action.WARN] == 0.0
    assert isinstance(config.tier_thresholds[Action.WARN], float)


def test_escalate_ticks_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="escalate_ticks must be an int >= 1"):
        PolicyConfig(escalate_ticks=0)


def test_deescalate_ticks_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="deescalate_ticks must be an int >= 1"):
        PolicyConfig(deescalate_ticks=0)


def test_ticks_bool_rejected() -> None:
    # bool is an int subclass: True must not silently become 1 tick.
    with pytest.raises(ValueError, match="escalate_ticks must be an int >= 1"):
        PolicyConfig(escalate_ticks=True)  # type: ignore[arg-type]


def test_ticks_non_int_rejected() -> None:
    with pytest.raises(ValueError, match="deescalate_ticks must be an int >= 1"):
        PolicyConfig(deescalate_ticks=1.5)  # type: ignore[arg-type]


# --- Pipeline integration: hysteresis delays a real STOP ------------------- #


class _ScriptedDetector(Detector):
    """Emits a pre-set confidence per event, to drive arbitration deterministically."""

    name = "loop"

    def __init__(self, confidences: Sequence[float]) -> None:
        self._confidences = list(confidences)

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        return Signal(self.name, confidence=self._confidences[len(history) - 1])


def test_pipeline_hysteresis_delays_stop_and_resets_on_dip() -> None:
    # escalate_ticks=2: a lone spike (idx 0) then a dip (idx 1) must not STOP;
    # only the sustained breach at idx 2-3 does. Without hysteresis the run would
    # have stopped at idx 0.
    detector = _ScriptedDetector([0.95, 0.2, 0.95, 0.95])
    pipeline = Pipeline([detector], Policy(PolicyConfig(escalate_ticks=2)))
    events = [ToolEvent(tool="t", args={}, ts=float(i)) for i in range(4)]
    actions = pipeline.run(events)

    assert actions == [Action.NONE, Action.NONE, Action.NONE, Action.STOP]
    assert pipeline.stopped is True
    assert pipeline.post_mortem is not None
    assert pipeline.post_mortem.stopped_at_index == 3


class _RaisingDetector(Detector):
    """Builds confidence, then raises -- to test rollback of the hysteresis state."""

    name = "loop"

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        if len(history) >= 2:
            raise RuntimeError("boom")
        return Signal(self.name, confidence=0.95)


def test_pipeline_rolls_back_hysteresis_state_on_detector_error() -> None:
    # A tick that raises must leave history AND the hysteresis counters untouched,
    # so the run stays deterministic and a partial tick cannot be half-committed.
    pipeline = Pipeline([_RaisingDetector()], Policy(PolicyConfig(escalate_ticks=3)))
    assert pipeline.process(ToolEvent(tool="t", ts=0.0)) is Action.NONE
    before = pipeline._hysteresis
    assert before.up_count == 1

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.process(ToolEvent(tool="t", ts=1.0))

    # History rolled back to the one good event, hysteresis unchanged.
    assert len(pipeline.history) == 1
    assert pipeline._hysteresis == before


# --------------------------------------------------------------------------- #
# ARB-4: priority overrides (bypass fusion + hysteresis)
# --------------------------------------------------------------------------- #

_BUDGET_STOP: dict[str, OverrideRule] = {"budget": OverrideRule(threshold=1.0, action=Action.STOP)}


def test_default_overrides_empty() -> None:
    assert PolicyConfig().overrides == {}


def test_override_action_none_without_rules() -> None:
    # Zero-config: no override machinery engaged.
    assert Policy().override_action([Signal("budget", confidence=1.0)]) is None


def test_override_fires_at_threshold() -> None:
    policy = Policy(PolicyConfig(overrides=_BUDGET_STOP))
    assert policy.override_action([Signal("budget", confidence=1.0)]) is Action.STOP


def test_override_inert_below_threshold() -> None:
    policy = Policy(PolicyConfig(overrides=_BUDGET_STOP))
    assert policy.override_action([Signal("budget", confidence=0.99)]) is None


def test_override_bypasses_stop_threshold() -> None:
    # A budget rule at 0.5 stops well below the 0.9 stop_threshold, and even
    # though the fused confidence alone (0.5) would not.
    policy = Policy(PolicyConfig(overrides={"budget": OverrideRule(0.5, Action.STOP)}))
    signals = [Signal("budget", confidence=0.5)]
    assert policy.fuse(signals) == pytest.approx(0.5)
    assert policy.decide(signals) is Action.STOP


def test_override_decide_matches_step() -> None:
    policy = Policy(PolicyConfig(overrides=_BUDGET_STOP))
    signals = [Signal("budget", confidence=1.0)]
    _, action = policy.step(HysteresisState(), signals)
    assert action is Action.STOP
    assert policy.decide(signals) is Action.STOP


def test_override_bypasses_hysteresis() -> None:
    # escalate_ticks=5 would delay a normal STOP, but the override commits on the
    # first tick. A non-override signal at the same confidence would wait.
    policy = Policy(PolicyConfig(escalate_ticks=5, overrides=_BUDGET_STOP))
    state, action = policy.step(HysteresisState(), [Signal("budget", confidence=1.0)])
    assert action is Action.STOP
    assert state.committed is Action.STOP
    # Contrast: a maxed loop (no override) stays inert under escalate_ticks=5.
    _, loop_action = policy.step(HysteresisState(), [Signal("loop", confidence=1.0)])
    assert loop_action is Action.NONE


def test_override_advisory_signal_stays_inert() -> None:
    # finding A: advisory wins -- an observe-only signal never fires its override.
    policy = Policy(PolicyConfig(advisory_signals=frozenset({"budget"}), overrides=_BUDGET_STOP))
    signals = [Signal("budget", confidence=1.0)]
    assert policy.override_action(signals) is None
    assert policy.step(HysteresisState(), signals)[1] is Action.NONE


def test_override_most_severe_wins() -> None:
    policy = Policy(
        PolicyConfig(
            tier_thresholds=_LADDER,
            overrides={
                "loop": OverrideRule(0.5, Action.WARN),
                "budget": OverrideRule(0.5, Action.STOP),
            },
        )
    )
    signals = [Signal("loop", confidence=0.6), Signal("budget", confidence=0.6)]
    assert policy.override_action(signals) is Action.STOP


def test_override_intermediate_tier_then_deescalates() -> None:
    # An override to a non-STOP tier commits at once (the fused path alone would
    # not act here), then normal de-escalation resumes once the rule stops firing.
    policy = Policy(PolicyConfig(overrides={"budget": OverrideRule(0.3, Action.ESCALATE)}))
    state = HysteresisState()
    state, a0 = policy.step(state, [Signal("budget", confidence=0.35)])
    assert a0 is Action.ESCALATE and state.committed is Action.ESCALATE
    # Rule no longer fires (low budget) and no other signal: de-escalate to NONE.
    state, a1 = policy.step(state, [Signal("budget", confidence=0.0)])
    assert a1 is Action.NONE


def test_override_never_downgrades_fused_stop() -> None:
    # An intermediate-tier override must not mask a more severe fused STOP:
    # step/decide commit max(override, fused path). loop@0.6 + budget@1.0 fuse to
    # 1.0 (STOP) while the loop rule only asks for WARN.
    policy = Policy(PolicyConfig(overrides={"loop": OverrideRule(0.5, Action.WARN)}))
    signals = [Signal("loop", confidence=0.6), Signal("budget", confidence=1.0)]
    assert policy.override_action(signals) is Action.WARN
    assert policy.fuse(signals) == pytest.approx(1.0)
    assert policy.step(HysteresisState(), signals)[1] is Action.STOP
    assert policy.decide(signals) is Action.STOP


def test_override_escalates_above_fused_path() -> None:
    # When the fused path would be inert, a fired override raises the action
    # immediately (bypassing hysteresis), which is the whole point.
    policy = Policy(
        PolicyConfig(escalate_ticks=9, overrides={"budget": OverrideRule(0.3, Action.ESCALATE)})
    )
    state, action = policy.step(HysteresisState(), [Signal("budget", confidence=0.4)])
    assert action is Action.ESCALATE
    assert state.committed is Action.ESCALATE


def test_override_signal_names_trigger_despite_zero_weight() -> None:
    # finding B: budget carries weight 0 (inert in fusion) but drives the STOP;
    # override_signal must still name budget, not the fusion-dominant loop.
    policy = Policy(
        PolicyConfig(weights={"budget": 0.0}, overrides={"budget": OverrideRule(0.5, Action.STOP)})
    )
    loop = Signal("loop", confidence=0.3)  # fusion-dominant (weight 1.0)
    budget = Signal("budget", confidence=1.0)  # override trigger, weight 0
    assert policy.dominant_signal([loop, budget]) is loop
    assert policy.override_signal([loop, budget]) is budget


def test_override_signal_none_without_trigger() -> None:
    policy = Policy(PolicyConfig(overrides=_BUDGET_STOP))
    assert policy.override_signal([Signal("budget", confidence=0.1)]) is None


# --- PolicyConfig override validation --------------------------------------- #


def test_override_rule_type_enforced() -> None:
    with pytest.raises(ValueError, match="must be an OverrideRule"):
        PolicyConfig(overrides={"budget": 1.0})  # type: ignore[dict-item]


def test_override_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="override threshold for 'budget' must be in"):
        PolicyConfig(overrides={"budget": OverrideRule(1.5, Action.STOP)})
    with pytest.raises(ValueError, match="override threshold for 'budget' must be in"):
        PolicyConfig(overrides={"budget": OverrideRule(-0.1, Action.STOP)})


def test_override_threshold_bool_rejected() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        PolicyConfig(overrides={"budget": OverrideRule(True, Action.STOP)})  # type: ignore[arg-type]


def test_override_action_none_rejected() -> None:
    with pytest.raises(ValueError, match="must be an actionable tier"):
        PolicyConfig(overrides={"budget": OverrideRule(1.0, Action.NONE)})


def test_override_threshold_int_coerced_to_float() -> None:
    config = PolicyConfig(overrides={"budget": OverrideRule(1, Action.STOP)})
    rule = config.overrides["budget"]
    assert rule.threshold == 1.0
    assert isinstance(rule.threshold, float)


def test_overrides_defensively_copied() -> None:
    source = {"budget": OverrideRule(1.0, Action.STOP)}
    config = PolicyConfig(overrides=source)
    source["budget"] = OverrideRule(0.1, Action.WARN)
    assert config.overrides["budget"].action is Action.STOP


# --- Pipeline integration: override stops immediately under hysteresis ------ #


class _ScriptedBudget(Detector):
    """Emits a scripted budget confidence per event."""

    name = "budget"

    def __init__(self, confidences: Sequence[float]) -> None:
        self._confidences = list(confidences)

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        c = self._confidences[len(history) - 1]
        return Signal(self.name, confidence=c, reason=f"${c:.2f} of budget")


def test_pipeline_override_stops_immediately_despite_hysteresis() -> None:
    # escalate_ticks=3 would delay a normal STOP, but the budget ceiling override
    # halts on the tick it is reached, and the post-mortem names budget.
    detector = _ScriptedBudget([0.4, 1.0, 1.0])
    policy = Policy(PolicyConfig(escalate_ticks=3, overrides=_BUDGET_STOP))
    pipeline = Pipeline([detector], policy)
    events = [ToolEvent(tool="spend", args={}, ts=float(i)) for i in range(3)]
    actions = pipeline.run(events)

    assert actions == [Action.NONE, Action.STOP]  # stops at idx 1, run halts
    assert pipeline.stopped is True
    assert pipeline.post_mortem is not None
    assert pipeline.post_mortem.stopped_at_index == 1
    assert pipeline.post_mortem.reason.startswith("budget")


# --------------------------------------------------------------------------- #
# LOOP-2 precision: conditional suppression (SuppressionRule)
# --------------------------------------------------------------------------- #

_LOOP_SUPPRESSED_BY_PROGRESS: dict[str, SuppressionRule] = {
    "loop": SuppressionRule(evidence="progress", min_confidence=1.0)
}


def test_default_suppressions_empty() -> None:
    assert PolicyConfig().suppressions == {}


def test_no_evidence_is_pre_loop2_behavior() -> None:
    # Omitting `evidence` entirely (the default {}) must behave exactly like
    # before this feature: a suppression rule configured but never fed evidence
    # never suppresses anything.
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    signals = [Signal("loop", confidence=0.95)]
    assert policy.decide(signals) is Action.STOP
    assert policy.fuse(signals) == pytest.approx(0.95)
    assert policy.dominant_signal(signals) is signals[0]


def test_fresh_progress_suppresses_loop_from_stopping() -> None:
    # The F1a bug this rule fixes: loop alone would STOP, but fresh progress
    # evidence at/above min_confidence excludes it from the actionable set.
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    signals = [Signal("loop", confidence=0.95)]
    assert policy.decide(signals, {"progress": 1.0}) is Action.NONE
    assert policy.fuse(signals, {"progress": 1.0}) == pytest.approx(0.0)
    assert policy.dominant_signal(signals, {"progress": 1.0}) is None


def test_evidence_below_min_confidence_does_not_suppress() -> None:
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    signals = [Signal("loop", confidence=0.95)]
    assert policy.decide(signals, {"progress": 0.99}) is Action.STOP


def test_suppression_only_affects_the_guarded_signal() -> None:
    # A convergent second signal (retry) is untouched by loop's suppression --
    # it still contributes to fusion and can still drive STOP on its own.
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    signals = [Signal("loop", confidence=0.95), Signal("retry", confidence=0.95)]
    assert policy.fuse(signals, {"progress": 1.0}) == pytest.approx(0.95)
    assert policy.dominant_signal(signals, {"progress": 1.0}) is signals[1]
    assert policy.decide(signals, {"progress": 1.0}) is Action.STOP


def test_suppression_never_confused_with_advisory() -> None:
    # Suppression is conditional (on evidence); advisory is unconditional. A
    # signal guarded but not currently suppressed still behaves normally.
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    signals = [Signal("loop", confidence=0.95)]
    assert policy.decide(signals, {"progress": 0.0}) is Action.STOP
    assert policy.decide(signals) is Action.STOP


def test_suppression_excludes_from_override_and_dominant_signal() -> None:
    # A suppressed signal cannot trigger an override rule, nor be named the
    # dominant/override signal -- a post-mortem must never blame a detector
    # that was excluded from the decision (design note in policies/base.py).
    policy = Policy(
        PolicyConfig(
            suppressions=_LOOP_SUPPRESSED_BY_PROGRESS,
            overrides={"loop": OverrideRule(0.5, Action.STOP)},
        )
    )
    signals = [Signal("loop", confidence=0.95)]
    assert policy.override_action(signals, {"progress": 1.0}) is None
    assert policy.override_signal(signals, {"progress": 1.0}) is None
    assert policy.dominant_signal(signals, {"progress": 1.0}) is None
    # Without suppression, the same override fires normally.
    assert policy.override_action(signals) is Action.STOP


def test_suppression_excludes_from_target_tier_and_step() -> None:
    policy = Policy(
        PolicyConfig(tier_thresholds=_LADDER, suppressions=_LOOP_SUPPRESSED_BY_PROGRESS)
    )
    signals = _sig(0.95)
    assert policy.target_tier(signals, {"progress": 1.0}) is Action.NONE
    state, action = policy.step(HysteresisState(), signals, {"progress": 1.0})
    assert action is Action.NONE
    assert state.committed is Action.NONE


def test_suppression_lifted_once_evidence_goes_stale() -> None:
    # A loop that progressed once and now stagnates must become stoppable again
    # as soon as the fresh-evidence tick passes (mirrors is_advancing's design:
    # strict on the *last* step only).
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    signals = [Signal("loop", confidence=0.95)]
    assert policy.decide(signals, {"progress": 1.0}) is Action.NONE
    assert policy.decide(signals, {"progress": 0.0}) is Action.STOP


def test_suppression_alternating_with_hysteresis_never_commits_stop() -> None:
    # Pins the examples/pathlayer.config.yaml "Trou D" mitigation: escalate_ticks
    # above the longest gap between progress ticks absorbs flickering
    # suppression. Worst case here is progress on every *other* tick (the
    # longest gap escalate_ticks=2 can absorb): loop is maxed throughout, but a
    # fresh-evidence tick always resets up_count to 0 (target == committed ==
    # NONE) before a stale tick's up_count reaches 2, so the run never STOPs.
    policy = Policy(PolicyConfig(escalate_ticks=2, suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    signals = [Signal("loop", confidence=0.95)]
    state = HysteresisState()
    actions = []
    for progress in (1.0, 0.0, 1.0, 0.0, 1.0, 0.0):
        state, action = policy.step(state, signals, {"progress": progress})
        actions.append(action)
    assert actions == [Action.NONE] * 6
    assert state.committed is Action.NONE


def test_suppression_stale_for_escalate_ticks_consecutive_still_stops() -> None:
    # Hysteresis delays the commit, it does not defeat it: once fresh evidence
    # stops arriving for `escalate_ticks` *consecutive* ticks, the suppression's
    # protection lifts for good and loop STOPs, same as a real loop would.
    policy = Policy(PolicyConfig(escalate_ticks=2, suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    signals = [Signal("loop", confidence=0.95)]
    state = HysteresisState()
    state, action = policy.step(state, signals, {"progress": 1.0})
    assert action is Action.NONE
    state, action = policy.step(state, signals, {"progress": 0.0})
    assert action is Action.NONE  # 1st consecutive confirming tick, not committed yet
    state, action = policy.step(state, signals, {"progress": 0.0})
    assert action is Action.STOP  # 2nd consecutive confirming tick commits


# --- PolicyConfig suppression validation ------------------------------------ #


def test_suppression_rule_type_enforced() -> None:
    with pytest.raises(ValueError, match="must be a SuppressionRule"):
        PolicyConfig(suppressions={"loop": OverrideRule(1.0, Action.STOP)})  # type: ignore[dict-item]


def test_suppression_evidence_must_be_non_empty_string() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        PolicyConfig(suppressions={"loop": SuppressionRule(evidence="")})
    with pytest.raises(ValueError, match="non-empty string"):
        PolicyConfig(suppressions={"loop": SuppressionRule(evidence=None)})  # type: ignore[arg-type]


def test_suppression_min_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        PolicyConfig(
            suppressions={"loop": SuppressionRule(evidence="progress", min_confidence=1.5)}
        )
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        PolicyConfig(
            suppressions={"loop": SuppressionRule(evidence="progress", min_confidence=-0.1)}
        )


def test_suppression_min_confidence_bool_rejected() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        PolicyConfig(
            suppressions={"loop": SuppressionRule(evidence="progress", min_confidence=True)}  # type: ignore[arg-type]
        )


def test_suppression_min_confidence_int_coerced_to_float() -> None:
    config = PolicyConfig(
        suppressions={"loop": SuppressionRule(evidence="progress", min_confidence=1)}
    )
    rule = config.suppressions["loop"]
    assert rule.min_confidence == 1.0
    assert isinstance(rule.min_confidence, float)


def test_suppressions_defensively_copied() -> None:
    source = {"loop": SuppressionRule(evidence="progress", min_confidence=1.0)}
    config = PolicyConfig(suppressions=source)
    source["loop"] = SuppressionRule(evidence="other", min_confidence=0.0)
    assert config.suppressions["loop"].evidence == "progress"


# --- Pipeline integration: suppression prevents a real STOP ----------------- #


class _ScriptedLoop(Detector):
    """Emits a scripted loop confidence per event, mirroring _ScriptedDetector."""

    name = "loop"

    def __init__(self, confidences: Sequence[float]) -> None:
        self._confidences = list(confidences)

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        return Signal(self.name, confidence=self._confidences[len(history) - 1])


def test_pipeline_suppresses_loop_stop_while_progress_anchor_advances() -> None:
    # End-to-end version of the F1a fixture: a maxed loop signal would STOP on
    # its own, but progress_anchors + the suppression rule keep it silent while
    # the ground-truth test anchor sets a fresh record every tick. The first
    # tick's loop confidence is deliberately below stop_threshold: is_advancing
    # has no baseline yet on event 1 (see test_progress.py), so a realistic
    # LoopDetector -- which also needs >=2 repeats to reach high confidence --
    # never actually races the suppression on the very first event.
    detector = _ScriptedLoop([0.5, 0.95, 0.95])
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    pipeline = Pipeline(
        [detector],
        policy,
        progress_anchors=[TestProgressAnchor(), ExitCodeProgressAnchor()],
    )
    events = [
        ToolEvent(tool="run_tests", args={}, result={"passed": p, "failed": 0}, ts=float(i))
        for i, p in enumerate((1, 2, 3))
    ]
    actions = pipeline.run(events)

    assert actions == [Action.NONE, Action.NONE, Action.NONE]
    assert pipeline.stopped is False
    assert pipeline.post_mortem is None


def test_pipeline_stops_once_progress_evidence_goes_stale() -> None:
    # Same setup, but the test count stops climbing after the 2nd event: the
    # 3rd tick has no fresh evidence, so the suppression lifts and loop STOPs.
    detector = _ScriptedLoop([0.5, 0.95, 0.95])
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    pipeline = Pipeline(
        [detector],
        policy,
        progress_anchors=[TestProgressAnchor(), ExitCodeProgressAnchor()],
    )
    events = [
        ToolEvent(tool="run_tests", args={}, result={"passed": p, "failed": 0}, ts=float(i))
        for i, p in enumerate((1, 2, 2))
    ]
    actions = pipeline.run(events)

    assert actions == [Action.NONE, Action.NONE, Action.STOP]
    assert pipeline.stopped is True
    assert pipeline.post_mortem is not None
    assert pipeline.post_mortem.reason.startswith("loop")


def test_suppression_also_guards_a_real_loop2_ngram_confidence() -> None:
    # Non-regression (LOOP-2, #18): the suppression rule keys off the signal
    # *name* ("loop"), not its internal source. Same guarantee must hold when
    # the confidence comes from the real LoopDetector's n-gram fallback, not
    # just the exact-repeat component exercised above. Cycle is (open_file,
    # run_tests) interrupted every time by a distinct probe -- LOOP-1 stays
    # silent (see tests/test_loop.py), only the n-gram fallback fires -- while
    # run_tests' ever-increasing `passed` count is genuine progress evidence.
    events: list[ToolEvent] = []
    for i in range(10):
        events.append(ToolEvent(tool="open_file", args={"path": "src/worker.py"}))
        events.append(ToolEvent(tool="run_tests", args={}, result={"passed": i + 1, "failed": 0}))
        events.append(ToolEvent(tool=f"probe_{i}", args={"n": i}))
    # End right on a run_tests tick: is_advancing is strict on the *last* event
    # only (see test_progress.py), so the trailing probe would make it abstain.
    events.append(ToolEvent(tool="open_file", args={"path": "src/worker.py"}))
    events.append(ToolEvent(tool="run_tests", args={}, result={"passed": 11, "failed": 0}))

    loop_signal = LoopDetector().evaluate(events)
    assert loop_signal.confidence > 0.0
    assert "recurring" in loop_signal.reason  # confirms the n-gram path fired, not LOOP-1

    anchors = loop_suppression_anchors()
    assert is_advancing(anchors, events) is True

    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    assert policy.fuse([loop_signal], {"progress": 1.0}) == 0.0
    assert policy.fuse([loop_signal], {"progress": 0.0}) == pytest.approx(loop_signal.confidence)
    assert policy.fuse([loop_signal]) == pytest.approx(loop_signal.confidence)


def test_pipeline_without_progress_anchors_never_suppresses() -> None:
    # A pipeline that never opts into progress_anchors computes no evidence at
    # all, so a configured suppression rule stays inert -- identical to the
    # pre-LOOP-2 pipeline.
    detector = _ScriptedLoop([0.95])
    policy = Policy(PolicyConfig(suppressions=_LOOP_SUPPRESSED_BY_PROGRESS))
    pipeline = Pipeline([detector], policy)
    events = [
        ToolEvent(tool="run_tests", args={}, result={"passed": 1, "failed": 0}, ts=0.0),
    ]
    actions = pipeline.run(events)
    assert actions == [Action.STOP]
