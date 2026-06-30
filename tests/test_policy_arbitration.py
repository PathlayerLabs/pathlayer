"""Tests for arbitration: weighted noisy-OR fusion + single stop threshold.

Both directions matter: a strong (or converging) signal must stop the run, and
anything below the threshold must stay inert. ARB-2 shipped the single
``stop_threshold``; ARB-1 replaces the raw ``max`` with a weighted noisy-OR so
distinct detectors reinforce each other while a single maxed detector (default
weight 1.0) still stops. The threshold and the per-detector weights are
configurable.
"""

from __future__ import annotations

import pytest

from pathlayer.detectors.base import Signal
from pathlayer.policies import Action, Policy
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
