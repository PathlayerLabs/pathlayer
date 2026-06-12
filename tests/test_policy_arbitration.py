"""Tests for minimal arbitration (ARB-2): single stop threshold.

Both directions matter: a strong signal must stop the run, and anything below
the threshold must stay inert. The threshold is configurable.
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


def test_uses_strongest_signal() -> None:
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
