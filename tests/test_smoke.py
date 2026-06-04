"""Smoke tests: the skeleton imports and the default policy is inert."""

from pathlayer import __version__
from pathlayer.detectors.base import Signal
from pathlayer.policies import Action, Policy


def test_version() -> None:
    assert __version__


def test_default_policy_is_inert() -> None:
    # With no signals, the default policy should not intervene.
    assert Policy().decide([]) is Action.NONE


def test_signal_roundtrip() -> None:
    s = Signal(name="loop", confidence=0.0, reason="stub")
    assert s.name == "loop"
    assert s.confidence == 0.0
