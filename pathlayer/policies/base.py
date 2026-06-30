"""Arbitration: combine detector signals into a graded action.

The policy itself only needs ``stop_threshold`` and the fusion ``weights`` (plus
which signals are observe-only); every per-detector threshold (``max_cost_usd``,
retry/loop counts, ...) lives on its own detector, configured via
:mod:`pathlayer.config`. There is a single source of truth per threshold.

ARB-1 fuses the detector signals into a single confidence with a **weighted
noisy-OR**: ``fused = 1 - prod(1 - weightᵢ * confidenceᵢ)`` over the actionable
signals. It stays in [0, 1], is monotone (more/stronger signals never lower the
score), and is a pure function of the signals -- no clock, no randomness. With
the default weight of 1.0 a single maxed-out detector still reaches 1.0 (no
regression vs the previous ``max``), while two independent signals reinforce
each other (0.6 + 0.6 -> 0.84), which is the point: convergence of distinct
failure modes is stronger evidence than any one alone.

Two consequences worth keeping in mind:

- **The fused floor rises with the number of detectors.** Each enabled detector
  adds a factor to the product, so ``stop_threshold`` should be re-validated
  against the golden set whenever the detector set changes (e.g. enabling
  ``budget``/``resource``).
- **A per-detector weight of 0 makes that signal inert** -- equivalent to
  listing it in ``advisory_signals``, just expressed numerically. Both knobs
  compose; advisory exclusion happens first.

The 5-tier ladder (warn -> summarize -> fallback -> escalate -> stop) and
hysteresis stay deferred (ARB-3); ARB-1 keeps the single ``stop_threshold``
mapping fused confidence to ``STOP``/``NONE``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pathlayer.detectors.base import Signal
from pathlayer.policies.actions import Action

# Signals delivered in "observe" mode: computed and logged, but excluded from
# the STOP decision. Empty by default since NOP-3: no_progress earns STOP
# authority because anchor success suppression (``ProgressAnchor.is_satisfied``)
# resolves the "done vs stuck" ambiguity that previously kept it advisory.
# Kept as a configurable knob -- a minimal precursor to dry-run mode (OBS-3),
# and any detector can be made observe-only per deployment.
DEFAULT_ADVISORY_SIGNALS: frozenset[str] = frozenset()

# Weight applied to a detector whose name has no explicit entry in ``weights``.
# 1.0 = full authority, so the zero-config policy behaves as if every detector
# contributes at full strength.
DEFAULT_WEIGHT: float = 1.0


@dataclass(slots=True)
class PolicyConfig:
    # Fused confidence at/above which the run is stopped.
    stop_threshold: float = 0.9
    # Detector names whose signals are observe-only: logged but never STOP.
    advisory_signals: frozenset[str] = DEFAULT_ADVISORY_SIGNALS
    # Per-detector fusion weights in [0, 1]; missing names default to 1.0.
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # stop_threshold is a confidence in [0, 1]; reject misconfigurations
        # early so they don't silently turn into always-stop / never-stop.
        if not 0.0 <= self.stop_threshold <= 1.0:
            raise ValueError("stop_threshold must be in [0, 1]")
        # Defensive copy: a caller's mapping must not alias policy state. Validate
        # type then range -- a weight outside [0, 1] would push the noisy-OR
        # product out of [0, 1], and (mirroring the config layer) a bool or
        # non-number must not silently misconfigure fusion (bool is an int
        # subclass, so True would otherwise sneak in as weight 1.0).
        weights: dict[str, float] = {}
        for name, weight in dict(self.weights).items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"weight for {name!r} must be a number in [0, 1]")
            weight = float(weight)
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"weight for {name!r} must be in [0, 1]")
            weights[name] = weight
        self.weights = weights

    def weight_for(self, name: str) -> float:
        """Fusion weight for a detector name (``DEFAULT_WEIGHT`` if unset).

        Permissive on purpose: an unknown name (a custom detector wired in code)
        gets full weight. Rejecting typos is the config layer's job, not the
        fusion's.
        """
        return self.weights.get(name, DEFAULT_WEIGHT)


@dataclass(slots=True)
class Policy:
    config: PolicyConfig = field(default_factory=PolicyConfig)

    def _actionable(self, signals: Sequence[Signal]) -> list[Signal]:
        """Signals eligible to act: everything not declared observe-only."""
        return [s for s in signals if s.name not in self.config.advisory_signals]

    def fuse(self, signals: Sequence[Signal]) -> float:
        """Fuse signals into one confidence in [0, 1] via weighted noisy-OR.

        Advisory signals are excluded first. With no actionable signal the
        product is empty (1.0) so the fused confidence is 0.0.
        """
        product = 1.0
        for signal in self._actionable(signals):
            product *= 1.0 - self.config.weight_for(signal.name) * signal.confidence
        return 1.0 - product

    def dominant_signal(self, signals: Sequence[Signal]) -> Signal | None:
        """The actionable signal that contributes most to the fused score.

        Ranked by weighted contribution (``weight * confidence``), not raw
        confidence, so a STOP driven by convergence names the signal that
        actually drove it rather than an unrelated louder-but-down-weighted one.
        ``None`` when there is no actionable signal.
        """
        actionable = self._actionable(signals)
        if not actionable:
            return None
        return max(actionable, key=lambda s: self.config.weight_for(s.name) * s.confidence)

    def decide(self, signals: Sequence[Signal]) -> Action:
        """Map the fused *actionable* confidence to a graded action.

        Single-threshold ladder: if the weighted noisy-OR of the non-advisory
        signals reaches ``stop_threshold``, return ``STOP``; otherwise ``NONE``.
        Advisory signals (see ``advisory_signals``) are still computed and logged
        upstream, they just cannot STOP on their own. With no actionable signal
        the run is never stopped (you cannot stop on nothing).
        """
        actionable = self._actionable(signals)
        if not actionable:
            return Action.NONE
        if self.fuse(actionable) >= self.config.stop_threshold:
            return Action.STOP
        return Action.NONE
