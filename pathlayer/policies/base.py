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

ARB-3 turns the single ``stop_threshold`` into a graded ladder and adds
hysteresis:

- **Ladder (multi-threshold mapping).** :meth:`Policy.target_tier` maps the fused
  confidence to the highest *active* tier whose threshold it reaches. Tiers are
  ``warn -> summarize -> fallback -> escalate -> stop`` (:class:`Action`). STOP is
  always active at ``stop_threshold`` (the single source of truth for the STOP
  threshold); intermediate tiers are opt-in via ``tier_thresholds`` and stay
  *off* by default, so the zero-config policy is exactly ARB-1/ARB-2 (NONE/STOP).
- **Hysteresis (anti-flapping).** A noisy signal that crosses a threshold for a
  single tick must not flip the action up and back down. :meth:`Policy.step`
  threads an explicit :class:`HysteresisState` and only commits a tier change
  after ``escalate_ticks`` (up) / ``deescalate_ticks`` (down) *consecutive*
  confirming ticks, then **jumps straight to the current target tier** -- never
  one tier per N ticks, which would multiply STOP latency by the tier count and
  regress the loop / no-progress cases. With the default ``escalate_ticks=1`` the
  behaviour is identical to ARB-1 (STOP on the first tick at/above threshold).

Both ``target_tier`` and ``step`` are pure: no clock, no randomness; a "tick" is
one call. The pipeline owns the :class:`HysteresisState` and feeds it back each
event, so the same event sequence yields the same action sequence.

ARB-4 adds **priority overrides**: some signals do not deserve a graduated
response. A configured :class:`OverrideRule` (``detector -> {threshold, action}``)
short-circuits both fusion *and* hysteresis -- when an actionable signal reaches
its override threshold, :meth:`Policy.step` commits the rule's action
immediately, regardless of the other detectors' fused score or the hysteresis
counters. The headline case is *budget at ceiling = stop now* (wired
automatically by the config layer when the budget detector is enabled). Two
guarantees keep it safe:

- **Advisory still wins.** An override only fires on *actionable* signals; a
  detector listed in ``advisory_signals`` stays observe-only even with a rule.
  The config layer rejects a name that is both advisory and overridden.
- **Execution depends on the actuator.** STOP is the only hard termination.
  LangGraph implements WARN and SUMMARIZE; FALLBACK and ESCALATE are committed
  and logged but remain unimplemented until ACT-3/ACT-4.

LOOP-2 precision adds **conditional suppression**: a configured
:class:`SuppressionRule` (``guarded signal -> {evidence, min_confidence}``)
excludes a signal from ``_actionable`` -- and therefore from fusion, the
ladder, overrides and post-mortem attribution -- for any tick where the named
``evidence`` (an auxiliary ``Mapping[str, float]`` passed alongside
``signals``, never itself a :class:`Signal`) is fresh. The motivating bug: an
agent that re-runs the same test suite and improves every time (``passed``
climbing) is real progress, but ``LoopDetector`` cannot see ``result`` by
design and stops it as a repetition. ``evidence`` is structurally kept out of
the ``Sequence[Signal]`` that feeds :meth:`fuse` -- unlike ``advisory_signals``,
which excludes a signal from acting but still computes and logs it, evidence
is a completely different kind of input, so a caller can never accidentally
list it as advisory and have a *strong* progress reading count *toward* STOP,
the inverse of its purpose. The suppression check happens inside
``_actionable`` itself (not just at the top of ``fuse``) so every method built
on it -- ``dominant_signal``, the override machinery, ``target_tier`` -- agrees
a suppressed signal did not drive the tick, keeping post-mortems honest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from pathlayer.detectors.base import Signal
from pathlayer.policies.actions import Action

# Shared empty evidence mapping: the default for every ``evidence`` parameter
# below. A ``MappingProxyType`` (not a literal ``{}``) so it is immutable and
# safely shareable as a default argument -- no per-call allocation, no risk of
# a caller mutating a "default" that every other caller also sees.
_NO_EVIDENCE: Mapping[str, float] = MappingProxyType({})


def _normalized_confidence(confidence: float) -> float:
    """Clamp a raw ``Signal.confidence`` into ``[0, 1]``, ``NaN`` -> ``0.0``.

    ``Signal.confidence`` is documented as living in ``[0, 1]`` (see
    ``pathlayer/detectors/base.py``) but the dataclass itself does not enforce
    it -- Pathlayer never raises from inside the detection loop on a custom
    detector's stray value (see :meth:`Policy.fuse`), so every site that
    *compares* or *multiplies* confidences normalizes at the point of use
    instead. ``NaN`` fails every ``<``/``>`` comparison, so left unnormalized it
    would make ``min``/``max`` silently pick based on iteration order rather
    than on the actual value -- mapped to the inert ``0.0`` first, then finite
    values are clamped into range.
    """
    return 0.0 if math.isnan(confidence) else max(0.0, min(1.0, confidence))


# Tiers that can be configured via ``tier_thresholds`` -- the graded ladder
# between NONE and STOP. STOP is excluded on purpose: it is the single source of
# truth ``stop_threshold``; NONE is the absence of action, never a threshold.
_LADDER_TIERS: tuple[Action, ...] = (
    Action.WARN,
    Action.SUMMARIZE,
    Action.FALLBACK,
    Action.ESCALATE,
)

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

# Tiers an override rule may target: any actionable tier (WARN..STOP), never
# NONE (an override is by definition an action).
_OVERRIDE_ACTIONS: frozenset[Action] = frozenset(_LADDER_TIERS) | {Action.STOP}


@dataclass(slots=True, frozen=True)
class OverrideRule:
    """A priority-override rule (ARB-4): confidence threshold -> immediate action.

    When an actionable signal from the matched detector reaches ``threshold``,
    the policy commits ``action`` at once, bypassing fusion and hysteresis.
    ``threshold`` is a confidence in [0, 1]; ``action`` is an actionable tier
    (WARN..STOP, never NONE). Validated by :meth:`PolicyConfig.__post_init__`.
    """

    threshold: float
    action: Action


@dataclass(slots=True, frozen=True)
class SuppressionRule:
    """A guarded signal cannot drive an action while its evidence fires (LOOP-2 precision).

    ``evidence`` is a key into the ``evidence`` mapping threaded through
    :class:`Policy`'s methods (e.g. ``"progress"``) -- **not** a detector or
    :class:`Signal` name. It never appears in ``signals``, so it structurally
    cannot double as its own suppression trigger. ``min_confidence`` is the
    evidence value (in ``[0, 1]``) at or above which the guarded signal is
    excluded from ``_actionable`` for that tick. Validated by
    :meth:`PolicyConfig.__post_init__`.
    """

    evidence: str
    min_confidence: float = 1.0


@dataclass(slots=True, frozen=True)
class HysteresisState:
    """Explicit, threaded state of the hysteresis machine (ARB-3).

    Immutable so a tick produces a *new* state rather than mutating shared state:
    the pipeline owns the latest one and feeds it back. ``up_count`` /
    ``down_count`` track consecutive confirming ticks toward escalation /
    de-escalation; the opposite counter resets whenever the pressure changes
    direction so a single counter-tick cannot let a flapping signal accumulate.
    """

    committed: Action = Action.NONE
    up_count: int = 0
    down_count: int = 0


@dataclass(slots=True)
class PolicyConfig:
    # Fused confidence at/above which the run is stopped. Single source of truth
    # for the STOP tier (never duplicated in ``tier_thresholds``).
    stop_threshold: float = 0.9
    # Detector names whose signals are observe-only: logged but never STOP.
    advisory_signals: frozenset[str] = DEFAULT_ADVISORY_SIGNALS
    # Per-detector fusion weights in [0, 1]; missing names default to 1.0.
    weights: Mapping[str, float] = field(default_factory=dict)
    # Opt-in intermediate ladder tiers (WARN..ESCALATE) -> fused-confidence
    # threshold in [0, 1]. Empty by default: only STOP is active, so the policy
    # is exactly ARB-1 (NONE/STOP). Each must be < stop_threshold and
    # non-decreasing with tier severity, so every tier is reachable in order.
    tier_thresholds: Mapping[Action, float] = field(default_factory=dict)
    # Consecutive confirming ticks required before committing a tier change.
    # 1 = no hysteresis (follow the target immediately, like ARB-1).
    escalate_ticks: int = 1
    deescalate_ticks: int = 1
    # Priority overrides (ARB-4): detector name -> rule. An actionable signal at
    # or above its rule threshold commits the rule action immediately, bypassing
    # fusion and hysteresis. Empty by default: zero-config stays exactly ARB-1/2
    # (NONE/STOP via stop_threshold). The budget->stop override is wired by the
    # config layer only when the budget detector is enabled.
    overrides: Mapping[str, OverrideRule] = field(default_factory=dict)
    # Conditional suppression (LOOP-2 precision): guarded signal name -> rule.
    # Empty by default: zero-config stays exactly as before this feature. Wired
    # automatically by the config layer (loop <- progress) when no_progress
    # (and therefore its anchors) is enabled -- never a user-facing config key.
    suppressions: Mapping[str, SuppressionRule] = field(default_factory=dict)
    # Runtime-wired, reason-scoped overrides. Appended after the historical
    # public fields to preserve positional PolicyConfig construction.
    conditional_overrides: Mapping[str, tuple[str, OverrideRule]] = field(default_factory=dict)

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

        self.tier_thresholds = self._validate_tier_thresholds()

        for attr in ("escalate_ticks", "deescalate_ticks"):
            ticks = getattr(self, attr)
            # bool is an int subclass: reject it so True does not become 1 tick.
            if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
                raise ValueError(f"{attr} must be an int >= 1")

        self.overrides = self._validate_overrides()
        conditional: dict[str, tuple[str, OverrideRule]] = {}
        for name, entry in dict(self.conditional_overrides).items():
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0]
            ):
                raise ValueError(f"conditional override for {name!r} must be (reason_prefix, rule)")
            prefix, rule = entry
            conditional[name] = (prefix, self._validate_override(name, rule))
        self.conditional_overrides = conditional
        self.suppressions = self._validate_suppressions()

    def _validate_suppressions(self) -> dict[str, SuppressionRule]:
        """Validate and defensively copy the conditional-suppression rules.

        Each value must be a :class:`SuppressionRule` with a non-empty string
        ``evidence`` key and a ``min_confidence`` in ``[0, 1]``. Keys are
        guarded signal names; typos are the config layer's job to reject
        (mirroring ``weight_for``'s permissiveness), since this constructor has
        no notion of which detector names actually exist.
        """
        validated: dict[str, SuppressionRule] = {}
        for name, rule in dict(self.suppressions).items():
            if not isinstance(rule, SuppressionRule):
                raise ValueError(f"suppression for {name!r} must be a SuppressionRule")
            if not isinstance(rule.evidence, str) or not rule.evidence:
                raise ValueError(f"suppression evidence for {name!r} must be a non-empty string")
            min_confidence = rule.min_confidence
            if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
                raise ValueError(f"suppression min_confidence for {name!r} must be a number")
            min_confidence = float(min_confidence)
            if not 0.0 <= min_confidence <= 1.0:
                raise ValueError(f"suppression min_confidence for {name!r} must be in [0, 1]")
            validated[name] = SuppressionRule(evidence=rule.evidence, min_confidence=min_confidence)
        return validated

    def _validate_tier_thresholds(self) -> dict[Action, float]:
        """Validate and defensively copy the intermediate ladder thresholds.

        Keys must be intermediate tiers (WARN..ESCALATE) -- STOP is owned by
        ``stop_threshold`` and NONE is not a tier. Each threshold is a number in
        [0, 1], strictly below ``stop_threshold`` (so STOP stays the top), and
        the set is non-decreasing with tier severity (so a higher tier never
        triggers before a lower one).
        """
        validated: dict[Action, float] = {}
        for tier, raw in dict(self.tier_thresholds).items():
            if not isinstance(tier, Action) or tier not in _LADDER_TIERS:
                raise ValueError(
                    "tier_thresholds keys must be intermediate tiers "
                    "(WARN, SUMMARIZE, FALLBACK, ESCALATE); "
                    "STOP uses stop_threshold and NONE is not a tier"
                )
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"tier threshold for {tier.name} must be a number in [0, 1]")
            threshold = float(raw)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"tier threshold for {tier.name} must be in [0, 1]")
            if threshold >= self.stop_threshold:
                raise ValueError(f"tier threshold for {tier.name} must be < stop_threshold")
            validated[tier] = threshold
        previous: float | None = None
        for tier in _LADDER_TIERS:
            if tier not in validated:
                continue
            threshold = validated[tier]
            if previous is not None and threshold < previous:
                raise ValueError("tier_thresholds must be non-decreasing with tier severity")
            previous = threshold
        return validated

    def _validate_overrides(self) -> dict[str, OverrideRule]:
        """Validate and defensively copy the priority-override rules (ARB-4).

        Each value must be an :class:`OverrideRule` whose ``threshold`` is a
        number in [0, 1] and whose ``action`` is an actionable tier (WARN..STOP,
        never NONE). Keys are detector names; typos are the config layer's job to
        reject (mirroring ``weight_for``'s permissiveness).
        """
        validated: dict[str, OverrideRule] = {}
        for name, rule in dict(self.overrides).items():
            validated[name] = self._validate_override(name, rule)
        return validated

    @staticmethod
    def _validate_override(name: str, rule: OverrideRule) -> OverrideRule:
        if not isinstance(rule, OverrideRule):
            raise ValueError(f"override for {name!r} must be an OverrideRule")
        threshold = rule.threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"override threshold for {name!r} must be a number in [0, 1]")
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"override threshold for {name!r} must be in [0, 1]")
        if rule.action not in _OVERRIDE_ACTIONS:
            raise ValueError(
                f"override action for {name!r} must be an actionable tier "
                "(WARN, SUMMARIZE, FALLBACK, ESCALATE, STOP)"
            )
        return OverrideRule(threshold=threshold, action=rule.action)

    def weight_for(self, name: str) -> float:
        """Fusion weight for a detector name (``DEFAULT_WEIGHT`` if unset).

        Permissive on purpose: an unknown name (a custom detector wired in code)
        gets full weight. Rejecting typos is the config layer's job, not the
        fusion's.
        """
        return self.weights.get(name, DEFAULT_WEIGHT)

    def active_tiers(self) -> dict[Action, float]:
        """All actionable tiers and their thresholds, STOP always included.

        STOP is derived from ``stop_threshold`` here rather than stored, keeping
        a single source of truth for the STOP threshold.
        """
        tiers = dict(self.tier_thresholds)
        tiers[Action.STOP] = self.stop_threshold
        return tiers


@dataclass(slots=True)
class Policy:
    """Arbitration: fuse detector signals into a graded :class:`Action`.

    Extension contract: subclassing and overriding ``step`` (or any other
    method below) is an established pattern in this codebase (see the
    scripted test-double policies in ``tests/``), not just an implementation
    detail. Every overridable method takes an ``evidence: Mapping[str,
    float]`` parameter with a default (see ``_NO_EVIDENCE``) so omitting it
    keeps pre-LOOP-2 behaviour. A subclass that overrides one of these
    methods must keep that parameter in its signature -- ``Pipeline`` always
    calls ``step`` with three positional arguments, so an override missing
    ``evidence`` raises ``TypeError`` at runtime rather than silently
    dropping suppression.
    """

    config: PolicyConfig = field(default_factory=PolicyConfig)

    def _actionable(
        self, signals: Sequence[Signal], evidence: Mapping[str, float] = _NO_EVIDENCE
    ) -> list[Signal]:
        """Signals eligible to act: not advisory, and not currently suppressed.

        A signal is dropped when it is listed in ``advisory_signals`` (observe
        only, unconditionally) *or* when a :class:`SuppressionRule` guards it and
        ``evidence`` carries its trigger at or above ``min_confidence`` (LOOP-2
        precision: e.g. ``loop`` suppressed while ``evidence["progress"]`` shows
        a fresh record). ``evidence`` defaults to empty, so omitting it is
        exactly today's behaviour -- no suppression ever fires.
        """
        out: list[Signal] = []
        for signal in signals:
            if signal.name in self.config.advisory_signals:
                continue
            rule = self.config.suppressions.get(signal.name)
            if rule is not None and evidence.get(rule.evidence, 0.0) >= rule.min_confidence:
                continue
            out.append(signal)
        return out

    def fuse(
        self, signals: Sequence[Signal], evidence: Mapping[str, float] = _NO_EVIDENCE
    ) -> float:
        """Fuse signals into one confidence in [0, 1] via weighted noisy-OR.

        Advisory and currently-suppressed signals are excluded first (see
        :meth:`_actionable`). With no actionable signal the product is empty
        (1.0) so the fused confidence is 0.0.

        Each ``signal.confidence`` is normalized into ``[0, 1]`` before entering
        the product (see :func:`_normalized_confidence`). Built-in detectors
        already stay in range, so this is a guard for custom detectors: a stray
        ``confidence`` outside ``[0, 1]`` would push the noisy-OR product out of
        ``[0, 1]`` (e.g. ``2.0`` -> ``fused == 2.0``, a spurious STOP; a negative
        value could even invert the product). We normalize rather than ``raise``
        on purpose: the fusion runs inside the detection loop, and Pathlayer must
        never crash the agent it observes. Detector range violations are
        surfaced by unit tests, not by a runtime exception.
        """
        product = 1.0
        for signal in self._actionable(signals, evidence):
            product *= 1.0 - self._weighted_contribution(signal)
        return 1.0 - product

    def _weighted_contribution(self, signal: Signal) -> float:
        """``weight * normalized(confidence)``, the common ranking key.

        Shared by :meth:`dominant_signal` and :meth:`override_signal` so both
        stay in lockstep with :meth:`fuse`'s normalization -- an unnormalized
        ``NaN`` fails every comparison, which would otherwise make ``max()``
        pick based on iteration order instead of on the actual weighted score
        (found by audit issue #108: it named the wrong signal, or one carrying
        ``NaN``, depending on list order).
        """
        return self.config.weight_for(signal.name) * _normalized_confidence(signal.confidence)

    def dominant_signal(
        self, signals: Sequence[Signal], evidence: Mapping[str, float] = _NO_EVIDENCE
    ) -> Signal | None:
        """The actionable signal that contributes most to the fused score.

        Ranked by weighted contribution (``weight * confidence``), not raw
        confidence, so a STOP driven by convergence names the signal that
        actually drove it rather than an unrelated louder-but-down-weighted one.
        A signal currently suppressed (see :meth:`_actionable`) can never be
        named here, so a post-mortem never blames a detector that was excluded
        from the decision. ``None`` when there is no actionable signal.
        """
        actionable = self._actionable(signals, evidence)
        if not actionable:
            return None
        return max(actionable, key=self._weighted_contribution)

    def _triggered_overrides(
        self, signals: Sequence[Signal], evidence: Mapping[str, float] = _NO_EVIDENCE
    ) -> list[tuple[Signal, OverrideRule]]:
        """Actionable ``(signal, rule)`` pairs whose confidence meets the rule.

        Advisory and currently-suppressed signals are excluded first (an
        override never resurrects a signal :meth:`_actionable` dropped). Empty
        when no rule fires -- the common case, so the normal fusion/hysteresis
        path runs unchanged.
        """
        overrides = self.config.overrides
        conditional = self.config.conditional_overrides
        if not overrides and not conditional:
            return []
        triggered: list[tuple[Signal, OverrideRule]] = []
        for signal in self._actionable(signals, evidence):
            rule = overrides.get(signal.name)
            if rule is None:
                conditional_entry = conditional.get(signal.name)
                if conditional_entry is not None and signal.reason.startswith(conditional_entry[0]):
                    rule = conditional_entry[1]
            # Normalize before comparing, like every other confidence site
            # (fusion, ladder, post-mortem): a custom detector may emit a raw
            # confidence outside [0, 1] or NaN, and the dataclass does not
            # enforce the range. Left raw, an out-of-range value would fire (or
            # skip) an override on a number the rest of the policy never sees.
            if rule is not None and _normalized_confidence(signal.confidence) >= rule.threshold:
                triggered.append((signal, rule))
        return triggered

    def override_action(
        self, signals: Sequence[Signal], evidence: Mapping[str, float] = _NO_EVIDENCE
    ) -> Action | None:
        """The most severe overriding action for these signals, else ``None`` (ARB-4).

        Scans the configured :class:`OverrideRule`\\ s: an *actionable* signal at
        or above its rule threshold yields that rule's action; the most severe
        wins. ``None`` means no override fired, so :meth:`step` / :meth:`decide`
        fall back to the normal fused-confidence path.
        """
        triggered = self._triggered_overrides(signals, evidence)
        if not triggered:
            return None
        return max(rule.action for _, rule in triggered)

    def override_signal(
        self, signals: Sequence[Signal], evidence: Mapping[str, float] = _NO_EVIDENCE
    ) -> Signal | None:
        """The signal that drove the winning override, for reporting (ARB-4).

        The actionable signal behind the most severe fired rule (tie-broken by
        weighted contribution ``weight * confidence``), so a post-mortem names
        the real trigger -- e.g. ``budget`` -- even when that detector carries a
        fusion weight of 0. ``None`` when no override fired.
        """
        triggered = self._triggered_overrides(signals, evidence)
        if not triggered:
            return None
        top = max(rule.action for _, rule in triggered)
        candidates = [signal for signal, rule in triggered if rule.action == top]
        return max(candidates, key=self._weighted_contribution)

    def target_tier(
        self, signals: Sequence[Signal], evidence: Mapping[str, float] = _NO_EVIDENCE
    ) -> Action:
        """Highest active tier whose threshold the fused confidence reaches.

        Pure mapping of the weighted noisy-OR to a graded :class:`Action`: the
        most severe tier in :meth:`PolicyConfig.active_tiers` with
        ``threshold <= fused`` (``>=`` to match ``decide``'s at-threshold STOP).
        With no actionable signal the run cannot act (you cannot act on nothing),
        so the target is ``NONE``. This is the *instantaneous* target -- the
        hysteresis in :meth:`step` decides when it is actually committed.
        """
        actionable = self._actionable(signals, evidence)
        if not actionable:
            return Action.NONE
        fused = self.fuse(signals, evidence)
        target = Action.NONE
        for tier, threshold in self.config.active_tiers().items():
            if fused >= threshold and tier > target:
                target = tier
        return target

    def step(
        self,
        state: HysteresisState,
        signals: Sequence[Signal],
        evidence: Mapping[str, float] = _NO_EVIDENCE,
    ) -> tuple[HysteresisState, Action]:
        """Advance the hysteresis machine by one tick; return state + action.

        Counts *consecutive* confirming ticks and, once the threshold is met,
        jumps **straight to the current target tier** (not one tier per N ticks).
        Escalation needs ``escalate_ticks`` ticks with ``target > committed``;
        de-escalation needs ``deescalate_ticks`` ticks with ``target <
        committed``. Any tick that does not confirm the in-progress direction
        resets the opposite counter, so a single contrary tick collapses a
        building spike -- the anti-flapping guarantee. A pure function of
        ``(state, signals)``: same inputs, same output.

        The counter tracks *pressure away from the committed tier*, not time
        spent at the destination tier. So when the fused confidence climbs
        through tiers, the jump lands on whatever the target is at the tick the
        counter trips -- STOP can therefore commit after fewer than
        ``escalate_ticks`` ticks *at STOP level* (e.g. with ``escalate_ticks=2``
        a WARN-level tick followed by a STOP-level tick commits STOP). This is
        intentional: a sustained, rising signal escalates promptly, while a lone
        spike from baseline still resets and never escalates.

        Priority overrides (ARB-4) come first: if a rule fires, we commit at once
        -- bypassing the hysteresis counters -- to ``max(override_action,
        target_tier)``. Taking the max means an override never *downgrades* a more
        severe action the fusion would reach on its own (non-override signals
        still follow the normal graduated path); a budget ceiling therefore stops
        now regardless of ``escalate_ticks``, and a lower-tier override can only
        escalate, never mask a fused STOP.

        ``evidence`` (LOOP-2 precision) flows into every signal-eligibility check
        below (``target_tier``, ``override_action``), so a signal it suppresses
        for this tick cannot drive an override, the fused score, or hysteresis.
        """
        override = self.override_action(signals, evidence)
        if override is not None:
            committed = max(override, self.target_tier(signals, evidence))
            return HysteresisState(committed, 0, 0), committed
        target = self.target_tier(signals, evidence)
        committed = state.committed
        if target > committed:
            up_count = state.up_count + 1
            if up_count >= self.config.escalate_ticks:
                return HysteresisState(target, 0, 0), target
            return HysteresisState(committed, up_count, 0), committed
        if target < committed:
            down_count = state.down_count + 1
            if down_count >= self.config.deescalate_ticks:
                return HysteresisState(target, 0, 0), target
            return HysteresisState(committed, 0, down_count), committed
        return HysteresisState(committed, 0, 0), committed

    def decide(
        self, signals: Sequence[Signal], evidence: Mapping[str, float] = _NO_EVIDENCE
    ) -> Action:
        """Map the fused *actionable* confidence to a graded action, **no hysteresis**.

        Stateless single-tick mapping kept for direct use and back-compat: if the
        weighted noisy-OR of the non-advisory, non-suppressed signals reaches
        ``stop_threshold``, return ``STOP``; otherwise ``NONE``. It ignores
        intermediate ladder tiers and applies no hysteresis -- the pipeline
        drives :meth:`step` instead, so anti-flapping and the graded ladder only
        apply through ``step``. Advisory signals are still computed and logged
        upstream; with no actionable signal the run is never stopped (you cannot
        stop on nothing).

        Priority overrides (ARB-4) are honoured here too, for parity with
        :meth:`step`: the result is ``max(override_action, base)`` where ``base``
        is the usual NONE/STOP, so an override can escalate but never downgrade.
        """
        actionable = self._actionable(signals, evidence)
        base = Action.NONE
        if actionable and self.fuse(actionable, evidence) >= self.config.stop_threshold:
            base = Action.STOP
        override = self.override_action(signals, evidence)
        if override is not None:
            return max(override, base)
        return base
