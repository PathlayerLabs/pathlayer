"""Configuration system (INF-4): one file for every tunable threshold.

Pathlayer is zero-config by default -- with no file present, :func:`load_config`
returns the built-in defaults, which match the values previously hardcoded in
each detector and in :class:`~pathlayer.policies.base.PolicyConfig`. A project
overrides any subset of them via ``.pathlayer/config.yaml`` (or ``.yml`` /
``.json``), without touching code.

Design choices:

- **JSON via the stdlib, YAML as an optional extra.** The core stays
  dependency-free (local-first): JSON always works; YAML is parsed with
  ``yaml.safe_load`` only, and only when a ``.yaml`` file is actually loaded. A
  missing PyYAML raises a clear, actionable error rather than a bare ImportError.
- **Fail fast on bad input.** Validation rejects unknown keys (per section, with
  the offending path named) and wrong types up front, so a typo never silently
  degrades into always-stop / never-stop. Value-range checks are delegated to the
  detectors' own ``__init__`` / ``__post_init__`` (e.g. ``k >= 2``,
  ``window >= 2*k``, ``stop_threshold in [0, 1]``) -- one source of truth.
- **Single build entry point.** :func:`build_pipeline` turns a config into a
  ready :class:`~pathlayer.pipeline.Pipeline`. It lives here (not as
  ``Pipeline.from_config``) to avoid a ``pipeline -> config -> pipeline`` import
  cycle.

All I/O happens here, at build time. The detection loop itself
(:meth:`Pipeline.process`) stays pure -- no file access, no clock, no network.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pathlayer.detectors.base import Detector
from pathlayer.detectors.budget import BudgetDetector, BudgetState, budget_delta
from pathlayer.detectors.canonical import ToolMask, ToolMasks
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors, loop_suppression_anchors
from pathlayer.detectors.resource import ResourceDetector
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent
from pathlayer.pipeline import PROGRESS_EVIDENCE_KEY, Pipeline
from pathlayer.policies import Action, OverrideRule, Policy, SuppressionRule
from pathlayer.policies.base import PolicyConfig
from pathlayer.pricing import DEFAULT_PRICES, ModelPrice
from pathlayer.store import (
    PersistentPipeline,
    SQLiteStateStore,
    StateStore,
    configuration_fingerprint,
)

__all__ = [
    "ConfigError",
    "PathlayerConfig",
    "PricingSection",
    "build_pipeline",
    "load_config",
]

# Default discovery location, relative to the current working directory.
CONFIG_DIR = ".pathlayer"
# Discovery candidates, in no particular priority: exactly one may exist.
_DISCOVERY_NAMES = ("config.yaml", "config.yml", "config.json")

# Detector signal names a config may legitimately reference (e.g. in
# ``advisory_signals``). Derived from the detectors themselves so it cannot
# drift from the actual ``name`` a detector emits. Used to reject typos that
# would otherwise silently no-op.
_KNOWN_SIGNALS: frozenset[str] = frozenset(
    {
        LoopDetector.name,
        RetryDetector.name,
        BudgetDetector.name,
        NoProgressDetector.name,
        ResourceDetector.name,
    }
)

# Config-facing names for the intermediate ladder tiers (ARB-3) -> Action.
# STOP is intentionally excluded: its threshold is ``policy.stop_threshold``
# (single source of truth); NONE is the absence of action, not a tier.
_LADDER_TIER_NAMES: dict[str, Action] = {
    Action.WARN.name.lower(): Action.WARN,
    Action.SUMMARIZE.name.lower(): Action.SUMMARIZE,
    Action.FALLBACK.name.lower(): Action.FALLBACK,
    Action.ESCALATE.name.lower(): Action.ESCALATE,
}

# Config-facing action names an override rule (ARB-4) may target. Unlike the
# ladder tiers, STOP is included: the headline override is budget -> stop.
_OVERRIDE_ACTION_NAMES: dict[str, Action] = {
    **_LADDER_TIER_NAMES,
    Action.STOP.name.lower(): Action.STOP,
}


class ConfigError(ValueError):
    """Raised when a config file is malformed: unknown key or bad value/type."""


# --------------------------------------------------------------------------- #
# Typed section readers (consume known keys, reject leftovers)
# --------------------------------------------------------------------------- #


def _as_mapping(value: Any, where: str) -> dict[str, Any]:
    """Coerce a parsed section to a dict. ``None`` / empty -> ``{}`` (defaults)."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        # Keys from JSON/YAML objects are strings; normalize defensively.
        return {str(key): val for key, val in value.items()}
    raise ConfigError(f"section '{where}' must be a mapping, got {type(value).__name__}")


class _Section:
    """Reads a config section by consuming known keys with type checks.

    After all known keys are popped, :meth:`finish` rejects anything left over,
    naming the section -- this is how unknown keys fail fast.
    """

    def __init__(self, data: Mapping[str, Any], name: str) -> None:
        self._data: dict[str, Any] = dict(data)
        self._name = name

    def _pop(self, key: str, default: Any) -> Any:
        return self._data.pop(key, default)

    def _err(self, key: str, expected: str, value: Any) -> ConfigError:
        return ConfigError(
            f"invalid value for '{self._name}.{key}': expected {expected}, "
            f"got {type(value).__name__}"
        )

    def bool_(self, key: str, default: bool) -> bool:
        value = self._pop(key, default)
        if not isinstance(value, bool):
            raise self._err(key, "bool", value)
        return value

    def int_(self, key: str, default: int) -> int:
        value = self._pop(key, default)
        # bool is an int subclass -- reject it so ``enabled``-style typos in a
        # numeric field are caught, and reject floats like ``2.0`` for an int.
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._err(key, "int", value)
        return value

    def float_(self, key: str, default: float) -> float:
        value = self._pop(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._err(key, "number", value)
        return float(value)

    def nonnegative_float(self, key: str, default: float) -> float:
        """Parse one finite number in ``[0, +inf)``."""
        number = self.float_(key, default)
        if not math.isfinite(number) or number < 0.0:
            raise ConfigError(
                f"invalid value for '{self._name}.{key}': must be finite and >= 0, got {number}"
            )
        return number

    def str_(self, key: str, default: str) -> str:
        value = self._pop(key, default)
        if not isinstance(value, str):
            raise self._err(key, "string", value)
        return value

    def str_list(self, key: str, default: Sequence[str]) -> list[str]:
        value = self._pop(key, list(default))
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise self._err(key, "list of strings", value)
        return list(value)

    def float_map(self, key: str) -> dict[str, float]:
        """Parse a mapping of names to floats in [0, 1]. Absent/``null`` -> ``{}``.

        Mirrors :meth:`float_`: rejects ``bool``, accepts ``int`` (coerced), and
        validates each value's range with a path-qualified error.
        """
        where = f"{self._name}.{key}"
        raw = _as_mapping(self._pop(key, {}), where)
        out: dict[str, float] = {}
        for subkey, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(
                    f"invalid value for '{where}.{subkey}': expected number, "
                    f"got {type(value).__name__}"
                )
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise ConfigError(
                    f"invalid value for '{where}.{subkey}': must be in [0, 1], got {number}"
                )
            out[str(subkey)] = number
        return out

    def nonnegative_float_map(self, key: str) -> dict[str, float]:
        """Parse a mapping of names to finite numbers in ``[0, +inf)``."""
        where = f"{self._name}.{key}"
        raw = _as_mapping(self._pop(key, {}), where)
        out: dict[str, float] = {}
        for subkey, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(
                    f"invalid value for '{where}.{subkey}': expected number, "
                    f"got {type(value).__name__}"
                )
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ConfigError(
                    f"invalid value for '{where}.{subkey}': must be finite and >= 0, got {number}"
                )
            out[str(subkey)] = number
        return out

    def finish(self) -> None:
        if self._data:
            unknown = ", ".join(repr(key) for key in sorted(self._data))
            raise ConfigError(f"unknown key(s) in section '{self._name}': {unknown}")


# --------------------------------------------------------------------------- #
# Config dataclasses (defaults == current hardcoded values)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LoopConfig:
    enabled: bool = True
    k: int = 2
    window: int = 256
    # LOOP-2: n-gram coverage fallback, only ever consulted when the exact
    # (LOOP-1) component above is silent. ``ngram_sizes`` stays code-level (like
    # no_progress's anchors) -- these scalar thresholds are what's tunable.
    ngram_min_hits: int = 3
    ngram_min_coverage: float = 0.3
    ngram_max_confidence: float = 0.85
    # LOOP-3 (#19): steepness of the sigmoid ramp centered on ngram_min_hits.
    ngram_steepness: float = 3.0


@dataclass(slots=True)
class RetryConfig:
    enabled: bool = True
    # RET-1: consecutive identical failing retries.
    threshold: int = 3
    # RET-2: one tool dominating the recent window regardless of args/error.
    window: int = 16
    dominance_min_calls: int = 5
    dominance_ratio: float = 0.75
    dominance_max_confidence: float = 0.85
    min_failures: int = 2


@dataclass(slots=True)
class BudgetConfig:
    # Disabled by default: enabling at the $5 default would re-introduce the
    # certified false positive on a legitimately expensive run ($6.40). Opt in
    # with a ceiling chosen for the workload.
    enabled: bool = False
    max_cost_usd: float = 5.0
    baseline_usd: float = 0.0
    default_cost_usd: float = 0.0
    tool_costs_usd: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class NoProgressConfig:
    enabled: bool = True
    target_rate: float = 0.2
    threshold: float = 2.0
    window: int = 64


@dataclass(slots=True)
class DetectorsConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    no_progress: NoProgressConfig = field(default_factory=NoProgressConfig)


@dataclass(slots=True)
class CanonicalizationConfig:
    # Per-tool keep/mask overrides (CAN-2). Empty by default = default heuristics.
    tool_masks: dict[str, ToolMask] = field(default_factory=dict)


@dataclass(slots=True)
class PolicySection:
    stop_threshold: float = 0.9
    advisory_signals: frozenset[str] = frozenset()
    # Per-detector fusion weights (ARB-1) in [0, 1]; missing names default to 1.0.
    weights: dict[str, float] = field(default_factory=dict)
    # Intermediate ladder tiers (ARB-3): tier name (warn/summarize/fallback/
    # escalate) -> fused-confidence threshold. Empty by default = only STOP.
    tier_thresholds: dict[str, float] = field(default_factory=dict)
    # Consecutive confirming ticks before committing a tier change (1 = none).
    escalate_ticks: int = 1
    deescalate_ticks: int = 1
    # Priority overrides (ARB-4): detector name -> rule (threshold + action).
    # Empty by default; the budget->stop rule is auto-wired by build_pipeline
    # only when the budget detector is enabled (and not advisory).
    overrides: dict[str, OverrideRule] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineSection:
    recent_history: int = 10
    # Dry-run (OBS-3): detectors and arbitration run and are logged, but no
    # action is emitted and the run never halts -- the observe-only mode for
    # tuning before trusting enforcement.
    dry_run: bool = False


@dataclass(slots=True)
class StoreSection:
    # SQLite state used when build_pipeline receives an explicit state_key.
    path: str = f"{CONFIG_DIR}/state.db"


@dataclass(slots=True)
class PricingSection:
    """Token prices feeding :func:`~pathlayer.pricing.cost_of` (USD per 1M tokens).

    Empty by default, in which case :meth:`effective_prices` returns exactly
    :data:`~pathlayer.pricing.DEFAULT_PRICES` -- configuring nothing changes
    nothing. ``models`` normally *extends* that table (add a model it does not
    list, or pin a sub-version whose price dropped); ``replace_defaults: true``
    makes it the whole table instead, for a deployment that wants to price only
    the models it has vetted and bill everything else at $0.

    Configuration only: nothing in Pathlayer reads this section today. A price
    table turns *reported token counts* into a cost, and the only adapter that
    reports them takes its table as a constructor argument, so the caller must
    pass it explicitly -- ``LangGraphAdapter(sink,
    prices=config.pricing.effective_prices())``. Until it does, this section
    changes no cost and no decision. The OpenClaw hooks report no tokens at all,
    so on that path it can produce no figure whatever the wiring.
    """

    models: dict[str, ModelPrice] = field(default_factory=dict)
    replace_defaults: bool = False

    def effective_prices(self) -> dict[str, ModelPrice]:
        """The full table to hand to ``cost_of(..., prices=...)``."""
        if self.replace_defaults:
            return dict(self.models)
        return {**DEFAULT_PRICES, **self.models}


@dataclass(slots=True)
class PathlayerConfig:
    detectors: DetectorsConfig = field(default_factory=DetectorsConfig)
    canonicalization: CanonicalizationConfig = field(default_factory=CanonicalizationConfig)
    policy: PolicySection = field(default_factory=PolicySection)
    pipeline: PipelineSection = field(default_factory=PipelineSection)
    pricing: PricingSection = field(default_factory=PricingSection)
    store: StoreSection = field(default_factory=StoreSection)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PathlayerConfig:
        """Build a validated config from a parsed mapping (fail fast)."""
        root = _Section(data, "<root>")
        detectors = _parse_detectors(_as_mapping(root._pop("detectors", {}), "detectors"))
        canonicalization = _parse_canonicalization(
            _as_mapping(root._pop("canonicalization", {}), "canonicalization")
        )
        policy = _parse_policy(_as_mapping(root._pop("policy", {}), "policy"))
        pipeline = _parse_pipeline(_as_mapping(root._pop("pipeline", {}), "pipeline"))
        pricing = _parse_pricing(_as_mapping(root._pop("pricing", {}), "pricing"))
        store = _parse_store(_as_mapping(root._pop("store", {}), "store"))
        root.finish()
        return cls(
            detectors=detectors,
            canonicalization=canonicalization,
            policy=policy,
            pipeline=pipeline,
            pricing=pricing,
            store=store,
        )


def _parse_detectors(data: Mapping[str, Any]) -> DetectorsConfig:
    section = _Section(data, "detectors")
    loop_raw = _as_mapping(section._pop("loop", {}), "detectors.loop")
    retry_raw = _as_mapping(section._pop("retry", {}), "detectors.retry")
    budget_raw = _as_mapping(section._pop("budget", {}), "detectors.budget")
    nop_raw = _as_mapping(section._pop("no_progress", {}), "detectors.no_progress")
    section.finish()

    loop_s = _Section(loop_raw, "detectors.loop")
    loop = LoopConfig(
        enabled=loop_s.bool_("enabled", True),
        k=loop_s.int_("k", 2),
        window=loop_s.int_("window", 256),
        ngram_min_hits=loop_s.int_("ngram_min_hits", 3),
        ngram_min_coverage=loop_s.float_("ngram_min_coverage", 0.3),
        ngram_max_confidence=loop_s.float_("ngram_max_confidence", 0.85),
        ngram_steepness=loop_s.float_("ngram_steepness", 3.0),
    )
    loop_s.finish()

    retry_s = _Section(retry_raw, "detectors.retry")
    retry = RetryConfig(
        enabled=retry_s.bool_("enabled", True),
        threshold=retry_s.int_("threshold", 3),
        window=retry_s.int_("window", 16),
        dominance_min_calls=retry_s.int_("dominance_min_calls", 5),
        dominance_ratio=retry_s.float_("dominance_ratio", 0.75),
        dominance_max_confidence=retry_s.float_("dominance_max_confidence", 0.85),
        min_failures=retry_s.int_("min_failures", 2),
    )
    retry_s.finish()

    budget_s = _Section(budget_raw, "detectors.budget")
    budget = BudgetConfig(
        enabled=budget_s.bool_("enabled", False),
        max_cost_usd=budget_s.float_("max_cost_usd", 5.0),
        baseline_usd=budget_s.float_("baseline_usd", 0.0),
        default_cost_usd=budget_s.nonnegative_float("default_cost_usd", 0.0),
        tool_costs_usd=budget_s.nonnegative_float_map("tool_costs_usd"),
    )
    budget_s.finish()

    nop_s = _Section(nop_raw, "detectors.no_progress")
    no_progress = NoProgressConfig(
        enabled=nop_s.bool_("enabled", True),
        target_rate=nop_s.float_("target_rate", 0.2),
        threshold=nop_s.float_("threshold", 2.0),
        window=nop_s.int_("window", 64),
    )
    nop_s.finish()

    return DetectorsConfig(loop=loop, retry=retry, budget=budget, no_progress=no_progress)


def _parse_canonicalization(data: Mapping[str, Any]) -> CanonicalizationConfig:
    section = _Section(data, "canonicalization")
    masks_raw = _as_mapping(section._pop("tool_masks", {}), "canonicalization.tool_masks")
    section.finish()

    tool_masks: dict[str, ToolMask] = {}
    for tool, spec in masks_raw.items():
        where = f"canonicalization.tool_masks.{tool}"
        spec_s = _Section(_as_mapping(spec, where), where)
        keep = spec_s.str_list("keep", [])
        mask = spec_s.str_list("mask", [])
        spec_s.finish()
        # ToolMask.__post_init__ rejects keep/mask overlap with a clear message.
        try:
            tool_masks[tool] = ToolMask(keep=frozenset(keep), mask=frozenset(mask))
        except ValueError as exc:
            raise ConfigError(f"invalid tool mask for '{tool}': {exc}") from exc
    return CanonicalizationConfig(tool_masks=tool_masks)


def _reject_unknown_signals(names: Iterable[str], where: str) -> None:
    """Reject any signal name not emitted by a known detector (anti-typo).

    A misspelled name would otherwise silently no-op (an advisory exclusion that
    excludes nothing, a weight applied to no detector).
    """
    unknown = sorted(set(names) - _KNOWN_SIGNALS)
    if unknown:
        known = ", ".join(sorted(_KNOWN_SIGNALS))
        raise ConfigError(
            f"unknown signal(s) in '{where}': "
            f"{', '.join(repr(name) for name in unknown)}; known signals: {known}"
        )


def _reject_unknown_tiers(names: Iterable[str], where: str) -> None:
    """Reject any ladder-tier name that is not a configurable intermediate tier.

    ``stop`` is rejected on purpose (its threshold is ``policy.stop_threshold``);
    ``none`` and typos are rejected so a misspelled tier never silently no-ops.
    """
    unknown = sorted(set(names) - set(_LADDER_TIER_NAMES))
    if unknown:
        known = ", ".join(sorted(_LADDER_TIER_NAMES))
        raise ConfigError(
            f"unknown tier(s) in '{where}': "
            f"{', '.join(repr(name) for name in unknown)}; "
            f"configurable tiers: {known} (STOP uses policy.stop_threshold)"
        )


def _parse_override_rules(raw: Mapping[str, Any]) -> dict[str, OverrideRule]:
    """Parse ``policy.overrides``: name -> {threshold, action} (ARB-4).

    ``threshold`` defaults to 1.0 (fire only at a maxed-out signal, the budget
    ceiling case) and ``action`` to ``stop``. Range/type of ``threshold`` is
    validated here for a path-qualified error; the action name is mapped through
    :data:`_OVERRIDE_ACTION_NAMES` (STOP allowed, unlike ladder tiers).
    """
    rules: dict[str, OverrideRule] = {}
    for name, spec in raw.items():
        where = f"policy.overrides.{name}"
        spec_s = _Section(_as_mapping(spec, where), where)
        threshold = spec_s.float_("threshold", 1.0)
        action_name = spec_s.str_("action", Action.STOP.name.lower())
        spec_s.finish()
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(f"invalid value for '{where}.threshold': must be in [0, 1]")
        action = _OVERRIDE_ACTION_NAMES.get(action_name)
        if action is None:
            known = ", ".join(sorted(_OVERRIDE_ACTION_NAMES))
            raise ConfigError(
                f"unknown override action '{action_name}' in '{where}.action'; "
                f"valid actions: {known}"
            )
        rules[str(name)] = OverrideRule(threshold=threshold, action=action)
    return rules


def _parse_policy(data: Mapping[str, Any]) -> PolicySection:
    section = _Section(data, "policy")
    stop_threshold = section.float_("stop_threshold", 0.9)
    advisory = section.str_list("advisory_signals", [])
    weights = section.float_map("weights")
    tier_thresholds = section.float_map("tier_thresholds")
    escalate_ticks = section.int_("escalate_ticks", 1)
    deescalate_ticks = section.int_("deescalate_ticks", 1)
    overrides_raw = _as_mapping(section._pop("overrides", {}), "policy.overrides")
    section.finish()
    _reject_unknown_signals(advisory, "policy.advisory_signals")
    # A weight for a disabled detector (e.g. budget) is tolerated: this layer
    # does not see ``detectors.*.enabled``, and it is the same dead-but-valid
    # config as an advisory entry for an unwired detector.
    _reject_unknown_signals(weights, "policy.weights")
    _reject_unknown_tiers(tier_thresholds, "policy.tier_thresholds")
    overrides = _parse_override_rules(overrides_raw)
    _reject_unknown_signals(overrides, "policy.overrides")
    # An override on an observe-only detector is contradictory: advisory means
    # "never act", an override means "act immediately". Fail fast rather than
    # silently letting advisory win (which it does in the policy layer anyway).
    conflicting = sorted(set(overrides) & set(advisory))
    if conflicting:
        raise ConfigError(
            "signal(s) cannot be both advisory and overridden: "
            f"{', '.join(repr(name) for name in conflicting)}; "
            "an advisory signal is observe-only, an override forces an action"
        )
    return PolicySection(
        stop_threshold=stop_threshold,
        advisory_signals=frozenset(advisory),
        weights=weights,
        tier_thresholds=tier_thresholds,
        escalate_ticks=escalate_ticks,
        deescalate_ticks=deescalate_ticks,
        overrides=overrides,
    )


def _parse_pipeline(data: Mapping[str, Any]) -> PipelineSection:
    section = _Section(data, "pipeline")
    recent_history = section.int_("recent_history", 10)
    dry_run = section.bool_("dry_run", False)
    section.finish()
    return PipelineSection(recent_history=recent_history, dry_run=dry_run)


def _parse_pricing(data: Mapping[str, Any]) -> PricingSection:
    """Parse ``pricing``: per-model USD-per-1M-token rates (fail fast).

    Both rates are required per model: a table entry that silently defaulted the
    output rate to 0 would under-charge every completion and quietly weaken the
    budget brake. An empty model id is rejected too -- ``cost_of`` can never
    match it, so it would be a typo that prices nothing.
    """
    section = _Section(data, "pricing")
    replace_defaults = section.bool_("replace_defaults", False)
    models_raw = _as_mapping(section._pop("models", {}), "pricing.models")
    section.finish()

    models: dict[str, ModelPrice] = {}
    for model, spec in models_raw.items():
        where = f"pricing.models.{model}"
        if not model:
            raise ConfigError(
                "invalid model id in 'pricing.models': an empty id can never match a "
                "model, so it would price nothing"
            )
        spec_map = _as_mapping(spec, where)
        for key in ("input_usd_per_mtok", "output_usd_per_mtok"):
            if key not in spec_map:
                raise ConfigError(
                    f"missing required key '{where}.{key}': both the input and the output "
                    "rate must be given, in USD per 1M tokens"
                )
        spec_s = _Section(spec_map, where)
        input_rate = spec_s.nonnegative_float("input_usd_per_mtok", 0.0)
        output_rate = spec_s.nonnegative_float("output_usd_per_mtok", 0.0)
        spec_s.finish()
        models[str(model)] = ModelPrice(
            input_usd_per_mtok=input_rate, output_usd_per_mtok=output_rate
        )
    return PricingSection(models=models, replace_defaults=replace_defaults)


def _parse_store(data: Mapping[str, Any]) -> StoreSection:
    section = _Section(data, "store")
    path = section.str_("path", f"{CONFIG_DIR}/state.db")
    section.finish()
    return StoreSection(path=path)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "YAML config requires PyYAML. Install it with: pip install 'pathlayer[yaml]', "
            "or use a JSON config instead."
        ) from exc
    # safe_load only: never construct arbitrary Python objects from a config file.
    return yaml.safe_load(text)


def _parse_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if suffix in (".yaml", ".yml"):
        try:
            return _load_yaml(text)
        except ConfigError:
            raise
        except Exception as exc:  # PyYAML raises yaml.YAMLError; keep core yaml-free
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    raise ConfigError(
        f"unsupported config extension '{path.suffix}' for {path}; use .yaml, .yml or .json"
    )


def _discover(base: Path) -> Path | None:
    """Return the discovered config under ``base/.pathlayer``, or None.

    Looks for ``config.yaml`` / ``config.yml`` / ``config.json``. If more than
    one exists, raise rather than silently pick one.
    """
    directory = base / CONFIG_DIR
    found = [directory / name for name in _DISCOVERY_NAMES if (directory / name).exists()]
    if len(found) > 1:
        names = ", ".join(str(path) for path in found)
        raise ConfigError(
            f"multiple config files found ({names}); keep only one to avoid ambiguity"
        )
    return found[0] if found else None


def load_config(path: str | Path | None = None) -> PathlayerConfig:
    """Load a Pathlayer config, falling back to built-in defaults.

    - ``path`` given: load that file (must exist; bad path fails fast).
    - ``path`` omitted: discover ``./.pathlayer/config.{yaml,yml,json}`` (no
      priority order). If none exists, return defaults. If more than one exists,
      raise rather than guess.
    """
    if path is None:
        discovered = _discover(Path.cwd())
        if discovered is None:
            return PathlayerConfig()
        path = discovered
    else:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
    data = _parse_file(path)
    return PathlayerConfig.from_mapping(_as_mapping(data, "<root>"))


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build_pipeline(
    config: PathlayerConfig | None = None,
    *,
    logger: logging.Logger | None = None,
    state_key: str | None = None,
    store: StateStore | None = None,
) -> Pipeline:
    """Construct a :class:`~pathlayer.pipeline.Pipeline` from a config.

    Only ``enabled`` detectors are wired. ``no_progress`` gets the built-in
    :func:`~pathlayer.detectors.progress.coding_anchors`; its anchors stay
    code-level (the config tunes only the scalar thresholds). Per-tool masks are
    threaded into the canonicalization-based detectors (loop, retry).
    """
    if config is None:
        config = PathlayerConfig()
    tool_masks: ToolMasks | None = config.canonicalization.tool_masks or None

    detectors: list[Detector] = []
    spec = config.detectors
    # load_config validates structure (keys/types) but delegates value-range
    # checks to the detectors' and PolicyConfig's own __init__ (one source of
    # truth). Surface those as ConfigError so a caller catching ConfigError around
    # build_pipeline sees a malformed config consistently, instead of a raw
    # ValueError leaking through.
    try:
        if spec.loop.enabled:
            detectors.append(
                LoopDetector(
                    k=spec.loop.k,
                    window=spec.loop.window,
                    tool_masks=tool_masks,
                    ngram_min_hits=spec.loop.ngram_min_hits,
                    ngram_min_coverage=spec.loop.ngram_min_coverage,
                    ngram_max_confidence=spec.loop.ngram_max_confidence,
                    ngram_steepness=spec.loop.ngram_steepness,
                )
            )
        if spec.retry.enabled:
            detectors.append(
                RetryDetector(
                    threshold=spec.retry.threshold,
                    tool_masks=tool_masks,
                    window=spec.retry.window,
                    dominance_min_calls=spec.retry.dominance_min_calls,
                    dominance_ratio=spec.retry.dominance_ratio,
                    dominance_max_confidence=spec.retry.dominance_max_confidence,
                    min_failures=spec.retry.min_failures,
                )
            )
        if spec.budget.enabled:
            detectors.append(
                BudgetDetector(
                    max_cost_usd=spec.budget.max_cost_usd,
                    baseline_usd=spec.budget.baseline_usd,
                    default_cost_usd=spec.budget.default_cost_usd,
                    tool_costs_usd=spec.budget.tool_costs_usd,
                )
            )
        if spec.no_progress.enabled:
            detectors.append(
                NoProgressDetector(
                    coding_anchors(),
                    target_rate=spec.no_progress.target_rate,
                    threshold=spec.no_progress.threshold,
                    window=spec.no_progress.window,
                )
            )
        tier_thresholds = {
            _LADDER_TIER_NAMES[name]: threshold
            for name, threshold in config.policy.tier_thresholds.items()
        }
        overrides = dict(config.policy.overrides)
        # ARB-4: budget at ceiling = stop now. Auto-wire the budget->stop
        # override when the detector is enabled, unless the user already set a
        # budget rule (theirs wins) or made budget observe-only (advisory wins).
        if (
            spec.budget.enabled
            and BudgetDetector.name not in overrides
            and BudgetDetector.name not in config.policy.advisory_signals
        ):
            overrides[BudgetDetector.name] = OverrideRule(threshold=1.0, action=Action.STOP)
        suppressions: dict[str, SuppressionRule] = {}
        # LOOP-2 precision: loop<-progress suppression, auto-wired the same way
        # as the budget override above -- only when both loop and no_progress
        # (the source of the anchors) are enabled, and only if the caller has
        # not already made loop observe-only (advisory wins, nothing to guard).
        if (
            spec.loop.enabled
            and spec.no_progress.enabled
            and LoopDetector.name not in config.policy.advisory_signals
        ):
            suppressions[LoopDetector.name] = SuppressionRule(
                evidence=PROGRESS_EVIDENCE_KEY, min_confidence=1.0
            )
        policy = Policy(
            PolicyConfig(
                stop_threshold=config.policy.stop_threshold,
                advisory_signals=config.policy.advisory_signals,
                weights=config.policy.weights,
                tier_thresholds=tier_thresholds,
                escalate_ticks=config.policy.escalate_ticks,
                deescalate_ticks=config.policy.deescalate_ticks,
                overrides=overrides,
                suppressions=suppressions,
            )
        )
    except ValueError as exc:
        raise ConfigError(f"invalid configuration value: {exc}") from exc

    resolved_logger = logger or logging.getLogger(__name__)
    if not detectors:
        # A pipeline with no detectors can never act -- almost always a config
        # mistake (every detector disabled). Warn rather than fail: an empty,
        # observe-nothing pipeline is technically valid.
        resolved_logger.warning("build_pipeline: no detectors enabled; the pipeline will never act")
    # Audit #117 finding M-1: LOOP-2's n-gram fallback is an uncalibrated
    # heuristic (see pathlayer.detectors.loop module docstring) that is only
    # safe today because two independent guards hold: it can never solo-STOP
    # (capped below stop_threshold, after fusion weighting), and the
    # loop<-progress suppression catches the case where it would matter in
    # practice (traces/healthy/repeated_checkpoint_pair.json). Both warnings
    # are gated on `loop` actually being able to act at all: an advisory
    # `loop` (Policy._actionable excludes it unconditionally) can never STOP
    # or be suppressed either, so nothing here would be true for it. Warn --
    # don't fail -- if a guard is weakened, so a deliberate override is still
    # possible but never silent.
    if spec.loop.enabled and LoopDetector.name not in config.policy.advisory_signals:
        # Policy.fuse() combines a lone actionable signal as
        # `weight * confidence` (weighted noisy-OR): a down-weighted loop
        # needs a higher raw confidence to reach stop_threshold than the cap
        # alone suggests, so the weight is folded into the comparison to
        # avoid warning on setups that are actually safe.
        loop_weight = config.policy.weights.get(LoopDetector.name, 1.0)
        if spec.loop.ngram_max_confidence * loop_weight >= config.policy.stop_threshold:
            resolved_logger.warning(
                "build_pipeline: detectors.loop.ngram_max_confidence * "
                "policy.weights.loop (%s * %s = %s) >= policy.stop_threshold (%s): "
                "the LOOP-2 n-gram fallback, an uncalibrated heuristic (see issue "
                "#19), can now solo-STOP an agent on its own -- see audit #117 "
                "finding M-1",
                spec.loop.ngram_max_confidence,
                loop_weight,
                spec.loop.ngram_max_confidence * loop_weight,
                config.policy.stop_threshold,
            )
        if not spec.no_progress.enabled:
            resolved_logger.warning(
                "build_pipeline: detectors.loop is enabled but detectors.no_progress "
                "is disabled: the loop<-progress suppression that protects against "
                "LOOP-2 false positives on routine checkpoint patterns (see "
                "traces/healthy/repeated_checkpoint_pair.json) will not be wired -- "
                "see audit #117 finding M-1"
            )
    # ACT-1 anti-noise guard: an enabled WARN tier with no hysteresis re-fires on
    # every crossing of the warn threshold, and an ACT-1 actuator turns each
    # re-fire into a fresh context injection. Warn -- don't fail -- so a
    # deliberate no-debounce setup stays possible but never silent. Tied to the
    # injection mechanism, not a tuned threshold (no data needed).
    if (
        Action.WARN in tier_thresholds
        and config.policy.escalate_ticks == 1
        and config.policy.deescalate_ticks == 1
    ):
        resolved_logger.warning(
            "build_pipeline: the WARN tier is enabled with no hysteresis "
            "(policy.escalate_ticks == policy.deescalate_ticks == 1): a fused score "
            "flapping across the warn threshold re-fires WARN on every crossing, "
            "which an ACT-1 actuator turns into repeated context injections. "
            "Set escalate_ticks / deescalate_ticks > 1 to debounce."
        )
    # ACT-2 anti-noise guard: same rationale as WARN above. An enabled SUMMARIZE
    # tier with no hysteresis re-arms the strict gate on every crossing of the
    # summarize threshold, forcing the agent to re-summarize on a merely flapping
    # score. Warn -- don't fail -- so a deliberate no-debounce setup stays
    # possible but never silent. Tied to the gate mechanism, not a tuned
    # threshold (no data needed).
    if (
        Action.SUMMARIZE in tier_thresholds
        and config.policy.escalate_ticks == 1
        and config.policy.deescalate_ticks == 1
    ):
        resolved_logger.warning(
            "build_pipeline: the SUMMARIZE tier is enabled with no hysteresis "
            "(policy.escalate_ticks == policy.deescalate_ticks == 1): a fused score "
            "flapping across the summarize threshold re-arms the ACT-2 strict gate on "
            "every crossing, forcing the agent to re-summarize repeatedly. "
            "Set escalate_ticks / deescalate_ticks > 1 to debounce."
        )
    # BUD-3 anti-surprise guard: the auto-wired (or user) budget override maps a
    # confidence >= its threshold straight to an action, bypassing hysteresis. A
    # single unpriced eligible event contributes at most
    # max(default_cost_usd, max(tool_costs_usd)) USD, i.e. a confidence of that
    # over max_cost_usd. When that alone meets the override threshold, the very
    # first unpriced event triggers the override (typically STOP) -- an estimate
    # calibration mistake, not a real runaway. Warn -- don't fail -- so a
    # deliberate aggressive estimate stays possible but never silent. Gated on
    # budget actually being able to act (enabled, not advisory, override wired).
    budget_rule = overrides.get(BudgetDetector.name)
    if (
        spec.budget.enabled
        and BudgetDetector.name not in config.policy.advisory_signals
        and budget_rule is not None
    ):
        max_single_estimate = max(
            [spec.budget.default_cost_usd, *spec.budget.tool_costs_usd.values()]
        )
        # A zero estimate never contributes, so the "first unpriced event" framing
        # would be misleading -- only warn when an estimate can actually fire.
        if max_single_estimate > 0.0 and (
            max_single_estimate >= budget_rule.threshold * spec.budget.max_cost_usd
        ):
            resolved_logger.warning(
                "build_pipeline: a single unpriced event's budget estimate "
                "(max(default_cost_usd, tool_costs_usd) = %s) is enough to reach the "
                "budget override threshold (%s * max_cost_usd %s = %s): the first "
                "unpriced eligible event will immediately trigger %s, bypassing "
                "hysteresis. Lower the estimate(s) or raise max_cost_usd if this is "
                "not intended.",
                max_single_estimate,
                budget_rule.threshold,
                spec.budget.max_cost_usd,
                budget_rule.threshold * spec.budget.max_cost_usd,
                budget_rule.action.name,
            )
    # Same gating as the suppression rule above: the anchors only matter paired
    # with a rule that consumes them, so skip building them otherwise.
    progress_anchors = loop_suppression_anchors() if suppressions else None
    pipeline = Pipeline(
        detectors,
        policy,
        recent_history=config.pipeline.recent_history,
        logger=logger,
        dry_run=config.pipeline.dry_run,
        progress_anchors=progress_anchors,
    )
    if state_key is None:
        if store is not None:
            raise ValueError("store requires a state_key")
        return pipeline

    initial_pipeline: Pipeline | None = pipeline

    def pipeline_factory() -> Pipeline:
        nonlocal initial_pipeline
        if initial_pipeline is not None:
            resolved = initial_pipeline
            initial_pipeline = None
            return resolved
        return build_pipeline(config, logger=logger)

    state_store = store or SQLiteStateStore(config.store.path)
    owns_store = store is None

    def resolve_cost(event: ToolEvent) -> BudgetState:
        return budget_delta(
            event,
            tool_costs_usd=spec.budget.tool_costs_usd if spec.budget.enabled else None,
            default_cost_usd=spec.budget.default_cost_usd if spec.budget.enabled else 0.0,
        )

    # PersistentPipeline intentionally mirrors Pipeline's public surface. Keep
    # the historical return annotation so existing typed callers do not need a
    # migration merely to opt into persistence.
    return cast(
        Pipeline,
        PersistentPipeline(
            pipeline_factory,
            state_store,
            state_key,
            configuration_fingerprint(config),
            cost_resolver=resolve_cost,
            owns_store=owns_store,
            logger=resolved_logger,
        ),
    )
