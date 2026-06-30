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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathlayer.detectors.base import Detector
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.canonical import ToolMask, ToolMasks
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import coding_anchors
from pathlayer.detectors.retry import RetryDetector
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Policy
from pathlayer.policies.base import PolicyConfig

__all__ = [
    "ConfigError",
    "PathlayerConfig",
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
    {LoopDetector.name, RetryDetector.name, BudgetDetector.name, NoProgressDetector.name}
)


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
                    f"invalid value for '{where}.{subkey}': weight must be in [0, 1], got {number}"
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


@dataclass(slots=True)
class RetryConfig:
    enabled: bool = True
    threshold: int = 3


@dataclass(slots=True)
class BudgetConfig:
    # Disabled by default: enabling at the $5 default would re-introduce the
    # certified false positive on a legitimately expensive run ($6.40). Opt in
    # with a ceiling chosen for the workload.
    enabled: bool = False
    max_cost_usd: float = 5.0
    baseline_usd: float = 0.0


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


@dataclass(slots=True)
class PipelineSection:
    recent_history: int = 10
    # Dry-run (OBS-3): detectors and arbitration run and are logged, but no
    # action is emitted and the run never halts -- the observe-only mode for
    # tuning before trusting enforcement.
    dry_run: bool = False


@dataclass(slots=True)
class StoreSection:
    # Carried for INF-1 (cross-run SQLite state); inert today -- nothing reads or
    # creates it yet.
    path: str = f"{CONFIG_DIR}/state.db"


@dataclass(slots=True)
class PathlayerConfig:
    detectors: DetectorsConfig = field(default_factory=DetectorsConfig)
    canonicalization: CanonicalizationConfig = field(default_factory=CanonicalizationConfig)
    policy: PolicySection = field(default_factory=PolicySection)
    pipeline: PipelineSection = field(default_factory=PipelineSection)
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
        store = _parse_store(_as_mapping(root._pop("store", {}), "store"))
        root.finish()
        return cls(
            detectors=detectors,
            canonicalization=canonicalization,
            policy=policy,
            pipeline=pipeline,
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
    )
    loop_s.finish()

    retry_s = _Section(retry_raw, "detectors.retry")
    retry = RetryConfig(
        enabled=retry_s.bool_("enabled", True),
        threshold=retry_s.int_("threshold", 3),
    )
    retry_s.finish()

    budget_s = _Section(budget_raw, "detectors.budget")
    budget = BudgetConfig(
        enabled=budget_s.bool_("enabled", False),
        max_cost_usd=budget_s.float_("max_cost_usd", 5.0),
        baseline_usd=budget_s.float_("baseline_usd", 0.0),
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


def _parse_policy(data: Mapping[str, Any]) -> PolicySection:
    section = _Section(data, "policy")
    stop_threshold = section.float_("stop_threshold", 0.9)
    advisory = section.str_list("advisory_signals", [])
    weights = section.float_map("weights")
    section.finish()
    _reject_unknown_signals(advisory, "policy.advisory_signals")
    # A weight for a disabled detector (e.g. budget) is tolerated: this layer
    # does not see ``detectors.*.enabled``, and it is the same dead-but-valid
    # config as an advisory entry for an unwired detector.
    _reject_unknown_signals(weights, "policy.weights")
    return PolicySection(
        stop_threshold=stop_threshold,
        advisory_signals=frozenset(advisory),
        weights=weights,
    )


def _parse_pipeline(data: Mapping[str, Any]) -> PipelineSection:
    section = _Section(data, "pipeline")
    recent_history = section.int_("recent_history", 10)
    dry_run = section.bool_("dry_run", False)
    section.finish()
    return PipelineSection(recent_history=recent_history, dry_run=dry_run)


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
                LoopDetector(k=spec.loop.k, window=spec.loop.window, tool_masks=tool_masks)
            )
        if spec.retry.enabled:
            detectors.append(RetryDetector(threshold=spec.retry.threshold, tool_masks=tool_masks))
        if spec.budget.enabled:
            detectors.append(
                BudgetDetector(
                    max_cost_usd=spec.budget.max_cost_usd, baseline_usd=spec.budget.baseline_usd
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
        policy = Policy(
            PolicyConfig(
                stop_threshold=config.policy.stop_threshold,
                advisory_signals=config.policy.advisory_signals,
                weights=config.policy.weights,
            )
        )
    except ValueError as exc:
        raise ConfigError(f"invalid configuration value: {exc}") from exc

    if not detectors:
        # A pipeline with no detectors can never act -- almost always a config
        # mistake (every detector disabled). Warn rather than fail: an empty,
        # observe-nothing pipeline is technically valid.
        (logger or logging.getLogger(__name__)).warning(
            "build_pipeline: no detectors enabled; the pipeline will never act"
        )
    return Pipeline(
        detectors,
        policy,
        recent_history=config.pipeline.recent_history,
        logger=logger,
        dry_run=config.pipeline.dry_run,
    )
