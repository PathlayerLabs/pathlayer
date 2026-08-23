"""INF-4: configuration loading, validation, and pipeline building.

Covers the zero-config defaults, JSON + YAML loading, fail-fast validation
(unknown keys, wrong types, ambiguous discovery), the optional-PyYAML path, and
``build_pipeline`` wiring -- including the per-tool canonicalization masks
(CAN-2) exposed through config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

import pathlayer
from pathlayer.config import (
    ConfigError,
    PathlayerConfig,
    build_pipeline,
    load_config,
)
from pathlayer.detectors.budget import BudgetDetector
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.resource import DURATION_REASON_PREFIX, ResourceDetector
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import PROGRESS_EVIDENCE_KEY
from pathlayer.policies import Action, OverrideRule
from pathlayer.pricing import DEFAULT_PRICES, ModelPrice, cost_of

EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "examples" / "pathlayer.config.yaml"


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_public_api_exports() -> None:
    # The package re-exports the config surface, including the error type users
    # must catch around load_config / build_pipeline.
    for name in ("PathlayerConfig", "load_config", "build_pipeline", "ConfigError"):
        assert name in pathlayer.__all__
        assert hasattr(pathlayer, name)


def test_no_file_returns_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_config() == PathlayerConfig()


def test_defaults_match_current_hardcoded_values() -> None:
    config = PathlayerConfig()
    assert (config.detectors.loop.k, config.detectors.loop.window) == (2, 256)
    assert config.detectors.retry.threshold == 3
    assert config.detectors.budget.enabled is False
    assert config.detectors.budget.max_cost_usd == 5.0
    assert config.detectors.budget.default_cost_usd == 0.0
    assert config.detectors.budget.tool_costs_usd == {}
    assert config.detectors.no_progress.window == 64
    assert config.policy.stop_threshold == 0.9
    assert config.policy.advisory_signals == frozenset()
    assert config.policy.weights == {}
    assert config.pipeline.recent_history == 10
    assert config.pipeline.dry_run is False
    assert config.store.path == ".pathlayer/state.db"
    assert config.canonicalization.tool_masks == {}


def test_empty_and_null_sections_fall_back_to_defaults() -> None:
    data: dict[str, Any] = {"detectors": {"loop": None}, "policy": {}, "canonicalization": None}
    assert PathlayerConfig.from_mapping(data) == PathlayerConfig()


# --------------------------------------------------------------------------- #
# JSON / YAML loading
# --------------------------------------------------------------------------- #


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_json_overrides(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        json.dumps({"detectors": {"loop": {"k": 3}}, "policy": {"stop_threshold": 0.5}}),
    )
    config = load_config(path)
    assert config.detectors.loop.k == 3
    assert config.policy.stop_threshold == 0.5
    # Untouched values keep their defaults.
    assert config.detectors.loop.window == 256


def test_load_yaml_overrides(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.yaml",
        "detectors:\n  retry:\n    threshold: 5\npipeline:\n  recent_history: 3\n",
    )
    config = load_config(path)
    assert config.detectors.retry.threshold == 5
    assert config.pipeline.recent_history == 3


def test_load_json_parses_per_tool_cost_estimates_above_one(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        json.dumps(
            {
                "detectors": {
                    "budget": {
                        "default_cost_usd": 0.25,
                        "tool_costs_usd": {"read_file": 0.01, "train_model": 3},
                    }
                }
            }
        ),
    )
    budget = load_config(path).detectors.budget

    assert budget.default_cost_usd == 0.25
    assert budget.tool_costs_usd == {"read_file": 0.01, "train_model": 3.0}


def test_load_yaml_parses_per_tool_cost_estimates(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.yaml",
        "detectors:\n"
        "  budget:\n"
        "    default_cost_usd: 0.2\n"
        "    tool_costs_usd:\n"
        "      train_model: 4.5\n",
    )
    budget = load_config(path).detectors.budget

    assert budget.default_cost_usd == 0.2
    assert budget.tool_costs_usd == {"train_model": 4.5}


def test_int_field_accepts_int_float_field_accepts_int(tmp_path: Path) -> None:
    # max_cost_usd given as an int must coerce to float.
    path = _write(
        tmp_path / "config.json", json.dumps({"detectors": {"budget": {"max_cost_usd": 7}}})
    )
    config = load_config(path)
    assert config.detectors.budget.max_cost_usd == 7.0
    assert isinstance(config.detectors.budget.max_cost_usd, float)


def test_explicit_missing_path_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"not found"):
        load_config(tmp_path / "nope.yaml")


def test_unsupported_extension(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", "x = 1")
    with pytest.raises(ConfigError, match=r"unsupported config extension"):
        load_config(path)


def test_invalid_json(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", "{not json}")
    with pytest.raises(ConfigError, match=r"invalid JSON"):
        load_config(path)


def test_invalid_yaml(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.yaml", "a:\n  - b\n  c: d\n")
    with pytest.raises(ConfigError, match=r"invalid YAML"):
        load_config(path)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_discovers_yaml_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / ".pathlayer" / "config.yaml", "detectors:\n  loop:\n    k: 4\n")
    monkeypatch.chdir(tmp_path)
    assert load_config().detectors.loop.k == 4


def test_discovers_json_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / ".pathlayer" / "config.json", json.dumps({"detectors": {"loop": {"k": 6}}}))
    monkeypatch.chdir(tmp_path)
    assert load_config().detectors.loop.k == 6


def test_discovers_yml_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / ".pathlayer" / "config.yml", "detectors:\n  loop:\n    k: 5\n")
    monkeypatch.chdir(tmp_path)
    assert load_config().detectors.loop.k == 5


def test_ambiguous_discovery_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / ".pathlayer" / "config.yaml", "")
    _write(tmp_path / ".pathlayer" / "config.json", "{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match=r"multiple config files found"):
        load_config()


# --------------------------------------------------------------------------- #
# Optional PyYAML
# --------------------------------------------------------------------------- #


def test_yaml_without_pyyaml_gives_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "config.yaml", "detectors: {}\n")
    # Simulate PyYAML being absent: importing 'yaml' then raises ImportError.
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ConfigError, match=r"requires PyYAML"):
        load_config(path)


# --------------------------------------------------------------------------- #
# Fail-fast validation
# --------------------------------------------------------------------------- #


def test_unknown_root_key() -> None:
    with pytest.raises(ConfigError, match=r"unknown key.*'detctors'"):
        PathlayerConfig.from_mapping({"detctors": {}})


def test_unknown_nested_key() -> None:
    with pytest.raises(ConfigError, match=r"detectors.loop.*'kk'"):
        PathlayerConfig.from_mapping({"detectors": {"loop": {"kk": 2}}})


def test_section_must_be_mapping() -> None:
    with pytest.raises(ConfigError, match=r"must be a mapping"):
        PathlayerConfig.from_mapping({"detectors": 5})


def test_bool_field_type_error() -> None:
    with pytest.raises(ConfigError, match=r"loop.enabled.*expected bool"):
        PathlayerConfig.from_mapping({"detectors": {"loop": {"enabled": "yes"}}})


def test_int_field_rejects_float() -> None:
    with pytest.raises(ConfigError, match=r"loop.k.*expected int"):
        PathlayerConfig.from_mapping({"detectors": {"loop": {"k": 2.0}}})


def test_int_field_rejects_bool() -> None:
    with pytest.raises(ConfigError, match=r"loop.k.*expected int"):
        PathlayerConfig.from_mapping({"detectors": {"loop": {"k": True}}})


def test_float_field_rejects_string() -> None:
    with pytest.raises(ConfigError, match=r"max_cost_usd.*expected number"):
        PathlayerConfig.from_mapping({"detectors": {"budget": {"max_cost_usd": "x"}}})


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_default_tool_cost_must_be_finite_and_nonnegative(value: float) -> None:
    with pytest.raises(ConfigError, match=r"default_cost_usd.*finite and >= 0"):
        PathlayerConfig.from_mapping({"detectors": {"budget": {"default_cost_usd": value}}})


@pytest.mark.parametrize("value", [True, "1.0"])
def test_default_tool_cost_rejects_non_number(value: object) -> None:
    with pytest.raises(ConfigError, match=r"default_cost_usd.*expected number"):
        PathlayerConfig.from_mapping({"detectors": {"budget": {"default_cost_usd": value}}})


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_tool_cost_mapping_values_must_be_finite_and_nonnegative(value: float) -> None:
    with pytest.raises(ConfigError, match=r"tool_costs_usd.call.*finite and >= 0"):
        PathlayerConfig.from_mapping({"detectors": {"budget": {"tool_costs_usd": {"call": value}}}})


@pytest.mark.parametrize("value", [True, "1.0"])
def test_tool_cost_mapping_rejects_non_number(value: object) -> None:
    with pytest.raises(ConfigError, match=r"tool_costs_usd.call.*expected number"):
        PathlayerConfig.from_mapping({"detectors": {"budget": {"tool_costs_usd": {"call": value}}}})


def test_tool_costs_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match=r"tool_costs_usd.*must be a mapping"):
        PathlayerConfig.from_mapping({"detectors": {"budget": {"tool_costs_usd": ["call"]}}})


def test_str_field_type_error() -> None:
    with pytest.raises(ConfigError, match=r"store.path.*expected string"):
        PathlayerConfig.from_mapping({"store": {"path": 5}})


def test_str_list_type_error() -> None:
    with pytest.raises(ConfigError, match=r"advisory_signals.*expected list of strings"):
        PathlayerConfig.from_mapping({"policy": {"advisory_signals": "loop"}})


def test_unknown_advisory_signal() -> None:
    with pytest.raises(ConfigError, match=r"unknown signal.*'no_progres'"):
        PathlayerConfig.from_mapping({"policy": {"advisory_signals": ["no_progres"]}})


def test_known_advisory_signal_accepted() -> None:
    config = PathlayerConfig.from_mapping({"policy": {"advisory_signals": ["no_progress"]}})
    assert config.policy.advisory_signals == frozenset({"no_progress"})


def test_resource_signal_accepted_when_detector_disabled_by_default() -> None:
    # ResourceDetector (RES-1) is registered in _KNOWN_SIGNALS so a config can
    # reference "resource" (e.g. in advisory_signals) without a spurious
    # "unknown signal" ConfigError. detectors.resource.enabled defaults to
    # False, so this alone does not wire ResourceDetector into the pipeline.
    config = PathlayerConfig.from_mapping({"policy": {"advisory_signals": ["resource"]}})
    assert config.policy.advisory_signals == frozenset({"resource"})


# --------------------------------------------------------------------------- #
# policy.weights (ARB-1)
# --------------------------------------------------------------------------- #


def test_weights_parsed_and_coerced() -> None:
    config = PathlayerConfig.from_mapping({"policy": {"weights": {"loop": 0.5, "retry": 1}}})
    assert config.policy.weights == {"loop": 0.5, "retry": 1.0}
    assert isinstance(config.policy.weights["retry"], float)


def test_weights_default_empty() -> None:
    assert PathlayerConfig.from_mapping({"policy": {}}).policy.weights == {}


def test_weights_unknown_signal_rejected() -> None:
    with pytest.raises(ConfigError, match=r"unknown signal.*'no_progres'.*policy.weights|weights"):
        PathlayerConfig.from_mapping({"policy": {"weights": {"no_progres": 0.5}}})


def test_weights_for_disabled_budget_tolerated() -> None:
    # budget is a known signal even though it is disabled by default: a weight
    # for it is dead-but-valid config, not an error.
    config = PathlayerConfig.from_mapping({"policy": {"weights": {"budget": 0.5}}})
    assert config.policy.weights == {"budget": 0.5}


def test_weight_out_of_range_rejected() -> None:
    with pytest.raises(ConfigError, match=r"policy.weights.loop.*\[0, 1\]"):
        PathlayerConfig.from_mapping({"policy": {"weights": {"loop": 1.5}}})


def test_weight_bool_rejected() -> None:
    with pytest.raises(ConfigError, match=r"policy.weights.loop.*expected number"):
        PathlayerConfig.from_mapping({"policy": {"weights": {"loop": True}}})


def test_weight_string_rejected() -> None:
    with pytest.raises(ConfigError, match=r"policy.weights.loop.*expected number"):
        PathlayerConfig.from_mapping({"policy": {"weights": {"loop": "0.5"}}})


def test_weights_not_a_mapping_rejected() -> None:
    with pytest.raises(ConfigError, match=r"policy.weights.*must be a mapping"):
        PathlayerConfig.from_mapping({"policy": {"weights": "loop"}})


def test_dry_run_parsed_and_defaults_false() -> None:
    assert PathlayerConfig.from_mapping({}).pipeline.dry_run is False
    config = PathlayerConfig.from_mapping({"pipeline": {"dry_run": True}})
    assert config.pipeline.dry_run is True


def test_dry_run_type_error() -> None:
    with pytest.raises(ConfigError, match=r"pipeline.dry_run.*expected bool"):
        PathlayerConfig.from_mapping({"pipeline": {"dry_run": "yes"}})


# --------------------------------------------------------------------------- #
# build_pipeline
# --------------------------------------------------------------------------- #


def test_build_pipeline_defaults_wires_three_detectors() -> None:
    pipeline = build_pipeline()
    detectors = [type(d) for d in pipeline._detectors]
    assert detectors == [LoopDetector, RetryDetector, NoProgressDetector]


def test_build_pipeline_all_enabled_includes_budget() -> None:
    config = PathlayerConfig.from_mapping({"detectors": {"budget": {"enabled": True}}})
    pipeline = build_pipeline(config)
    assert BudgetDetector in [type(d) for d in pipeline._detectors]


def test_build_pipeline_threads_budget_cost_estimates() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "budget": {
                    "enabled": True,
                    "default_cost_usd": 0.25,
                    "tool_costs_usd": {"train_model": 3.0},
                }
            }
        }
    )
    pipeline = build_pipeline(config)
    budget = next(d for d in pipeline._detectors if isinstance(d, BudgetDetector))

    assert budget._default_cost_usd == 0.25
    assert budget._tool_costs_usd == {"train_model": 3.0}


def test_estimated_budget_override_bypasses_hysteresis_end_to_end() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "loop": {"enabled": False},
                "retry": {"enabled": False},
                "no_progress": {"enabled": False},
                "budget": {
                    "enabled": True,
                    "max_cost_usd": 5.0,
                    "tool_costs_usd": {"read_file": 0.1, "train_model": 5.0},
                },
            },
            "policy": {"escalate_ticks": 3},
        }
    )
    cheap = build_pipeline(config)
    expensive = build_pipeline(config)

    assert cheap.run([ToolEvent("read_file"), ToolEvent("read_file")]) == [
        Action.NONE,
        Action.NONE,
    ]
    assert expensive.process(ToolEvent("train_model")) is Action.STOP


def test_build_pipeline_disabled_detectors_excluded() -> None:
    config = PathlayerConfig.from_mapping(
        {"detectors": {"loop": {"enabled": False}, "retry": {"enabled": False}}}
    )
    types = [type(d) for d in build_pipeline(config)._detectors]
    assert LoopDetector not in types
    assert RetryDetector not in types
    # no_progress stays on by default.
    assert NoProgressDetector in types


def test_build_pipeline_warns_when_no_detector_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "loop": {"enabled": False},
                "retry": {"enabled": False},
                "no_progress": {"enabled": False},
            }
        }
    )
    with caplog.at_level("WARNING"):
        pipeline = build_pipeline(config)
    assert pipeline._detectors == []
    assert "no detectors enabled" in caplog.text


def test_build_pipeline_passes_thresholds() -> None:
    config = PathlayerConfig.from_mapping(
        {"detectors": {"loop": {"k": 3, "window": 10}}, "pipeline": {"recent_history": 2}}
    )
    pipeline = build_pipeline(config)
    loop = next(d for d in pipeline._detectors if isinstance(d, LoopDetector))
    assert loop._k == 3
    assert loop._window == 10
    assert pipeline._recent_history == 2


def test_build_pipeline_threads_retry_dominance_params() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "retry": {
                    "window": 20,
                    "dominance_min_calls": 6,
                    "dominance_ratio": 0.8,
                    "dominance_max_confidence": 0.7,
                    "min_failures": 3,
                }
            }
        }
    )
    pipeline = build_pipeline(config)
    retry = next(d for d in pipeline._detectors if isinstance(d, RetryDetector))
    assert retry._window == 20
    assert retry._dominance_min_calls == 6
    assert retry._dominance_ratio == 0.8
    assert retry._dominance_max_confidence == 0.7
    assert retry._min_failures == 3


def test_build_pipeline_wraps_retry_range_error_as_config_error() -> None:
    # A window smaller than dominance_min_calls is a range error from the
    # detector; build_pipeline must surface it as ConfigError like the others.
    config = PathlayerConfig.from_mapping(
        {"detectors": {"retry": {"window": 3, "dominance_min_calls": 5}}}
    )
    with pytest.raises(ConfigError, match=r"invalid configuration value.*window must be"):
        build_pipeline(config)


def test_build_pipeline_threads_loop_ngram_params() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "loop": {
                    "ngram_min_hits": 4,
                    "ngram_min_coverage": 0.4,
                    "ngram_max_confidence": 0.7,
                    "ngram_steepness": 5.0,
                }
            }
        }
    )
    pipeline = build_pipeline(config)
    loop = next(d for d in pipeline._detectors if isinstance(d, LoopDetector))
    assert loop._ngram_min_hits == 4
    assert loop._ngram_min_coverage == 0.4
    assert loop._ngram_max_confidence == 0.7
    assert loop._ngram_steepness == 5.0


def test_build_pipeline_wraps_loop_ngram_range_error_as_config_error() -> None:
    config = PathlayerConfig.from_mapping({"detectors": {"loop": {"ngram_min_hits": 1}}})
    with pytest.raises(ConfigError, match=r"invalid configuration value.*ngram_min_hits"):
        build_pipeline(config)


def test_build_pipeline_wraps_loop_steepness_range_error_as_config_error() -> None:
    config = PathlayerConfig.from_mapping({"detectors": {"loop": {"ngram_steepness": 0.0}}})
    with pytest.raises(ConfigError, match=r"invalid configuration value.*ngram_steepness"):
        build_pipeline(config)


def test_build_pipeline_warns_when_ngram_cap_meets_stop_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Audit #117 finding M-1: raising the LOOP-2 cap to/above stop_threshold
    # removes one of the two guards that keep the uncalibrated n-gram fallback
    # from ever solo-stopping on its own.
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {"loop": {"ngram_max_confidence": 0.9}},
            "policy": {"stop_threshold": 0.9},
        }
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "ngram_max_confidence" in caplog.text
    assert "M-1" in caplog.text


def test_build_pipeline_does_not_warn_when_ngram_cap_stays_below_stop_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        build_pipeline()  # built-in defaults: 0.85 < 0.9
    assert "ngram_max_confidence" not in caplog.text


def test_build_pipeline_does_not_warn_when_loop_is_advisory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Copilot review on PR #119: an advisory `loop` can never act at all
    # (Policy._actionable excludes it unconditionally), so neither M-1 guard
    # is even relevant -- warning here would be actively misleading.
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {"loop": {"ngram_max_confidence": 0.9}, "no_progress": {"enabled": False}},
            "policy": {"stop_threshold": 0.9, "advisory_signals": ["loop"]},
        }
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "M-1" not in caplog.text


def test_build_pipeline_does_not_warn_when_loop_weight_keeps_cap_below_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Copilot review on PR #119: Policy.fuse() combines a lone actionable
    # signal as weight * confidence, so a down-weighted loop needs a higher
    # raw confidence to reach stop_threshold than the raw cap suggests.
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {"loop": {"ngram_max_confidence": 0.9}},
            "policy": {"stop_threshold": 0.9, "weights": {"loop": 0.5}},
        }
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "ngram_max_confidence" not in caplog.text


def test_build_pipeline_warns_when_warn_tier_enabled_without_hysteresis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # ACT-1: an enabled WARN tier with escalate_ticks == deescalate_ticks == 1
    # re-fires on every threshold crossing, which an ACT-1 actuator turns into
    # repeated context injections. Warn (don't fail) so the user can debounce.
    config = PathlayerConfig.from_mapping({"policy": {"tier_thresholds": {"warn": 0.4}}})
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "WARN tier is enabled with no hysteresis" in caplog.text


def test_build_pipeline_does_not_warn_when_warn_tier_has_escalate_hysteresis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = PathlayerConfig.from_mapping(
        {"policy": {"tier_thresholds": {"warn": 0.4}, "escalate_ticks": 2}}
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "WARN tier is enabled with no hysteresis" not in caplog.text


def test_build_pipeline_does_not_warn_when_warn_tier_has_deescalate_hysteresis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The guard is `escalate_ticks == 1 AND deescalate_ticks == 1`; debouncing on
    # the de-escalation side alone is enough to silence it (covers that clause).
    config = PathlayerConfig.from_mapping(
        {"policy": {"tier_thresholds": {"warn": 0.4}, "deescalate_ticks": 2}}
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "WARN tier is enabled with no hysteresis" not in caplog.text


def test_build_pipeline_does_not_warn_about_warn_tier_when_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Default config: no ladder tiers, so the WARN-hysteresis guard is silent.
    with caplog.at_level("WARNING"):
        build_pipeline()
    assert "WARN tier is enabled with no hysteresis" not in caplog.text


def test_build_pipeline_warns_when_summarize_tier_enabled_without_hysteresis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # ACT-2: same anti-noise guard as WARN. An enabled SUMMARIZE tier with no
    # hysteresis re-arms the strict gate on every threshold crossing.
    config = PathlayerConfig.from_mapping({"policy": {"tier_thresholds": {"summarize": 0.4}}})
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "SUMMARIZE tier is enabled with no hysteresis" in caplog.text


def test_build_pipeline_does_not_warn_when_summarize_tier_has_hysteresis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = PathlayerConfig.from_mapping(
        {"policy": {"tier_thresholds": {"summarize": 0.4}, "escalate_ticks": 2}}
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "SUMMARIZE tier is enabled with no hysteresis" not in caplog.text


def test_build_pipeline_warns_when_single_estimate_reaches_budget_override(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # BUD-3: a per-tool estimate as large as the ceiling means the first unpriced
    # call of that tool immediately trips the auto-wired budget->stop override,
    # bypassing hysteresis -- almost always a calibration mistake.
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "budget": {"enabled": True, "max_cost_usd": 5.0, "tool_costs_usd": {"train": 5.0}}
            }
        }
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "budget estimate" in caplog.text
    assert "STOP" in caplog.text


def test_build_pipeline_does_not_warn_when_estimate_stays_below_override(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "budget": {"enabled": True, "max_cost_usd": 5.0, "tool_costs_usd": {"train": 0.5}}
            }
        }
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "budget estimate" not in caplog.text


def test_build_pipeline_does_not_warn_about_budget_estimate_when_advisory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Advisory budget can never STOP, so no override is wired and the guard is
    # irrelevant -- warning here would be misleading.
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "budget": {"enabled": True, "max_cost_usd": 5.0, "tool_costs_usd": {"train": 5.0}}
            },
            "policy": {"advisory_signals": ["budget"]},
        }
    )
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "budget estimate" not in caplog.text


def test_build_pipeline_warns_when_loop_enabled_without_no_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Audit #117 finding M-1: without no_progress, the loop<-progress
    # suppression that protects traces/healthy/repeated_checkpoint_pair.json
    # is never wired.
    config = PathlayerConfig.from_mapping({"detectors": {"no_progress": {"enabled": False}}})
    with caplog.at_level("WARNING"):
        build_pipeline(config)
    assert "no_progress" in caplog.text
    assert "M-1" in caplog.text


def test_build_pipeline_does_not_warn_when_loop_and_no_progress_both_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        build_pipeline()  # built-in defaults: both enabled
    assert "loop<-progress" not in caplog.text


def test_build_pipeline_threads_dry_run() -> None:
    assert build_pipeline().dry_run is False
    config = PathlayerConfig.from_mapping({"pipeline": {"dry_run": True}})
    assert build_pipeline(config).dry_run is True


def test_build_pipeline_threads_weights() -> None:
    config = PathlayerConfig.from_mapping({"policy": {"weights": {"no_progress": 0.0}}})
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.weight_for("no_progress") == 0.0
    # Unconfigured detectors keep full authority.
    assert pipeline._policy.config.weight_for("loop") == 1.0


def test_build_pipeline_rejects_out_of_range_threshold() -> None:
    # Range checks are delegated to the detector's own __init__ (one source), but
    # build_pipeline re-wraps the ValueError as ConfigError (M-1) so a caller
    # catching ConfigError sees malformed config consistently.
    config = PathlayerConfig.from_mapping({"detectors": {"loop": {"k": 1}}})
    with pytest.raises(ConfigError, match=r"invalid configuration value.*k must be >= 2"):
        build_pipeline(config)


def test_build_pipeline_rejects_bad_stop_threshold() -> None:
    config = PathlayerConfig.from_mapping({"policy": {"stop_threshold": 2.0}})
    with pytest.raises(ConfigError, match=r"invalid configuration value.*stop_threshold"):
        build_pipeline(config)


def test_build_pipeline_wraps_no_progress_range_error_as_config_error() -> None:
    # A no_progress window too small for the threshold/target_rate is a range
    # error from the detector; build_pipeline must surface it as ConfigError too.
    config = PathlayerConfig.from_mapping(
        {"detectors": {"no_progress": {"window": 2, "threshold": 2.0, "target_rate": 0.2}}}
    )
    with pytest.raises(ConfigError, match=r"invalid configuration value.*window must be"):
        build_pipeline(config)


def test_build_pipeline_all_enabled_includes_resource() -> None:
    config = PathlayerConfig.from_mapping({"detectors": {"resource": {"enabled": True}}})
    pipeline = build_pipeline(config)
    assert ResourceDetector in [type(d) for d in pipeline._detectors]


def test_build_pipeline_threads_resource_thresholds() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "resource": {
                    "enabled": True,
                    "max_spawn_count": 10,
                    "max_delegation_depth": 4,
                    "max_restart_count": 6,
                    "max_duration_seconds": 120,
                }
            }
        }
    )
    pipeline = build_pipeline(config)
    resource = next(d for d in pipeline._detectors if isinstance(d, ResourceDetector))

    assert resource._max_spawn_count == 10
    assert resource._max_delegation_depth == 4
    assert resource._max_restart_count == 6
    assert resource.max_duration_seconds == 120


def test_resource_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError, match=r"detectors.resource.*'max_spawn'"):
        PathlayerConfig.from_mapping({"detectors": {"resource": {"max_spawn": 10}}})


def test_build_pipeline_wraps_resource_range_error_as_config_error() -> None:
    config = PathlayerConfig.from_mapping(
        {"detectors": {"resource": {"enabled": True, "max_spawn_count": 0}}}
    )
    with pytest.raises(ConfigError, match=r"invalid configuration value.*max_spawn_count"):
        build_pipeline(config)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_resource_duration_requires_a_positive_integer(bad: object) -> None:
    with pytest.raises(ConfigError, match="max_duration_seconds"):
        config = PathlayerConfig.from_mapping(
            {"detectors": {"resource": {"enabled": True, "max_duration_seconds": bad}}}
        )
        build_pipeline(config)


# --------------------------------------------------------------------------- #
# policy ladder + hysteresis (ARB-3)
# --------------------------------------------------------------------------- #


def test_tier_thresholds_parsed_and_built() -> None:
    config = PathlayerConfig.from_mapping(
        {"policy": {"tier_thresholds": {"warn": 0.5, "escalate": 0.75}}}
    )
    assert config.policy.tier_thresholds == {"warn": 0.5, "escalate": 0.75}
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.tier_thresholds == {
        Action.WARN: 0.5,
        Action.ESCALATE: 0.75,
    }


def test_tier_thresholds_default_empty_and_ticks_default_one() -> None:
    config = PathlayerConfig.from_mapping({"policy": {}})
    assert config.policy.tier_thresholds == {}
    assert config.policy.escalate_ticks == 1
    assert config.policy.deescalate_ticks == 1


def test_unknown_tier_name_rejected() -> None:
    with pytest.raises(ConfigError, match=r"unknown tier.*policy.tier_thresholds.*'wrn'"):
        PathlayerConfig.from_mapping({"policy": {"tier_thresholds": {"wrn": 0.5}}})


def test_stop_tier_rejected_uses_stop_threshold() -> None:
    with pytest.raises(ConfigError, match=r"unknown tier.*'stop'.*stop_threshold"):
        PathlayerConfig.from_mapping({"policy": {"tier_thresholds": {"stop": 0.5}}})


def test_tier_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ConfigError, match=r"policy.tier_thresholds.warn.*\[0, 1\]"):
        PathlayerConfig.from_mapping({"policy": {"tier_thresholds": {"warn": 1.5}}})


def test_ticks_parsed() -> None:
    config = PathlayerConfig.from_mapping({"policy": {"escalate_ticks": 3, "deescalate_ticks": 2}})
    assert config.policy.escalate_ticks == 3
    assert config.policy.deescalate_ticks == 2


def test_escalate_ticks_non_int_rejected() -> None:
    with pytest.raises(ConfigError, match=r"policy.escalate_ticks.*expected int"):
        PathlayerConfig.from_mapping({"policy": {"escalate_ticks": 1.5}})


def test_build_pipeline_rejects_tier_at_or_above_stop_threshold() -> None:
    config = PathlayerConfig.from_mapping(
        {"policy": {"stop_threshold": 0.9, "tier_thresholds": {"escalate": 0.95}}}
    )
    with pytest.raises(ConfigError, match=r"invalid configuration value.*< stop_threshold"):
        build_pipeline(config)


def test_build_pipeline_rejects_non_monotone_tiers() -> None:
    config = PathlayerConfig.from_mapping(
        {"policy": {"tier_thresholds": {"warn": 0.8, "summarize": 0.5}}}
    )
    with pytest.raises(ConfigError, match=r"invalid configuration value.*non-decreasing"):
        build_pipeline(config)


def test_build_pipeline_threads_ticks() -> None:
    config = PathlayerConfig.from_mapping({"policy": {"escalate_ticks": 3, "deescalate_ticks": 2}})
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.escalate_ticks == 3
    assert pipeline._policy.config.deescalate_ticks == 2


# --------------------------------------------------------------------------- #
# ARB-4: priority overrides through config
# --------------------------------------------------------------------------- #


def test_overrides_default_empty() -> None:
    assert PathlayerConfig.from_mapping({"policy": {}}).policy.overrides == {}


def test_overrides_parsed_and_built() -> None:
    config = PathlayerConfig.from_mapping(
        {"policy": {"overrides": {"loop": {"threshold": 0.8, "action": "stop"}}}}
    )
    assert config.policy.overrides == {"loop": OverrideRule(0.8, Action.STOP)}
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.overrides["loop"] == OverrideRule(0.8, Action.STOP)


def test_override_defaults_threshold_one_action_stop() -> None:
    # An empty rule body means "stop at ceiling" -- the budget case.
    config = PathlayerConfig.from_mapping({"policy": {"overrides": {"loop": {}}}})
    assert config.policy.overrides["loop"] == OverrideRule(1.0, Action.STOP)


def test_override_unknown_signal_rejected() -> None:
    with pytest.raises(ConfigError, match=r"unknown signal.*policy.overrides.*'budgt'"):
        PathlayerConfig.from_mapping({"policy": {"overrides": {"budgt": {"threshold": 1.0}}}})


def test_override_unknown_action_rejected() -> None:
    with pytest.raises(ConfigError, match=r"unknown override action 'halt'"):
        PathlayerConfig.from_mapping({"policy": {"overrides": {"budget": {"action": "halt"}}}})


def test_override_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ConfigError, match=r"policy.overrides.budget.threshold.*\[0, 1\]"):
        PathlayerConfig.from_mapping({"policy": {"overrides": {"budget": {"threshold": 1.5}}}})


def test_override_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError, match=r"unknown key.*policy.overrides.budget"):
        PathlayerConfig.from_mapping({"policy": {"overrides": {"budget": {"thresh": 1.0}}}})


def test_override_and_advisory_conflict_rejected() -> None:
    with pytest.raises(ConfigError, match=r"cannot be both advisory and overridden.*budget"):
        PathlayerConfig.from_mapping(
            {"policy": {"advisory_signals": ["budget"], "overrides": {"budget": {}}}}
        )


def test_budget_override_auto_wired_when_enabled() -> None:
    config = PathlayerConfig.from_mapping({"detectors": {"budget": {"enabled": True}}})
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.overrides == {"budget": OverrideRule(1.0, Action.STOP)}


def test_no_budget_override_when_budget_disabled() -> None:
    # Default config: budget disabled -> zero-config stays ARB-1/2 (no overrides).
    assert build_pipeline()._policy.config.overrides == {}


def test_user_budget_override_wins_over_auto() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {"budget": {"enabled": True}},
            "policy": {"overrides": {"budget": {"threshold": 0.95, "action": "escalate"}}},
        }
    )
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.overrides == {"budget": OverrideRule(0.95, Action.ESCALATE)}


def test_no_budget_override_when_budget_advisory() -> None:
    # finding A/F: budget observe-only -> no auto override (advisory wins).
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {"budget": {"enabled": True}},
            "policy": {"advisory_signals": ["budget"]},
        }
    )
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.overrides == {}


def test_resource_override_auto_wired_when_enabled() -> None:
    config = PathlayerConfig.from_mapping({"detectors": {"resource": {"enabled": True}}})
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.overrides == {}
    assert pipeline._policy.config.conditional_overrides == {
        "resource": (DURATION_REASON_PREFIX, OverrideRule(1.0, Action.STOP))
    }


def test_user_resource_override_wins_over_auto() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {"resource": {"enabled": True}},
            "policy": {"overrides": {"resource": {"threshold": 1.0, "action": "escalate"}}},
        }
    )
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.overrides["resource"] == OverrideRule(1.0, Action.ESCALATE)
    assert pipeline._policy.config.conditional_overrides == {}


def test_no_resource_override_when_resource_advisory() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {"resource": {"enabled": True}},
            "policy": {"advisory_signals": ["resource"]},
        }
    )
    pipeline = build_pipeline(config)
    assert "resource" not in pipeline._policy.config.overrides
    assert pipeline._policy.config.conditional_overrides == {}


def test_resource_counter_ceiling_keeps_normal_hysteresis() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {"resource": {"enabled": True, "max_spawn_count": 1}},
            "policy": {"escalate_ticks": 3},
        }
    )
    pipeline = build_pipeline(config)

    assert pipeline.process(ToolEvent("spawn", kind=EventKind.SUBAGENT_SPAWN)) is Action.NONE


def test_resource_timeout_override_wins_with_saturated_counter_and_hysteresis() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "resource": {
                    "enabled": True,
                    "max_spawn_count": 1,
                    "max_duration_seconds": 10,
                }
            },
            "policy": {"escalate_ticks": 3, "weights": {"resource": 0.01}},
        }
    )
    pipeline = build_pipeline(config)

    assert (
        pipeline.process(
            ToolEvent(
                "pathlayer_run_start",
                kind=EventKind.RUN_START,
                metadata={"started_at_unix": 100.0, "max_duration_seconds": 10},
            )
        )
        is Action.NONE
    )
    assert pipeline.process(ToolEvent("spawn", kind=EventKind.SUBAGENT_SPAWN)) is Action.NONE
    assert (
        pipeline.process(
            ToolEvent(
                "pathlayer_run_timeout",
                kind=EventKind.RUN_TIMEOUT,
                metadata={"started_at_unix": 100.0, "max_duration_seconds": 10},
            )
        )
        is Action.STOP
    )


# --------------------------------------------------------------------------- #
# LOOP-2 precision: loop<-progress suppression auto-wired by build_pipeline
# --------------------------------------------------------------------------- #


def test_loop_suppression_auto_wired_by_default() -> None:
    # Both loop and no_progress are on by default, so the zero-config pipeline
    # already carries the F1a fix.
    pipeline = build_pipeline()
    rule = pipeline._policy.config.suppressions["loop"]
    assert rule.evidence == PROGRESS_EVIDENCE_KEY
    assert rule.min_confidence == 1.0
    assert pipeline._progress_anchors


def test_no_loop_suppression_when_no_progress_disabled() -> None:
    # The anchors backing the suppression come from the no_progress feature
    # set; disabling it removes the suppression along with them.
    config = PathlayerConfig.from_mapping({"detectors": {"no_progress": {"enabled": False}}})
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.suppressions == {}
    assert pipeline._progress_anchors == []


def test_no_loop_suppression_when_loop_disabled() -> None:
    config = PathlayerConfig.from_mapping({"detectors": {"loop": {"enabled": False}}})
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.suppressions == {}
    assert pipeline._progress_anchors == []


def test_no_loop_suppression_when_loop_advisory() -> None:
    # Advisory already makes loop observe-only; nothing to guard, so no
    # suppression rule is wired (mirrors the budget-override/advisory guard).
    config = PathlayerConfig.from_mapping({"policy": {"advisory_signals": ["loop"]}})
    pipeline = build_pipeline(config)
    assert pipeline._policy.config.suppressions == {}
    assert pipeline._progress_anchors == []


def test_f1a_repetitive_but_passing_no_longer_stops() -> None:
    # End-to-end regression pin for the bug this fix resolves (issue #113): a
    # test suite re-run with identical args, passed count climbing, must not be
    # stopped by loop alone through the real production pipeline.
    events = [
        ToolEvent(tool="run_tests", args={}, result={"passed": p, "failed": 0}) for p in range(1, 9)
    ]
    assert Action.STOP not in build_pipeline().run(events)


def test_git_only_progress_never_suppresses_loop_through_build_pipeline() -> None:
    # Trou C adversarial case: a ping-pong between two files with a growing
    # git diff and zero test/exit_code evidence must still STOP -- the git
    # anchor is deliberately excluded from the suppression's anchor set
    # (loop_suppression_anchors), unlike the full set NoProgressDetector uses.
    events = []
    for i in range(1, 13):
        events.append(
            ToolEvent(
                tool="edit_file",
                args={"path": "a.py" if i % 2 else "b.py", "patch": "pad"},
                result={"insertions": i * 5, "deletions": 0},
            )
        )
    assert Action.STOP in build_pipeline().run(events)


# --------------------------------------------------------------------------- #
# tool_masks (CAN-2) exposed through config
# --------------------------------------------------------------------------- #


def test_tool_mask_overlap_rejected() -> None:
    with pytest.raises(ConfigError, match=r"invalid tool mask for 'api_call'"):
        PathlayerConfig.from_mapping(
            {"canonicalization": {"tool_masks": {"api_call": {"keep": ["x"], "mask": ["x"]}}}}
        )


def test_tool_mask_spec_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError, match=r"tool_masks.api_call.*'kep'"):
        PathlayerConfig.from_mapping(
            {"canonicalization": {"tool_masks": {"api_call": {"kep": ["x"]}}}}
        )


def test_mask_collapses_volatile_id_into_a_loop() -> None:
    # Without a mask, an incrementing attempt_id keeps each call distinct, so no
    # loop fires. Declaring it volatile per tool collapses the calls to one token.
    events = [ToolEvent("api_call", {"attempt_id": i, "url": "x"}) for i in range(1, 6)]

    no_mask = build_pipeline(
        PathlayerConfig.from_mapping({"detectors": {"no_progress": {"enabled": False}}})
    )
    assert Action.STOP not in no_mask.run(events)

    masked = build_pipeline(
        PathlayerConfig.from_mapping(
            {
                "detectors": {"no_progress": {"enabled": False}},
                "canonicalization": {"tool_masks": {"api_call": {"mask": ["attempt_id"]}}},
            }
        )
    )
    assert masked.run(events)[-1] is Action.STOP


def test_keep_preserves_distinguishing_id() -> None:
    # A UUID-valued file_id is masked by value-shape by default, falsely merging
    # distinct reads into a loop. ``keep`` preserves it so no loop fires.
    events = [ToolEvent("read_file", {"file_id": str(uuid4())}) for _ in range(5)]

    default = build_pipeline(
        PathlayerConfig.from_mapping({"detectors": {"no_progress": {"enabled": False}}})
    )
    assert default.run(events)[-1] is Action.STOP

    kept = build_pipeline(
        PathlayerConfig.from_mapping(
            {
                "detectors": {"no_progress": {"enabled": False}},
                "canonicalization": {"tool_masks": {"read_file": {"keep": ["file_id"]}}},
            }
        )
    )
    assert Action.STOP not in kept.run(events)


# --------------------------------------------------------------------------- #
# pricing: the token price table (OBS-2)
# --------------------------------------------------------------------------- #


def test_pricing_defaults_change_nothing() -> None:
    config = PathlayerConfig()
    assert config.pricing.models == {}
    assert config.pricing.replace_defaults is False
    # Configuring nothing must leave the built-in table exactly as it is.
    assert config.pricing.effective_prices() == DEFAULT_PRICES
    assert PathlayerConfig.from_mapping({}).pricing == config.pricing


def test_pricing_extends_the_builtin_table_by_default() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "pricing": {
                "models": {
                    "house-model": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0},
                    "gpt-4o": {"input_usd_per_mtok": 9.0, "output_usd_per_mtok": 9.0},
                }
            }
        }
    )
    prices = config.pricing.effective_prices()

    assert prices["house-model"] == ModelPrice(1.0, 2.0)
    # An entry for a known model overrides it; every other default survives.
    assert prices["gpt-4o"] == ModelPrice(9.0, 9.0)
    assert prices["claude-3-opus"] == DEFAULT_PRICES["claude-3-opus"]
    # The configured table is what cost_of multiplies the token counts by.
    assert cost_of("house-model", 1_000_000, 1_000_000, prices=prices) == pytest.approx(3.0)


def test_pricing_can_replace_the_builtin_table_entirely() -> None:
    config = PathlayerConfig.from_mapping(
        {
            "pricing": {
                "replace_defaults": True,
                "models": {"house-model": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0}},
            }
        }
    )
    prices = config.pricing.effective_prices()

    assert set(prices) == {"house-model"}
    # Everything unlisted is now invisible to the budget detector -- an explicit
    # opt-in, which is why it is not the default.
    assert cost_of("gpt-4o", 1_000_000, 0, prices=prices) == 0.0


def test_pricing_zero_rate_is_a_valid_free_model() -> None:
    config = PathlayerConfig.from_mapping(
        {"pricing": {"models": {"local": {"input_usd_per_mtok": 0.0, "output_usd_per_mtok": 0.0}}}}
    )
    assert config.pricing.models["local"] == ModelPrice(0.0, 0.0)


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"models": {"m": {"input_usd_per_mtok": 1.0}}}, "output_usd_per_mtok"),
        ({"models": {"m": {"output_usd_per_mtok": 1.0}}}, "input_usd_per_mtok"),
        ({"models": {"m": {}}}, "missing required key"),
        ({"models": {"m": None}}, "missing required key"),
        (
            {"models": {"m": {"input_usd_per_mtok": -1.0, "output_usd_per_mtok": 1.0}}},
            r">= 0",
        ),
        (
            {"models": {"m": {"input_usd_per_mtok": "free", "output_usd_per_mtok": 1.0}}},
            "expected number",
        ),
        (
            {
                "models": {
                    "m": {
                        "input_usd_per_mtok": 1.0,
                        "output_usd_per_mtok": 1.0,
                        "currency": "EUR",
                    }
                }
            },
            r"unknown key\(s\) in section 'pricing.models.m'",
        ),
        ({"models": {"m": 2.5}}, "must be a mapping"),
        ({"models": 2.5}, "must be a mapping"),
        ({"replace_defaults": "yes"}, "expected bool"),
        ({"model": {}}, r"unknown key\(s\) in section 'pricing'"),
        (
            {"models": {"": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 1.0}}},
            "empty id can never match",
        ),
    ],
)
def test_invalid_pricing_fails_fast(section: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        PathlayerConfig.from_mapping({"pricing": section})


def test_pricing_is_not_a_cost_source_by_itself(tmp_path: Path) -> None:
    # A price table only becomes a cost when something reports token counts.
    # Nothing in the config can conjure them, so an infinite rate still prices
    # a call with no reported usage at $0 -- which is why an OBS-2 report says
    # the cost is unmeasured on the OpenClaw path instead of printing $0.00.
    config = PathlayerConfig.from_mapping(
        {
            "pricing": {
                "models": {"m": {"input_usd_per_mtok": 1000.0, "output_usd_per_mtok": 1000.0}}
            }
        }
    )
    prices = config.pricing.effective_prices()
    assert cost_of("m", 0, 0, prices=prices) == 0.0
    assert cost_of(None, 1_000_000, 1_000_000, prices=prices) == 0.0


# --------------------------------------------------------------------------- #
# Shipped example must stay valid (anti-drift)
# --------------------------------------------------------------------------- #


def test_shipped_example_loads_and_matches_defaults() -> None:
    # Every value in the reference example is a default, so it must round-trip to
    # the default config -- this guards against the example drifting from code.
    assert load_config(EXAMPLE_CONFIG) == PathlayerConfig()
    build_pipeline(load_config(EXAMPLE_CONFIG))
