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
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import ToolEvent
from pathlayer.policies import Action

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
    assert config.detectors.no_progress.window == 64
    assert config.policy.stop_threshold == 0.9
    assert config.policy.advisory_signals == frozenset()
    assert config.pipeline.recent_history == 10
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


def test_build_pipeline_rejects_out_of_range_threshold() -> None:
    # Range checks are delegated to the detector's own __init__ (one source).
    config = PathlayerConfig.from_mapping({"detectors": {"loop": {"k": 1}}})
    with pytest.raises(ValueError, match=r"k must be >= 2"):
        build_pipeline(config)


def test_build_pipeline_rejects_bad_stop_threshold() -> None:
    config = PathlayerConfig.from_mapping({"policy": {"stop_threshold": 2.0}})
    with pytest.raises(ValueError, match=r"stop_threshold"):
        build_pipeline(config)


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
# Shipped example must stay valid (anti-drift)
# --------------------------------------------------------------------------- #


def test_shipped_example_loads_and_matches_defaults() -> None:
    # Every value in the reference example is a default, so it must round-trip to
    # the default config -- this guards against the example drifting from code.
    assert load_config(EXAMPLE_CONFIG) == PathlayerConfig()
    build_pipeline(load_config(EXAMPLE_CONFIG))
