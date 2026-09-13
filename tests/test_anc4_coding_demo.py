"""ANC-4 acceptance: the end-to-end coding example (examples/demo_coding_contract.py).

The example lives outside the ``pathlayer`` package, so it is loaded from its
path like ``tests/test_demo2.py`` does. Every scenario runs once, for real: a
compiled LangGraph graph, the default detectors, a contract compiled from YAML,
the generated probe running pytest after every tool call, and STOP enforced
through ``LangGraphActuator``. The expected outcomes below were written before
the scenarios were first run; no threshold is tuned for them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")
pytest.importorskip("yaml")

_DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "demo_coding_contract.py"


def _load_demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_coding_contract", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    demo = _load_demo()
    return {result.name: result for result in demo.run_all(tmp_path_factory.mktemp("anc4"))}


def test_every_scenario_decides_identically_live_replayed_and_low_level(
    results: dict[str, Any],
) -> None:
    assert set(results) == {
        "productive",
        "stagnant",
        "oscillating",
        "objective_reached",
        "probe_unavailable",
        "unobservable_mid_run",
    }
    for result in results.values():
        assert result.replay_identical, result.name
        assert result.low_level_identical, result.name


def test_productive_repetition_is_left_alone(results: dict[str, Any]) -> None:
    productive = results["productive"]
    assert productive.live_stop is None
    assert productive.events == productive.proofs == 16
    # A test turns green every four calls: patience never runs out.
    assert productive.no_progress_max == pytest.approx(0.3)
    assert productive.baseline_stop is None  # no false positive added or removed


def test_stagnation_is_stopped_only_with_the_contract(results: dict[str, Any]) -> None:
    """Activity without success: calculator.py changes on every call (an observed
    effect worth one step, once) while no test ever passes."""
    stagnant = results["stagnant"]
    assert stagnant.live_stop == 11
    assert stagnant.no_progress_max == pytest.approx(1.0)
    # Without the contract the same stuck agent runs its whole script.
    assert stagnant.baseline_stop is None
    assert stagnant.baseline_events == 16
    assert stagnant.ablation_stop is None


def test_oscillation_never_counts_as_new_progress(results: dict[str, Any]) -> None:
    oscillating = results["oscillating"]
    assert oscillating.live_stop == 11
    assert oscillating.baseline_stop is None


def test_reached_objective_silences_no_progress(results: dict[str, Any]) -> None:
    reached = results["objective_reached"]
    assert reached.live_stop is None
    assert reached.no_progress_max == 0.0
    assert reached.events == 16


def test_unavailable_probe_is_fail_open_and_explicit(results: dict[str, Any]) -> None:
    unavailable = results["probe_unavailable"]
    assert unavailable.proofs == 0
    assert unavailable.unobserved == unavailable.events == 16
    assert unavailable.live_stop is None
    assert unavailable.no_progress_max == 0.0
    assert unavailable.baseline_stop is None  # exactly today's behaviour
    assert len(unavailable.warnings) == unavailable.events == 16
    assert all("no proof this tick" in message for message in unavailable.warnings)
    assert results["stagnant"].warnings == ()


def test_probe_outage_mid_run_is_a_gap_not_a_stop(results: dict[str, Any]) -> None:
    """Ten consecutive calls whose tests cannot be collected (the agent broke
    the module's syntax mid-refactor) used to freeze the level and STOP at
    the eleventh; they are now gaps, and the run completes."""
    outage = results["unobservable_mid_run"]
    assert outage.live_stop is None
    assert outage.events == outage.proofs == 22  # the file is observed on every call
    assert outage.unobserved == 10
    assert len(outage.warnings) == 10
    assert all("did not report 4 of 5 required checkpoint(s)" in m for m in outage.warnings)
    assert outage.baseline_stop is None
    assert outage.ablation_stop is None
    for result in results.values():
        if result.name not in {"probe_unavailable", "unobservable_mid_run"}:
            assert result.unobserved == 0, result.name
