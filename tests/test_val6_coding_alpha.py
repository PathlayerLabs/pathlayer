"""VAL-6 acceptance on real tools, captured evidence, enforcement and replay."""

from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("yaml")
from examples.demo_coding_alpha import SCENARIOS, run_scenario


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("val6")
    return {name: run_scenario(name, root / name) for name in SCENARIOS}


def test_productive_repetition_reaches_contract_without_stop(results):
    result = results["productive"]
    assert result["stop"] is None
    assert result["diagnostic"]["contract_state"] == "reached"
    assert "test_divide" in result["diagnostic"]["last_progress"]


def test_stagnation_stops_and_activity_is_not_success(results):
    stagnant = results["stagnant"]
    assert stagnant["stop"] == 3
    assert stagnant["baseline_stop"] == 4
    assert stagnant["diagnostic"]["state_outcome"] == "unchanged"
    activity = results["activity_without_success"]
    assert activity["stop"] == 10
    assert activity["diagnostic"]["intervention"] == "no_progress"
    assert activity["diagnostic"]["contract_state"] == "incomplete"
    assert activity["diagnostic"]["state_outcome"] == "changed"
    assert activity["diagnostic"]["state_resources"] == ["calculator.py"]


@pytest.mark.parametrize("name", ["unavailable", "collection_failed"])
def test_missing_evidence_is_not_stagnation(results, name):
    result = results[name]
    assert result["diagnostic"]["state_outcome"] == name
    assert result["no_progress_max"] == 0
    assert result["diagnostic"]["intervention"] == "loop"
    assert result["diagnostic"]["contract_state"] == "no contract observed"


def test_all_recorded_decisions_replay_after_workspace_deletion(results):
    assert all(result["replay_identical"] for result in results.values())


def test_unknown_scenario_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown scenario"):
        run_scenario("typo", tmp_path)
