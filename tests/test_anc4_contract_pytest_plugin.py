"""ANC-4: the private pytest plugin, exercised in-process.

End to end it only ever runs inside the probe's pytest subprocess -- possibly
under the project's own interpreter, where Pathlayer is not installed (covered
by ``tests/test_anc4_contract_probe.py``). These tests pin each hook directly
and the constraints that let it run there at all.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from pathlayer import _contract_pytest_plugin as plugin


@dataclass
class _Item:
    nodeid: str


@dataclass
class _CollectReport:
    nodeid: str
    failed: bool = True


@dataclass
class _TestReport:
    nodeid: str
    when: str
    outcome: str

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"

    @property
    def skipped(self) -> bool:
        return self.outcome == "skipped"


@dataclass
class _Hook:
    deselected: list[Any] = field(default_factory=list)

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected.extend(items)


@dataclass
class _Config:
    hook: _Hook = field(default_factory=_Hook)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        plugin.SELECT_ENV,
        plugin.COLLECT_ENV,
        plugin.COLLECT_ERRORS_ENV,
        plugin.OUTPUT_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    plugin.pytest_sessionstart(None)


def test_plugin_is_self_contained_and_valid_python_3_8() -> None:
    """It is copied next to the project's interpreter, where only the stdlib exists."""
    source = Path(plugin.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 8))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported <= {"__future__", "json", "os", "typing"}


def test_outcomes_collapse_setup_call_and_teardown_per_nodeid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "outcomes.json"
    monkeypatch.setenv(plugin.OUTPUT_ENV, str(output))
    reports = [
        _TestReport("t.py::ok", "setup", "passed"),
        _TestReport("t.py::ok", "call", "passed"),
        _TestReport("t.py::ok", "teardown", "passed"),
        _TestReport("t.py::bad_setup", "setup", "failed"),
        _TestReport("t.py::skipped", "setup", "skipped"),
        _TestReport("t.py::bad_teardown", "call", "passed"),
        _TestReport("t.py::bad_teardown", "teardown", "failed"),
        _TestReport("t.py::fails", "call", "failed"),
    ]

    for report in reports:
        plugin.pytest_runtest_logreport(report)
    plugin.pytest_sessionfinish(None, 1)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "exit_code": 1,
        "outcomes": {
            "t.py::bad_setup": "failed",
            "t.py::bad_teardown": "failed",
            "t.py::fails": "failed",
            "t.py::ok": "passed",
            "t.py::skipped": "failed",
        },
    }


def test_unbounded_outcomes_invalidate_the_whole_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "outcomes.json"
    monkeypatch.setenv(plugin.OUTPUT_ENV, str(output))

    plugin.pytest_runtest_logreport(_TestReport("t.py::ok", "call", "passed"))
    plugin.pytest_runtest_logreport(
        _TestReport("t.py::" + "x" * plugin.MAX_NODEID_LENGTH, "call", "passed")
    )
    plugin.pytest_runtest_logreport(_TestReport("t.py::late", "call", "passed"))
    plugin.pytest_sessionfinish(None, 0)

    assert json.loads(output.read_text(encoding="utf-8")) == {"exit_code": 0, "outcomes": None}


def test_selection_keeps_required_nodeids_and_deselects_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = tmp_path / "select.json"
    selection.write_text(json.dumps(["a.py::keep"]), encoding="utf-8")
    monkeypatch.setenv(plugin.SELECT_ENV, str(selection))
    items = [_Item("a.py::keep"), _Item("a.py::drop")]
    config = _Config()

    plugin.pytest_collection_modifyitems(None, config, items)

    assert [item.nodeid for item in items] == ["a.py::keep"]
    assert [item.nodeid for item in config.hook.deselected] == ["a.py::drop"]


def test_without_selection_nothing_is_deselected() -> None:
    items = [_Item("a.py::x")]
    config = _Config()

    plugin.pytest_collection_modifyitems(None, config, items)

    assert [item.nodeid for item in items] == ["a.py::x"]
    assert config.hook.deselected == []


def test_collection_is_exported_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "collected.json"
    monkeypatch.setenv(plugin.COLLECT_ENV, str(output))

    plugin.pytest_collection_modifyitems(None, _Config(), [_Item("a.py::x"), _Item("b.py::y")])
    assert json.loads(output.read_text(encoding="utf-8")) == ["a.py::x", "b.py::y"]

    too_many = [_Item(f"a.py::t{index}") for index in range(plugin.MAX_OUTCOMES + 1)]
    plugin.pytest_collection_modifyitems(None, _Config(), too_many)
    assert json.loads(output.read_text(encoding="utf-8")) is None


def test_failed_collectors_are_written_after_the_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = tmp_path / "outcomes.json"
    errors = tmp_path / "errors.json"
    monkeypatch.setenv(plugin.OUTPUT_ENV, str(outcomes))
    monkeypatch.setenv(plugin.COLLECT_ERRORS_ENV, str(errors))

    plugin.pytest_collectreport(_CollectReport("tests/test_broken.py"))
    plugin.pytest_collectreport(_CollectReport("tests/test_fine.py", failed=False))
    plugin.pytest_sessionfinish(None, 1)

    assert json.loads(errors.read_text(encoding="utf-8")) == ["tests/test_broken.py"]
    assert json.loads(outcomes.read_text(encoding="utf-8")) == {"exit_code": 1, "outcomes": {}}


def test_session_start_resets_every_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = tmp_path / "outcomes.json"
    errors = tmp_path / "errors.json"
    monkeypatch.setenv(plugin.OUTPUT_ENV, str(outcomes))
    monkeypatch.setenv(plugin.COLLECT_ERRORS_ENV, str(errors))
    plugin.pytest_collectreport(_CollectReport("tests/test_old.py"))
    plugin.pytest_runtest_logreport(_TestReport("t.py::" + "x" * 5000, "call", "passed"))

    plugin.pytest_sessionstart(None)
    plugin.pytest_sessionfinish(None, 0)

    assert json.loads(errors.read_text(encoding="utf-8")) == []
    assert json.loads(outcomes.read_text(encoding="utf-8")) == {"exit_code": 0, "outcomes": {}}
