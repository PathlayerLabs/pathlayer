"""Private, self-contained pytest plugin used by :mod:`pathlayer.coding_contract_probe`.

The probe copies this file into a temporary directory under a random module
name and loads it with ``-p``. The tests may run under the *project's*
interpreter (the contract's ``python`` key), where Pathlayer is not installed,
so this module imports only the standard library and must stay valid
Python 3.8. That is also why the outcome collapse below repeats the ARCH-1B
evidence plugin (``pathlayer/validation/_pytest_evidence_plugin.py``) instead
of importing it.

Every output is opt-in through an environment variable naming a file:

- ``OUTPUT_ENV``: ``{"exit_code": int, "outcomes": {nodeid: "passed"|"failed"}}``
  -- setup, call and teardown collapse to one pass/non-pass per nodeid, a skip
  is not a pass, and more than ``MAX_OUTCOMES`` nodeids (or an unusable one)
  invalidate the capture (``"outcomes": null``);
- ``SELECT_ENV`` (input): keep only the listed nodeids, deselect everything
  else. The probe passes test *files*, so a required test that was deleted or
  renamed is simply absent instead of aborting the whole session the way an
  unknown nodeid argument does (pytest exit status 4);
- ``COLLECT_ERRORS_ENV``: the nodeids of collectors that failed -- a module or
  package that cannot be imported, maybe for want of a dependency;
- ``COLLECT_ENV``: the collected nodeids, or ``null`` when unusable. Used to
  expand a ``tests_improve`` path and to check an interpreter at compile time.
"""

from __future__ import annotations

import json
import os
from typing import Any

OUTPUT_ENV = "PATHLAYER_CONTRACT_PYTEST_OUTPUT"
SELECT_ENV = "PATHLAYER_CONTRACT_PYTEST_SELECT"
COLLECT_ENV = "PATHLAYER_CONTRACT_PYTEST_COLLECT"
COLLECT_ERRORS_ENV = "PATHLAYER_CONTRACT_PYTEST_COLLECT_ERRORS"
MAX_OUTCOMES = 1_000
MAX_NODEID_LENGTH = 4_096

_outcomes: dict[str, str] = {}
_collection_errors: set[str] = set()
_capture_invalid = False


def _usable_nodeid(nodeid: object) -> bool:
    return isinstance(nodeid, str) and 0 < len(nodeid) <= MAX_NODEID_LENGTH


def _write_json(variable: str, payload: object) -> None:
    path = os.environ.get(variable)
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def pytest_sessionstart(session: Any) -> None:
    """Reset process-local capture state before pytest starts a session."""
    global _capture_invalid
    del session
    _outcomes.clear()
    _collection_errors.clear()
    _capture_invalid = False


def pytest_collectreport(report: Any) -> None:
    """Remember every collector that failed (import or syntax error)."""
    bounded = len(_collection_errors) < MAX_OUTCOMES
    if report.failed and isinstance(report.nodeid, str) and bounded:
        _collection_errors.add(report.nodeid)


def pytest_collection_modifyitems(session: Any, config: Any, items: list[Any]) -> None:
    """Export the collection, then keep only the selected nodeids."""
    del session
    nodeids = [item.nodeid for item in items]
    usable = len(nodeids) <= MAX_OUTCOMES and all(_usable_nodeid(nodeid) for nodeid in nodeids)
    _write_json(COLLECT_ENV, nodeids if usable else None)
    selection = os.environ.get(SELECT_ENV)
    if not selection:
        return
    with open(selection, encoding="utf-8") as handle:
        wanted = set(json.load(handle))
    kept = [item for item in items if item.nodeid in wanted]
    deselected = [item for item in items if item.nodeid not in wanted]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = kept


def pytest_runtest_logreport(report: Any) -> None:
    """Collapse setup/call/teardown reports into pass or non-pass per nodeid."""
    global _capture_invalid
    if _capture_invalid:
        return
    nodeid = report.nodeid
    if not _usable_nodeid(nodeid) or (nodeid not in _outcomes and len(_outcomes) >= MAX_OUTCOMES):
        _outcomes.clear()
        _capture_invalid = True
        return
    if report.failed or report.skipped:
        _outcomes[nodeid] = "failed"
    elif report.when == "call" and report.passed and _outcomes.get(nodeid) != "failed":
        _outcomes[nodeid] = "passed"


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Write the outcomes and the failed collectors."""
    del session
    outcomes = None if _capture_invalid else dict(sorted(_outcomes.items()))
    _write_json(OUTPUT_ENV, {"exit_code": int(exitstatus), "outcomes": outcomes})
    _write_json(COLLECT_ERRORS_ENV, sorted(_collection_errors))
