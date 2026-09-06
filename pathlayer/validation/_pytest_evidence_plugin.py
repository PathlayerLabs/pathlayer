"""Private pytest plugin used by :mod:`pathlayer.validation.state_collectors`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pathlayer.validation._pytest_evidence_protocol import (
    MAX_NODEID_LENGTH,
    MAX_OUTCOMES,
    OUTPUT_ENV,
)

_outcomes: dict[str, str] = {}
_capture_invalid = False


def pytest_sessionstart(session: Any) -> None:
    """Reset process-local capture state before pytest starts a session."""
    global _capture_invalid
    del session
    _outcomes.clear()
    _capture_invalid = False


def pytest_runtest_logreport(report: Any) -> None:
    """Collapse setup/call/teardown reports into pass or non-pass per nodeid."""
    global _capture_invalid
    if _capture_invalid:
        return
    nodeid = report.nodeid
    if (
        not isinstance(nodeid, str)
        or not nodeid
        or len(nodeid) > MAX_NODEID_LENGTH
        or (nodeid not in _outcomes and len(_outcomes) >= MAX_OUTCOMES)
    ):
        _outcomes.clear()
        _capture_invalid = True
        return
    if report.failed or report.skipped:
        _outcomes[nodeid] = "failed"
    elif report.when == "call" and report.passed and _outcomes.get(nodeid) != "failed":
        _outcomes[nodeid] = "passed"


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Write the small stable payload consumed by the parent collector."""
    del session
    output = os.environ.get(OUTPUT_ENV)
    if not output:
        return
    outcomes = None if _capture_invalid else dict(sorted(_outcomes.items()))
    payload = {"exit_code": int(exitstatus), "outcomes": outcomes}
    Path(output).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
