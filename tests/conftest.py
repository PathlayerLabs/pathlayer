"""Shared test helpers: load labeled fixture traces from ``traces/``.

The on-disk trace format is intentionally minimal for Sprint 1 (a JSON object
with a ``label`` and a list of ``events``); the formal schema is VAL-1.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from pathlayer.events import ToolEvent

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def load_trace(name: str) -> list[ToolEvent]:
    data = json.loads((_TRACES_DIR / name).read_text(encoding="utf-8"))
    return [ToolEvent(tool=event["tool"], args=event.get("args", {})) for event in data["events"]]


@pytest.fixture
def trace() -> Callable[[str], list[ToolEvent]]:
    return load_trace


def git_workspace(root: Path) -> Path:
    """A real repository with one commit, one ignored directory and one file.

    Shared by the STATE-1 preset and runtime suites, which both need a genuine
    working tree rather than a simulated one.
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "."], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "src").mkdir()
    (root / "src" / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    (root / ".gitignore").write_text("build/\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
    return root
