"""Run the reference OpenClaw plugin's Node tests (``tests/js/``).

The plugin's ``before_tool_call`` gate must fail open: OpenClaw blocks a tool
call when that handler throws or times out. Its behavior is tested in
JavaScript, against a stub listener; this wrapper runs those tests under
``pytest`` wherever Node.js 18+ is installed, and skips them elsewhere.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_NODE_TESTS = Path(__file__).resolve().parent / "js" / "openclaw_plugin.test.mjs"


def test_openclaw_plugin_gate() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if int(version.lstrip("v").split(".")[0]) < 18:
        pytest.skip(f"the plugin needs Node.js 18+ (fetch, AbortSignal.timeout), found {version}")
    completed = subprocess.run(
        [node, "--test", str(_NODE_TESTS)], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
