"""Smoke tests for the DEMO-1 example (examples/demo_loop.py).

The demo lives in ``examples/`` (outside the ``pathlayer`` package and outside
coverage), so it is loaded by path. These tests are a regression guard: the
loop must be caught and stopped through the real ACT-0 path, and the healthy
bounded counterpart must never be stopped (no false positive).
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path
from types import ModuleType

from pathlayer.policies import Action

_DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "demo_loop.py"


def _load_demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_loop", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the demo uses `from __future__ import annotations`,
    # and dataclasses resolves field types via sys.modules[cls.__module__] at
    # class-creation time -- an unregistered module breaks @dataclass.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()
_LOGGER = logging.getLogger("test.demo1")


def test_loop_is_detected_and_stopped_via_actuator() -> None:
    pipeline, control, executed = demo._run_guarded(demo.looping_agent(), _LOGGER)

    assert pipeline.stopped is True
    assert control.stopped is True  # the actuator drained the run
    assert executed < demo.NAIVE_CAP  # stopped well before the naive cap
    pm = pipeline.post_mortem
    assert pm is not None
    assert "loop" in pm.reason
    assert demo._triggering_detector(pm) == "loop"


def test_healthy_agent_is_not_stopped() -> None:
    pipeline, control, executed = demo._run_guarded(demo.healthy_agent(), _LOGGER)

    assert pipeline.stopped is False
    assert control.stopped is False
    assert pipeline.post_mortem is None
    assert executed == 7  # the bounded healthy plan finishes on its own


def test_demo_actuator_ignores_non_stop_actions() -> None:
    control = demo.DemoRunControl()
    demo.DemoActuator(control).apply(Action.NONE, None)

    assert control.stopped is False
    assert control.reason == ""


def test_run_naive_bounds_the_infinite_agent() -> None:
    # looping_agent() never terminates; the naive run must halt at the cap,
    # otherwise the demo (and the before/after story) would hang forever.
    assert demo._run_naive(demo.looping_agent(), demo.NAIVE_CAP) == demo.NAIVE_CAP


def test_main_prints_before_after_and_no_false_positive(capsys) -> None:  # type: ignore[no-untyped-def]
    demo.main()

    out = capsys.readouterr().out
    assert "Sans Pathlayer" in out
    assert "STOP apres" in out
    assert "Detecteur declencheur : loop" in out
    # DoD #43: work avoided must be surfaced, with a coherent count.
    match = re.search(r"Travail evite\s*:\s*(\d+) actions sur (\d+)", out)
    assert match is not None
    avoided, total = int(match.group(1)), int(match.group(2))
    assert 0 < avoided < total
    assert "aucune intervention" in out
