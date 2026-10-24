"""A loop detected and stopped through the real runtime path.

The smallest complete Pathlayer story: an agent gets stuck and repeats the
same actions. Without a guardrail it would run forever; with Pathlayer the
loop is detected and the agent is stopped cleanly.

The STOP goes through the real runtime path, not a shortcut:

    agent -> wire(pipeline, actuator) -> LoopDetector -> Policy -> STOP
          -> DemoActuator.apply(STOP) -> the agent drains and stops

``DemoActuator`` is the offline counterpart of ``LangGraphActuator``: on STOP
it requests a cooperative drain (a flag) that the agent observes. It never
decides -- the ``Action`` arrives already arbitrated by the pipeline.

A healthy, bounded counterpart (varied tools) is wired the same way and is NOT
stopped: the cost of a false positive is central.

Deterministic, offline, no LLM, no network. Run:

    python examples/demo_loop.py
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from pathlayer import Pipeline
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import ToolEvent
from pathlayer.observability import JsonFormatter, get_logger
from pathlayer.policies import Action, Actuator, wire
from pathlayer.postmortem import PostMortem

# Without a guardrail the stuck agent would never stop. The demo is capped so
# that it ends, while showing that at this cap the agent is "still going".
NAIVE_CAP = 200


@dataclass
class DemoRunControl:
    """Cooperative drain flag: the offline counterpart of LangGraph's RunControl.

    The actuator requests the drain (``stopped``); the agent checks it after
    each step and stops on its own. No framework, no network.
    """

    stopped: bool = False
    reason: str = ""


class DemoActuator(Actuator):
    """Applies STOP by requesting a drain on a ``DemoRunControl``.

    Modeled on ``LangGraphActuator``: acts only on ``Action.STOP`` and never
    decides -- the ``Action`` arrives already arbitrated by the pipeline.
    """

    def __init__(self, control: DemoRunControl) -> None:
        self._control = control

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        if action is not Action.STOP:
            return
        self._control.stopped = True
        self._control.reason = post_mortem.reason if post_mortem is not None else "stop"


def looping_agent() -> Iterator[ToolEvent]:
    """Stuck agent: lists, then rereads the same file, and never advances.

    Each step carries a distinct ``ts`` (events are not byte-identical);
    canonicalization matches them so that the loop is visible.
    """
    step = 0
    while True:
        for tool in ("list_dir", "read_file"):
            yield ToolEvent(tool=tool, args={"path": "src/main.py"}, ts=float(step))
            step += 1


def healthy_agent() -> Iterator[ToolEvent]:
    """Healthy, bounded agent: varied tools that advance the task, then it ends."""
    plan: list[tuple[str, dict[str, object]]] = [
        ("list_dir", {"path": "src"}),
        ("read_file", {"path": "src/main.py"}),
        ("edit_file", {"path": "src/main.py", "patch": "add guard"}),
        ("run_tests", {"target": "tests/test_main.py"}),
        ("edit_file", {"path": "src/main.py", "patch": "fix off-by-one"}),
        ("run_tests", {"target": "tests/test_main.py"}),
        ("git_commit", {"message": "fix guard and off-by-one"}),
    ]
    for i, (tool, args) in enumerate(plan):
        yield ToolEvent(tool=tool, args=args, ts=float(i))


def _run_naive(agent: Iterator[ToolEvent], cap: int) -> int:
    """Without Pathlayer: the agent runs until the cap (or until it stops on its own)."""
    executed = 0
    for _ in agent:
        executed += 1
        if executed >= cap:
            break
    return executed


def _run_guarded(
    agent: Iterator[ToolEvent], logger: logging.Logger
) -> tuple[Pipeline, DemoRunControl, int]:
    """With Pathlayer: each step goes through ``wire()``; STOP drains the agent."""
    pipeline = Pipeline([LoopDetector()], logger=logger)
    control = DemoRunControl()
    sink = wire(pipeline, DemoActuator(control), logger=logger)
    executed = 0
    for event in agent:
        sink(event)
        executed += 1
        if control.stopped:
            break
        if executed >= NAIVE_CAP:  # safety net in case detection failed
            break
    return pipeline, control, executed


def _triggering_detector(pm: PostMortem) -> str:
    if not pm.signals:
        return "?"
    return max(pm.signals, key=lambda s: s.confidence).name


def scenario_loop(logger: logging.Logger) -> None:
    print("--- LOOP: stuck agent ---")
    naive = _run_naive(looping_agent(), NAIVE_CAP)
    pipeline, control, _ = _run_guarded(looping_agent(), logger)
    pm = pipeline.post_mortem

    print(f"Without Pathlayer: {naive} actions and still going (demo cap {NAIVE_CAP}).")
    if pm is None:
        print("With Pathlayer:    nothing detected (UNEXPECTED for this trace).\n")
        return
    print(f"With Pathlayer:    STOP after {pm.events_processed} actions.")
    print(f"   Triggering detector : {_triggering_detector(pm)} ({pm.reason})")
    # stopped_at_index is zero-based; people count actions from 1.
    print(f"   Detected at         : action {pm.stopped_at_index + 1}")
    print(f"   Actuator            : {'drain requested' if control.stopped else 'not triggered'}")
    avoided = naive - pm.events_processed
    print(f"   Work avoided        : {avoided} of {naive} actions.\n")


def scenario_healthy(logger: logging.Logger) -> None:
    print("--- HEALTHY: agent making progress (counterexample) ---")
    pipeline, control, executed = _run_guarded(healthy_agent(), logger)

    if pipeline.stopped or control.stopped:
        pm = pipeline.post_mortem
        reason = pm.reason if pm is not None else "?"
        print(f"With Pathlayer:    STOP after {executed} actions (FALSE POSITIVE: {reason}).\n")
        return
    print(
        f"With Pathlayer:    {executed} actions, no intervention. The agent finishes on its own.\n"
    )


def main() -> None:
    # Signal / decision / action logs go to stderr as JSON lines, to show exactly
    # when and why Pathlayer acts without cluttering the demo output.
    logger = get_logger("pathlayer.demo")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    print("=" * 60)
    print("PATHLAYER DEMO - a loop detected and stopped")
    print("=" * 60 + "\n")
    scenario_loop(logger)
    scenario_healthy(logger)


if __name__ == "__main__":
    main()
