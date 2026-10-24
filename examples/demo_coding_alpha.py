"""A coding contract and the coding state preset on a real scripted LangGraph agent.

Run ``python examples/demo_coding_alpha.py``. No LLM or network is used.
The agent runs pytest itself; the independent contract probe verifies its objective.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph

from examples.demo_coding_contract import (
    _FUNCTIONS,
    Call,
    _decide,
    _first_stop,
    _productive_script,
    _render_calculator,
    _stuck_script,
    append_new_decision,
    install_contract,
    prepare_workspace,
)
from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.coding_state import state_evidence_producer_from_config
from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.proof_producer import producer_from_config
from pathlayer.events import ToolEvent
from pathlayer.pipeline import PipelineDecision
from pathlayer.policies.langgraph_actuator import LangGraphActuator
from pathlayer.policies.wiring import wire
from pathlayer.postmortem import TickRecord, build_coding_diagnostic
from pathlayer.validation.traces import Trace, dump_trace, load_trace

SCENARIOS = (
    "productive",
    "stagnant",
    "activity_without_success",
    "collection_failed",
    "unavailable",
)


class _AgentState(TypedDict):
    step: int


def run_live(
    workspace: Path, config: PathlayerConfig, script: Sequence[Call]
) -> tuple[list[ToolEvent], list[PipelineDecision], bool]:
    pipeline = build_pipeline(config)
    actuator = LangGraphActuator()
    sink = wire(pipeline, actuator)
    events: list[ToolEvent] = []
    decisions: list[PipelineDecision] = []

    def observe(event: ToolEvent) -> None:
        events.append(event)
        sink(event)
        append_new_decision(decisions, pipeline.last_decision)

    adapter = LangGraphAdapter(
        observe,
        state_evidence=state_evidence_producer_from_config(config),
        progress_proof=producer_from_config(config),
    )
    adapter.attach()
    bodies = dict.fromkeys(_FUNCTIONS, "raise NotImplementedError")

    @tool
    def edit_function(name: str, body: str, file_path: str = "calculator.py") -> str:
        """Replace the body of one calculator function."""
        bodies[name] = body
        (workspace / "calculator.py").write_text(_render_calculator(bodies), encoding="utf-8")
        return f"edited {name}"

    @tool
    def run_tests() -> str:
        """Run the test suite (the scripted agent does not read the output)."""
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return f"pytest exit status: {completed.returncode}"

    @tool
    def write_note(text: str) -> str:
        """Append a line to the release notes."""
        with (workspace / "NOTES.md").open("a", encoding="utf-8") as notes:
            notes.write(text + "\n")
        return "noted"

    @tool
    def edit_batch(name: str, body: str, file_path: str = "calculator.py") -> str:
        """Delegate an edit to a nested tool, making the capture boundaries overlap."""
        return str(edit_function.invoke({"name": name, "body": body, "file_path": file_path}))

    tools = {
        "edit_function": edit_function,
        "run_tests": run_tests,
        "write_note": write_note,
        "edit_batch": edit_batch,
    }

    def act(state: _AgentState) -> _AgentState:
        name, args = script[state["step"]]
        tools[name].invoke(args)
        return {"step": state["step"] + 1}

    graph = StateGraph(_AgentState)
    graph.add_node("act", act)
    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", lambda s: END if s["step"] >= len(script) else "act")
    drained = False
    try:
        graph.compile().invoke(
            _AgentState(step=0),
            config=cast(
                RunnableConfig, {**adapter.as_config(), "recursion_limit": 4 * len(script)}
            ),
            control=actuator.control,
        )
    except GraphDrained:
        drained = True
    assert drained == pipeline.stopped == actuator.control.drain_requested
    return events, decisions, drained


def run_scenario(name: str, root: Path) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    root.mkdir(parents=True)
    workspace = prepare_workspace(root)
    installed = install_contract(root)
    script = (
        _productive_script()
        if name == "productive"
        else _stuck_script()
        if name == "activity_without_success"
        else [("edit_function", {"name": "add", "body": "return None"})] * 16
    )
    script = [
        (tool_name, {**args, "file_path": "calculator.py"})
        if tool_name == "edit_function"
        else (tool_name, args)
        for tool_name, args in script
    ]
    if name == "unavailable":
        script = [("edit_batch", args) for _, args in script]
    state: dict[str, Any] = {
        "enabled": True,
        "workspace": str(workspace),
        "git": False,
        "tools": {
            name: {"effect": "write", "paths": ["file_path"]}
            for name in ("edit_function", "edit_batch")
        },
    }
    detectors = {"state_evidence": state}
    if name not in ("collection_failed", "unavailable"):
        detectors["progress_proof"] = installed.progress_proof_config()
    elif name == "collection_failed":
        state["build"] = {"id": "build", "command": [str(root / "missing-validator")]}
    config = PathlayerConfig.from_mapping({"detectors": detectors})
    events, decisions, _ = run_live(workspace, config, script)
    dump_trace(
        Trace(label="unknown", events=events, description=f"coding alpha scenario: {name}"),
        root / "trace.json",
    )
    shutil.rmtree(workspace)
    replay = _decide(config, load_trace(root / "trace.json").events)
    baseline_workspace = prepare_workspace(root / "baseline")
    _, baseline, _ = run_live(baseline_workspace, PathlayerConfig(), script)
    diagnostic = build_coding_diagnostic(events, [TickRecord.from_decision(d) for d in decisions])
    return {
        "scenario": name,
        "stop": _first_stop(decisions),
        "baseline_stop": _first_stop(baseline),
        "replay_identical": replay == decisions,
        "diagnostic": diagnostic.to_dict(),
        "no_progress_max": max(
            (
                signal.confidence
                for decision in decisions
                if decision is not None
                for signal in decision.signals
                if signal.name == "no_progress"
            ),
            default=0.0,
        ),
        "explanation": diagnostic.format(),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pathlayer-alpha-") as tmp:
        for name in SCENARIOS:
            result = run_scenario(name, Path(tmp) / name)
            print(f"\n[{name}]\n{result['explanation']}")
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ("diagnostic", "explanation")},
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
