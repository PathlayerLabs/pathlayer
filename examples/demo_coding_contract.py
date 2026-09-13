"""ANC-4 -- a declarative coding contract on a real LangGraph agent.

A scripted coding agent (no LLM, no network) works in a throwaway workspace:
``calculator.py`` starts with four unimplemented functions and
``tests/test_calculator.py`` has one test for each. The operator describes
progress in five lines of YAML, compiles them once, and runs the agent through
a real compiled LangGraph ``StateGraph`` with the *default* Pathlayer detectors::

    task: implement-calculator
    objective: Implement the four calculator operations so their tests pass.
    progress:
      - type: tests_improve
        path: tests/test_calculator.py
      - type: file_changes
        path: calculator.py

    pathlayer-compile-contract -> proof_spec.json + probe.py
    LangGraphAdapter(progress_proof=...) -> probe after each tool call
    build_pipeline(config) -> NoProgressDetector(+ProofProgressAnchor) -> STOP

The agent's own tools return unstructured text, as a shell tool would, so the
default coding anchors have nothing to measure: without the contract Pathlayer
cannot tell a productive run from a stuck one. With it, the generated probe runs
the real tests and checks the real file after every tool call.

Six scenarios are fixed in advance; production thresholds are untouched:

- ``productive``: each function takes a wrong attempt, then a fix -- the same
  edit/test rhythm repeated, with a test turning green every four calls;
- ``stagnant``: busy but stuck -- a different wrong edit every time; the file
  changes (an observed effect) but no test ever passes;
- ``oscillating``: fixes ``add``, breaks it, fixes it again, with varied edits;
- ``objective_reached``: all four functions done early, then unrelated notes;
- ``probe_unavailable``: the stuck agent again, but the probe runs under an
  interpreter that cannot import Pathlayer, so no proof is ever produced;
- ``unobservable_mid_run``: the productive agent, except that after two
  functions it breaks ``calculator.py``'s syntax for ten calls (a refactor in
  progress): the test module can no longer be imported, so the probe observes
  the file but not one test. Before this scenario existed those ten calls
  froze the last level and ended in a STOP; they are now gaps.

Each scenario is checked four ways: the live run (enforced through
``LangGraphActuator``), a replay of the recorded trace after the workspace is
deleted, the same events through the hand-written low-level ANC path (a
manifest and a probe replaying the recorded checkpoints), and an ablation with
the proofs removed under the default configuration.

Requires the ``langgraph`` and ``yaml`` extras. Run::

    python examples/demo_coding_contract.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.tools import tool
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph

from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.coding_contracts import (
    InstalledCodingContract,
    load_coding_contract,
    write_coding_contract,
)
from pathlayer.config import PathlayerConfig, build_pipeline
from pathlayer.detectors.proof import PROOF_METADATA_KEYS
from pathlayer.detectors.proof_producer import build_producer, producer_from_config
from pathlayer.events import ToolEvent
from pathlayer.pipeline import PipelineDecision
from pathlayer.policies import Action
from pathlayer.policies.langgraph_actuator import LangGraphActuator
from pathlayer.policies.wiring import wire
from pathlayer.validation.traces import Trace, dump_trace, load_trace

CONTRACT_YAML = """\
task: implement-calculator
objective: Implement the four calculator operations so their tests pass.
workspace: workspace
timeout_seconds: 20
progress:
  - type: tests_improve
    path: tests/test_calculator.py
  - type: file_changes
    path: calculator.py
"""

_TESTS = """\
from calculator import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(4, 3) == 12


def test_divide():
    assert divide(8, 2) == 4
"""

_FUNCTIONS = ("add", "subtract", "multiply", "divide")
_CORRECT = {
    "add": "return a + b",
    "subtract": "return a - b",
    "multiply": "return a * b",
    "divide": "return a / b",
}

Call = tuple[str, dict[str, Any]]


def _edit(name: str, body: str) -> Call:
    return ("edit_function", {"name": name, "body": body})


def _run_tests() -> Call:
    return ("run_tests", {})


def _stuck_script() -> list[Call]:
    script: list[Call] = []
    for attempt in range(8):
        script += [_edit("add", f"return a * b * {attempt + 2}"), _run_tests()]  # never 5
    return script


def _productive_script() -> list[Call]:
    script: list[Call] = []
    for name in _FUNCTIONS:
        script += [_edit(name, "return None"), _run_tests()]
        script += [_edit(name, _CORRECT[name]), _run_tests()]
    return script


def _oscillating_script() -> list[Call]:
    fixes = ("return a + b", "return b + a", "return (a + b)", "return sum((a, b))")
    breaks = ("return a - b", "return a * b", "return b - a", "return a + b + 1")
    script: list[Call] = []
    for fix, broken in zip(fixes, breaks, strict=True):
        script += [_edit("add", fix), _run_tests(), _edit("add", broken), _run_tests()]
    return script


def _unobservable_mid_run_script() -> list[Call]:
    script: list[Call] = []
    for name in _FUNCTIONS[:2]:
        script += [_edit(name, "return None"), _run_tests()]
        script += [_edit(name, _CORRECT[name]), _run_tests()]
    # A refactor in progress: the file changes on every call but cannot be
    # imported, so no test can be collected for ten consecutive calls.
    for attempt in range(5):
        script += [_edit("multiply", f"return a * (b  # draft {attempt}"), _run_tests()]
    for name in _FUNCTIONS[2:]:
        script += [_edit(name, _CORRECT[name]), _run_tests()]
    return script


def _objective_reached_script() -> list[Call]:
    script = [_edit(name, _CORRECT[name]) for name in _FUNCTIONS] + [_run_tests()]
    return script + [("write_note", {"text": f"release note {index}"}) for index in range(11)]


SCENARIOS: dict[str, Callable[[], list[Call]]] = {
    "productive": _productive_script,
    "stagnant": _stuck_script,
    "oscillating": _oscillating_script,
    "objective_reached": _objective_reached_script,
    "probe_unavailable": _stuck_script,
    "unobservable_mid_run": _unobservable_mid_run_script,
}


@dataclass(frozen=True)
class ScenarioResult:
    """Everything the demo claims about one scenario, measured."""

    name: str
    events: int
    live_stop: int | None
    no_progress_max: float
    replay_identical: bool
    low_level_identical: bool
    ablation_stop: int | None
    proofs: int
    unobserved: int
    warnings: tuple[str, ...]
    baseline_events: int
    baseline_stop: int | None


def _render_calculator(bodies: dict[str, str]) -> str:
    return "\n\n".join(f"def {name}(a, b):\n    {bodies[name]}\n" for name in _FUNCTIONS)


def prepare_workspace(root: Path) -> Path:
    workspace = root / "workspace"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "calculator.py").write_text(
        _render_calculator(dict.fromkeys(_FUNCTIONS, "raise NotImplementedError")), "utf-8"
    )
    (workspace / "tests" / "test_calculator.py").write_text(_TESTS, encoding="utf-8")
    return workspace


def install_contract(root: Path) -> InstalledCodingContract:
    """What an operator does once, before the run (the CLI does the same)."""
    contract_path = root / "contract.yaml"
    contract_path.write_text(CONTRACT_YAML, encoding="utf-8")
    return write_coding_contract(load_coding_contract(contract_path), output_dir=root / "bundle")


class _AgentState(TypedDict):
    step: int


def run_live(
    workspace: Path, config: PathlayerConfig, script: Sequence[Call]
) -> tuple[list[ToolEvent], list[PipelineDecision | None], bool]:
    pipeline = build_pipeline(config)
    actuator = LangGraphActuator()
    sink = wire(pipeline, actuator)
    events: list[ToolEvent] = []
    decisions: list[PipelineDecision | None] = []

    def observe(event: ToolEvent) -> None:
        events.append(event)
        sink(event)
        decisions.append(pipeline.last_decision)

    adapter = LangGraphAdapter(observe, progress_proof=producer_from_config(config))
    adapter.attach()
    bodies = dict.fromkeys(_FUNCTIONS, "raise NotImplementedError")

    @tool
    def edit_function(name: str, body: str) -> str:
        """Replace the body of one calculator function."""
        bodies[name] = body
        (workspace / "calculator.py").write_text(_render_calculator(bodies), encoding="utf-8")
        return f"edited {name}"

    @tool
    def run_tests() -> str:
        """Run the test suite (the scripted agent does not read the output)."""
        return "tests executed"

    @tool
    def write_note(text: str) -> str:
        """Append a line to the release notes."""
        with (workspace / "NOTES.md").open("a", encoding="utf-8") as notes:
            notes.write(text + "\n")
        return "noted"

    tools = {"edit_function": edit_function, "run_tests": run_tests, "write_note": write_note}

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
            {"step": 0},
            config={**adapter.as_config(), "recursion_limit": 4 * len(script)},
            control=actuator.control,
        )
    except GraphDrained:
        drained = True
    assert drained == pipeline.stopped == actuator.control.drain_requested
    return events, decisions, drained


def _without_proofs(event: ToolEvent) -> ToolEvent:
    metadata = {
        key: value for key, value in event.metadata.items() if key not in PROOF_METADATA_KEYS
    }
    return replace(event, metadata=metadata)


def _decide(config: PathlayerConfig, events: Sequence[ToolEvent]) -> list[PipelineDecision | None]:
    pipeline = build_pipeline(config)
    decisions = []
    for event in events:
        pipeline.process(event)
        decisions.append(pipeline.last_decision)
    return decisions


def _first_stop(decisions: Sequence[PipelineDecision | None]) -> int | None:
    for index, decision in enumerate(decisions, 1):
        if decision is not None and decision.arbitrated_action is Action.STOP:
            return index
    return None


def _recorded_observation(event: ToolEvent) -> tuple[dict[str, bool] | None, int]:
    """What the recorded probe reported: its checkpoints, or the exit status it
    failed with (the producer keeps that status in the marker's reason)."""
    proof = event.metadata.get("progress_proof")
    if proof is not None:
        return proof["checkpoints"], 0
    reason: str = event.metadata["progress_proof_unobserved"]["reason"]
    status = reason.removeprefix("exited with status ")
    return None, int(status) if status.isdigit() else 4


def low_level_decisions(
    root: Path, installed: InstalledCodingContract, events: Sequence[ToolEvent]
) -> list[PipelineDecision | None]:
    """The same events through a hand-written ANC-1 manifest and ANC-2 probe."""
    spec = installed.compiled.bundle.proof_spec
    manifest = root / "low-level" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": spec.task_id,
                "proof_id": spec.proof_id,
                "scope_id": spec.scope_id,
                "kind": "checkpoint_set",
                "required": list(spec.required),
            }
        ),
        encoding="utf-8",
    )
    schedule = manifest.parent / "schedule.json"
    schedule.write_text(json.dumps([_recorded_observation(event) for event in events]))
    probe = manifest.parent / "probe.py"
    probe.write_text(
        "import json, pathlib, sys\n"
        f"state = pathlib.Path({str(manifest.parent / 'count')!r})\n"
        "count = int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(count + 1))\n"
        f"checkpoints, status = json.loads(pathlib.Path({str(schedule)!r}).read_text())[count]\n"
        "if checkpoints is None:\n    sys.exit(status)\n"
        "print(json.dumps({'checkpoints': checkpoints}))\n",
        encoding="utf-8",
    )
    config = PathlayerConfig.from_mapping(
        {
            "detectors": {
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": str(manifest),
                    "probe_command": [sys.executable, str(probe)],
                }
            }
        }
    )
    producer = build_producer(str(manifest), (sys.executable, str(probe)), 10.0)
    pipeline = build_pipeline(config)
    decisions = []
    for event in events:
        replayed = _without_proofs(event)
        producer.attach(replayed)
        pipeline.process(replayed)
        decisions.append(pipeline.last_decision)
    return decisions


def run_scenario(name: str, root: Path) -> ScenarioResult:
    root.mkdir(parents=True)
    workspace = prepare_workspace(root)
    installed = install_contract(root)
    contract_section = installed.progress_proof_config()
    if name == "probe_unavailable":
        # "-S" skips site-packages: the probe cannot import pathlayer.
        contract_section["probe_command"] = [sys.executable, "-S", str(installed.paths.probe)]
    config = PathlayerConfig.from_mapping({"detectors": {"progress_proof": contract_section}})

    with _ProducerWarnings() as warnings:
        events, decisions, _ = run_live(workspace, config, SCENARIOS[name]())

    trace_path = root / "trace.json"
    dump_trace(Trace(label="unknown", events=events), trace_path)
    shutil.rmtree(workspace)  # replay reads recorded proofs only; nothing left to probe
    replayed = _decide(config, load_trace(trace_path).events)
    ablation = _decide(PathlayerConfig(), [_without_proofs(event) for event in events])
    with _ProducerWarnings():  # the low-level replay repeats the same probe failures
        low_level = low_level_decisions(root, installed, events)
    # Today's behaviour: the same agent, full script, default config, no contract.
    baseline_events, baseline_decisions, _ = run_live(
        prepare_workspace(root / "baseline"), PathlayerConfig(), SCENARIOS[name]()
    )
    return ScenarioResult(
        name=name,
        events=len(events),
        live_stop=_first_stop(decisions),
        no_progress_max=max(
            (
                signal.confidence
                for decision in decisions
                if decision is not None
                for signal in decision.signals
                if signal.name == "no_progress"
            ),
            default=0.0,
        ),
        replay_identical=replayed == decisions,
        low_level_identical=low_level == decisions,
        ablation_stop=_first_stop(ablation),
        proofs=sum("progress_proof" in event.metadata for event in events),
        unobserved=sum("progress_proof_unobserved" in event.metadata for event in events),
        warnings=tuple(warnings.messages),
        baseline_events=len(baseline_events),
        baseline_stop=_first_stop(baseline_decisions),
    )


class _ProducerWarnings(logging.Handler):
    """Collect the producer's explicit collection-failure warnings for the table."""

    _LOGGER = logging.getLogger("pathlayer.detectors.proof_producer")

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> _ProducerWarnings:
        self._LOGGER.addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._LOGGER.removeHandler(self)


def run_all(root: Path) -> list[ScenarioResult]:
    return [run_scenario(name, root / name) for name in SCENARIOS]


def _outcome(stop: int | None, events: int) -> str:
    return f"STOP at call {stop}" if stop is not None else f"no action ({events} calls)"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pathlayer-anc4-demo-") as scratch:
        results = run_all(Path(scratch))
    header = (
        f"{'scenario':<20} {'with contract':<22} {'proofs':>6} {'gaps':>4} {'warnings':>8} "
        f"{'replay':>7} {'low-level':>9}  without contract"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<20} {_outcome(result.live_stop, result.events):<22} "
            f"{result.proofs:>6} {result.unobserved:>4} {len(result.warnings):>8} "
            f"{'same' if result.replay_identical else 'DIFF':>7} "
            f"{'same' if result.low_level_identical else 'DIFF':>9}  "
            f"{_outcome(result.baseline_stop, result.baseline_events)}"
        )
    stops = [result.name for result in results if result.ablation_stop is not None]
    print(
        "\nAblation (the same recorded calls, proofs removed, default config) stops in: "
        + (", ".join(stops) if stops else "no scenario")
    )
    for result in results:
        if result.warnings:
            print(f"{result.name}: e.g. {result.warnings[0]!r}")


if __name__ == "__main__":
    main()
