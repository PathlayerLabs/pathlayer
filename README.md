# Pathlayer

> Runtime guardrails for AI agents. Open source, local-first, with no LLM in
> the detection loop.

Pathlayer observes an agent while it runs, detects loops, repeated failures,
missing progress and cost explosions, then selects a graded action:
`warn -> summarize -> fallback -> escalate -> stop`. LangGraph currently
enforces WARN, SUMMARIZE and STOP; FALLBACK and ESCALATE are planned.

Pathlayer is alpha software. Its detectors are deterministic and run locally,
without a cloud account, API key or network call. Progress detection is
strongest when an agent produces objective evidence such as test results, exit
codes or changes in Git diff size.

## Install

When a release is available on PyPI, install it with:

```bash
pip install pathlayer
```

Optional integrations are installed separately:

```bash
pip install "pathlayer[yaml]"       # YAML configuration
pip install "pathlayer[langgraph]"  # LangGraph capture and actuation
```

To test an unreleased source checkout, build and install its wheel instead:

```bash
uv build
pip install dist/pathlayer-*.whl
```

Pathlayer requires Python 3.10 or later; release CI currently tests Python
3.10, 3.11 and 3.12.

## Quickstart

This example uses the dependency-free core. A small agent repeatedly lists and
reads the same file. The adapter observes each action, the default pipeline
detects the loop, and a cooperative actuator stops the agent for real.

Construction and runtime are separate:

```text
load_config -> build_pipeline -> actuator -> wire -> adapter
agent -> adapter -> sink -> pipeline -> actuator -> agent
```

<!-- pathlayer-quickstart:start -->
```python
from pathlayer import PostMortem, build_pipeline, load_config
from pathlayer.adapters import InMemoryAdapter
from pathlayer.events import ToolEvent
from pathlayer.policies import Action, Actuator, wire


class CooperativeStopActuator(Actuator):
    def __init__(self) -> None:
        self.stopped = False
        self.post_mortem: PostMortem | None = None

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        if action is Action.STOP:
            self.stopped = True
            self.post_mortem = post_mortem


pipeline = build_pipeline(load_config())
actuator = CooperativeStopActuator()
adapter = InMemoryAdapter(wire(pipeline, actuator))

for step in range(100):
    tool = ("list_dir", "read_file")[step % 2]
    adapter.feed(
        ToolEvent(
            tool=tool,
            args={"path": "src/main.py"},
            ts=float(step),
        )
    )
    if actuator.stopped:
        break

assert actuator.stopped
assert actuator.post_mortem is not None
print(
    f"Pathlayer stopped the loop after "
    f"{actuator.post_mortem.events_processed} actions: "
    f"{actuator.post_mortem.reason}"
)
```
<!-- pathlayer-quickstart:end -->

`build_pipeline()` is the recommended entry point. Besides constructing the
enabled detectors, it wires the coding progress anchors, loop suppression when
fresh progress exists, and the immediate budget-stop override. Constructing a
bare `Policy()` does not add those safeguards automatically, and calling
`Policy.decide()` directly also bypasses the graded ladder and hysteresis.

## Configuration

No file is required: `load_config()` returns safe built-in defaults. If present,
Pathlayer discovers exactly one of `.pathlayer/config.json`,
`.pathlayer/config.yaml` or `.pathlayer/config.yml` from the current project.
JSON uses only the Python standard library; YAML requires `pathlayer[yaml]`.

Copy [`examples/pathlayer.config.yaml`](examples/pathlayer.config.yaml) for the
complete commented reference. Override only the values your traces justify.
For example, budget enforcement is disabled by default:

```yaml
detectors:
  budget:
    enabled: true
    max_cost_usd: 5.0
    default_cost_usd: 0.01
    tool_costs_usd:
      train_model: 2.5
```

Captured positive costs always win over fallback estimates.

## LangGraph integration

Install `pathlayer[langgraph]`, then connect capture and actuation around your
compiled graph:

```python
from langgraph.errors import GraphDrained

from pathlayer import build_pipeline, load_config
from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.policies import wire
from pathlayer.policies.langgraph_actuator import LangGraphActuator

pipeline = build_pipeline(load_config())
actuator = LangGraphActuator()
actuator.attach()
sink = wire(pipeline, actuator)
adapter = LangGraphAdapter(sink)
adapter.attach()

try:
    result = graph.invoke(  # your compiled graph and initial state
        initial_state,
        config=adapter.as_config(),
        control=actuator.control,
    )
except GraphDrained as stopped:
    print(f"Pathlayer stopped the run: {stopped.reason}")
finally:
    adapter.detach()
```

The fragment assumes `graph` and `initial_state` already exist. For WARN to
reach the model, add `actuator.injection_node()` before the model node.
SUMMARIZE additionally requires `summary_tool()`, `summary_gate_node(sink)` and
the conditional route returned by `summary_route()`. The complete runnable
wiring lives in [`examples/demo_no_progress.py`](examples/demo_no_progress.py).

## Development

Install a source checkout with its development dependencies in a virtual
environment using Python 3.10 or later:

```bash
python -m pip install -e ".[dev]"
```

On Python 3.12+, an opt-in line-coverage profile speeds up the local edit-run
loop by using `sys.monitoring`:

```bash
COVERAGE_CORE=sysmon pytest --cov-config=.coveragerc.fast --cov-fail-under=0
```

This fast profile does not measure branch coverage and is not the project gate.
Before submitting or merging a change, run plain `pytest` and the complete
validation gate documented in [`AGENTS.md`](AGENTS.md#validation).

## Demos

- [`examples/demo_loop.py`](examples/demo_loop.py): a repeated loop stopped
  through the core runtime path, plus a healthy counterexample.
- [`examples/demo_no_progress.py`](examples/demo_no_progress.py): a real,
  deterministic LangGraph run that receives WARN, SUMMARIZE and STOP while
  repetition detectors remain silent.
- [`docs/demo-2.md`](docs/demo-2.md): a non-technical walkthrough of the
  no-progress demo and what it proves.

The no-progress demo relies on deterministic ground truth from tests. Detecting
slow burns with no tests, exit codes, code evidence or other objective signal is
outside this V1 claim.

## License

Apache-2.0.
