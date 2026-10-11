# Pathlayer

> Runtime guardrails for reliable and efficient AI agents.

AI agents can waste time, tokens, and compute when they get stuck in loops,
repeat failed actions, or continue executing without meaningful progress.

Pathlayer is an open-source, local-first runtime layer that detects unproductive
agent behavior and intervenes before more resources are wasted. Unlike
LLM-based monitoring, Pathlayer uses deterministic detectors and external
progress evidence, including filesystem changes and test results, without
requiring another LLM in the detection loop.

Pathlayer observes agents during execution, detects loops, repeated failures,
missing progress, and cost overruns, then selects a graded action:
`warn -> summarize -> fallback -> escalate -> stop`. LangGraph currently
enforces WARN, SUMMARIZE, and STOP; FALLBACK and ESCALATE are planned.

Pathlayer is alpha software. Its detectors run locally without a cloud account,
API key, or network call. Progress detection is strongest when an agent
produces objective evidence such as test results, exit codes, or changes in
Git diff size.

## See Pathlayer in action

**A code change is not necessarily progress.** Pathlayer uses independently
checked test results to distinguish productive retries from repeated activity
that does not advance a task.

The [coding contract demonstration](examples/demo_coding_contract.py) runs a
scripted LangGraph coding agent with real file edits and test checks. It
compares execution with and without an external progress contract:

| Scenario | What the demo tests |
| --- | --- |
| Productive retries | Repeated edits and tests with measurable improvements should continue. |
| Stagnation | Edits that do not improve tests can trigger a STOP. |
| Unavailable evidence | Missing proof is treated as an observation gap, not as failure. |
| Deterministic replay | Recorded events are replayed and compared with live decisions. |

Run it from a source checkout using the [installation steps below](#install):

```bash
python examples/demo_coding_contract.py
```

The output compares **with contract** and **without contract** for each
scenario, including STOP decisions, evidence gaps, and replay agreement.
The demo runs a reproducible coding-agent workflow through LangGraph, using real
file operations and test results to validate Pathlayer's detection, intervention,
and replay capabilities. It runs entirely locally, with no LLM API or network dependency.

For a simpler loop example, see
[`demo_loop.py`](examples/demo_loop.py). For graded
WARN → SUMMARIZE → STOP intervention, see
[`demo_no_progress.py`](examples/demo_no_progress.py).

## Install

Pathlayer is currently alpha software. The most direct way to try the demos
is from a local source checkout (Python 3.10+):

```bash
git clone https://github.com/Kropemann/Pathlayer.git
cd Pathlayer
python -m pip install -e ".[dev]"
python examples/demo_coding_contract.py
```

Repository access is required while the source repository is private. The
development extra includes the dependencies needed for the demos; for an
application integration, install only the optional extras you need:

```bash
python -m pip install -e ".[langgraph,yaml]"
```

For a packaged source installation instead of editable mode, build a wheel
and install it:

```bash
uv build
python -m pip install dist/pathlayer-*.whl
```

Once a release is published to PyPI, the package can be installed with
`pip install pathlayer` (and optional extras such as
`pathlayer[langgraph]`).

Release CI currently tests Python 3.10, 3.11, and 3.12.

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


config = load_config()
pipeline = build_pipeline(config)
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

Captured positive costs always win over fallback estimates. Cost accounting
uses the same convention throughout Pathlayer: `cost_usd > 0` is a captured,
priced event; `cost_usd == 0` means that no captured price is available and may
therefore use a configured estimate when BUD is enabled.

Duration enforcement is also opt-in and requires the live run-aware sink:

```yaml
detectors:
  resource:
    enabled: true
    max_duration_seconds: 14400
```

`wire()` returns a callable sink with `start_run()` and `end_run()`. LangGraph
starts it on the first root callback and OpenClaw starts it on
`before_agent_run`; custom adapters should call those lifecycle methods at the
real run boundaries. If they do not, the first event starts the watchdog with a
warning, and time before that event cannot be measured. The detector remains
pure: durable `run_start` and `run_timeout` events carry the lifecycle through
SQLite and exact replay. The configured duration is a hard ceiling, not a
confidence ramp: the duration signal is `0` before the deadline and `1` at it.
`build_pipeline()` auto-wires that ceiling to STOP so one timeout bypasses
normal multi-tick hysteresis; an explicit resource override or advisory mode
still wins.
The shipped value is an uncalibrated default and the entire resource detector
is disabled until explicitly enabled. `policy.advisory_signals`, dry-run mode
and the actuator's real capabilities remain authoritative over enforcement.

RSS memory tracking shares the same `resource` section:

```yaml
detectors:
  resource:
    enabled: true
    max_rss_bytes: 2147483648
```

Each observed event is stamped with the process's current resident set size,
read live from `/proc/self/status` on Linux; on platforms without `/proc`
(e.g. macOS, Windows) the signal silently stays at `0` -- no hard dependency
on a platform API, per the detector's "gracefully unavailable" contract.

### Progress proofs for tasks without native ground truth

When tests, exit codes and code-state evidence cannot define success, generate
an explicit `checkpoint_set` contract and an external probe skeleton:

```bash
pathlayer-scaffold-proof \
  --task-id customer-import-42 \
  --name "Accepted deliverables" \
  --objective "Import, validate and accept the customer dataset" \
  --checkpoint file_created \
  --checkpoint schema_valid \
  --checkpoint accepted \
  --output-dir /path/owned/by/operator/customer-import-42
```

The command writes a validated `proof_spec.json`, an intentionally inert
`probe.py`, and a local `README.md` with the opt-in configuration. It refuses
to overwrite any of those files. Implement the probe against state outside the
observed agent's control; until then it raises `NotImplementedError`, ANC-2
emits no proof and detection stays inert.

Keep the manifest, probe code and state it reads outside the observed agent's
write boundary, or enforce equivalent integration controls. An agent that can
edit its own probe can forge success or stagnation, so that producer is not a
valid trust boundary.

The same deterministic renderer and safe writer are available in Python:

```python
from pathlayer.proof_scaffold import (
    render_progress_proof_scaffold,
    write_progress_proof_scaffold,
)

bundle = render_progress_proof_scaffold(
    task_id="customer-import-42",
    name="Accepted deliverables",
    objective="Import, validate and accept the customer dataset",
    checkpoints=["file_created", "schema_valid", "accepted"],
)
assert bundle.proof_spec.required == ("accepted", "file_created", "schema_valid")

paths = write_progress_proof_scaffold(
    output_dir="/path/owned/by/operator/customer-import-42",
    task_id="customer-import-42",
    name="Accepted deliverables",
    objective="Import, validate and accept the customer dataset",
    checkpoints=["file_created", "schema_valid", "accepted"],
)
```

`proof_id` defaults to `name` and `scope_id` to `task_id`; both APIs and the
CLI accept explicit overrides. Names and objectives are descriptive only and
never determine checkpoint values. For coding tasks, the contract library below
generates a working probe instead of a skeleton; Pathlayer still provides no
automatic contract selection, semantic inference, or ready-made business checks.

### Coding progress contracts

For coding agents, declare what progress means in a few lines instead of
writing a probe. JSON works too; YAML needs `pathlayer[yaml]`:

```yaml
task: fix-login
objective: Make the login flow pass its tests.
progress:
  - type: tests_improve
    path: tests/test_login.py
  - type: file_changes
    path: src/auth.py
```

```bash
pathlayer-compile-contract contract.yaml --output-dir /path/owned/by/operator/fix-login
```

The compiler validates the contract, captures `file_changes` baselines,
expands `tests_improve` selections to exact test ids, and writes an ordinary
progress-proof bundle: `proof_spec.json`, a generated `probe.py` and a
`README.md`. It then runs the probe once to print the current state, and prints
the `detectors.progress_proof` section to add to your config. A malformed
contract fails here, before any run. The workspace defaults to the contract
file's directory.

| Type | Checkpoint is true when | Level |
| --- | --- | --- |
| `file_exists` (`path`) | a regular file exists at `path` | objective |
| `artifact_produced` (`path`, `min_bytes: 1`, optional `sha256`) | that file has at least `min_bytes`, and the given SHA-256 | objective |
| `test_passes` (`test`) | this exact pytest nodeid passes | objective |
| `tests_improve` (`tests: [...]` or `path`) | one checkpoint per selected test | progress toward the objective |
| `command_succeeds` (`id`, `command: [...]`) | the command, run without a shell, exits 0 | objective |
| `file_changes` (`path`) | the file differs from its content at compile time | observed effect |

Every checkpoint counts once and weighs the same. `no_progress` restarts its
patience each time the number of true checkpoints reaches a new high, and stays
silent once all of them are true: the objective is reached. An observed effect
never reaches the objective on its own, so a contract made only of
`file_changes` is rejected. A test that is deleted, skipped or flips back and
forth never counts as new progress.

A check that cannot observe its resource -- pytest missing or timing out, a
test module that cannot be imported, a directory where a file was expected, the
probe's time budget exhausted -- is omitted for that tool call rather than
reported false. Pathlayer logs a warning, keeps the checkpoint's last known
value for later, and treats that tool call as a *gap*: `no_progress` stays
silent on it, even if its other signals (test results, exit codes, git diff)
have been flat for a while, and restarts its patience afterwards. A probe that
keeps failing in the middle of a run therefore never causes a STOP on its own.
The price is recall: an agent that stays stuck while its probe keeps failing is
left to the other detectors.

Tests run once per tool call, in a single pytest session with the installed
plugins and the project's ini options, except `addopts`, which is ignored so
that options such as `-x` cannot hide a required test. When the project has its
own environment, name its interpreter in the contract, relative to the
workspace or absolute:

```yaml
python: .venv/bin/python
```

Pathlayer does not need to be installed there: its small pytest plugin is
copied next to each session. The compiler checks that this interpreter -- or,
without `python`, the one running Pathlayer -- can run pytest, and fails before
the run otherwise. A test module whose imports fail at run time stays
unobserved. The probe itself runs synchronously after every tool call: locally
about 60 ms for file checks and about half a second with a small pytest
selection. The printed configuration uses the interpreter Pathlayer runs in.
The OpenClaw recorder runs the probe automatically. With LangGraph, pass the
producer to the adapter:

```python
from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.config import build_pipeline, load_config
from pathlayer.detectors.proof_producer import producer_from_config

config = load_config()  # includes the printed detectors.progress_proof section
pipeline = build_pipeline(config)
adapter = LangGraphAdapter(pipeline.sink, progress_proof=producer_from_config(config))
```

The same vocabulary is available in Python from `pathlayer.coding_contracts`
(`CodingContract`, `FileExists`, `FileChanges`, `ArtifactProduced`,
`TestPasses`, `TestsImprove`, `CommandSucceeds`, `write_coding_contract`).

Keep the bundle outside the observed agent's write boundary. Test checkpoints
execute the workspace's own tests and `conftest.py` files with the workspace on
the import path, so an agent that can edit that test code -- or add a module
shadowing one pytest imports, or alter a `.venv` inside the workspace -- can
forge a passing test: protect it when test checkpoints must be trusted. [`examples/demo_coding_contract.py`](examples/demo_coding_contract.py)
runs a complete example.

### Crash-safe run state

Pass a stable logical run identity to resume detectors, policy hysteresis and
budget accounting after a process restart:

```python
from pathlayer import build_pipeline, load_config
from pathlayer.events import ToolEvent

config = load_config()
pipeline = build_pipeline(config, state_key="task-2026-08-10-001")

event = ToolEvent("read_file", {"path": "src/main.py"})
pipeline.process(event, event_id="framework-event-42")
pipeline.close("completed")
```

The default database is `.pathlayer/state.db`; change it with `store.path`.
Relative paths resolve from the current working directory. Without
`state_key`, `build_pipeline()` remains purely in-memory and does not create a
database. Durable operation requires SQLite WAL mode; construction fails
explicitly if the platform or filesystem cannot activate WAL, allowing an
integration to choose its documented memory-only fallback. `event_id` is
optional, but a stable framework identifier makes
retransmission idempotent: a duplicate returns the action already committed
for that event (including `WARN` or `STOP`) without advancing history or
emitting a new decision, provided the encoded event also matches. Reusing a
stable `event_id` for a different event is an identity conflict: the core raises
`StateConcurrencyError`, returns no action and does not silently degrade the
scope. OpenClaw contains that conflict by marking the run non-durable and
interrupted. Repeated calls without a stable identifier are intentionally
treated as distinct agent actions; Pathlayer never deduplicates from event
content alone.

Persistence continues only the same logical run. A completed, interrupted or
stopped run is terminal, and a new run key starts from empty state. Reopening a
key with decision-affecting configuration changes fails rather than silently
reinterpreting its history.

Runtime availability, write, serialization and decoding failures are
fail-open: the action already calculated is returned, `pipeline.durable`
becomes `False`, and that pipeline instance stays memory-only.
`pipeline.persistence_error` exposes the first failure. SQLite is then only a
durable prefix and the same `state_key` must not be resumed automatically,
because its in-memory suffix is absent. Identity conflicts are deliberately
excluded from this guarantee. Construction and replay also remain strict:
incompatible configuration, corruption, and newer database schemas raise
explicitly. Integrations can inspect an immutable,
transactionally consistent history through `SQLiteStateStore.read_scope()`;
the returned snapshot includes complete ticks, signals, event IDs, decisions,
and cost counters. `snapshot.durable` reports whether SQLite knows of a missing
tail; `snapshot.complete` is true only when that durable journal is also
terminal. An active scope is never complete. `snapshot.costs` covers only ticks
that Pathlayer analyzed and committed, so a STOP makes it a prefix rather than
the total spend of a session that continued running. Reporting integrations
must fold the complete raw event stream separately when they need actual
session spend, including events observed after STOP.

Pathlayer makes one best-effort SQLite write to poison a scope at the first
runtime persistence failure. If the store is unavailable enough that it cannot
record its own outage, the database can only show its last active prefix; the
live pipeline's `durable` flag remains authoritative. OpenClaw additionally
persists the loss in `manifest.json`, so it never automatically resumes that
prefix. Generic integrations must likewise retain the failure outside the
unavailable store or choose a new `state_key`.

Each stored tick carries the event timestamp separately from its encoded event,
so reporting can read chronology without decoding arguments, results or
metadata. A `NONE` decision has an empty action reason; detector observations
remain available in its signals. Replay compares the full stored decision, not
only the action. Maintainers must bump the dedicated state-semantics version
when unchanged configuration can change future decisions in a way an existing
prefix cannot reveal; ordinary package or observability changes do not
invalidate run state.

The v2-to-v3 migration backfills timestamps and priced-event counters one scope
at a time inside the serialized migration transaction. If one scope contains
an event payload that cannot be decoded losslessly, that scope is quarantined
as non-durable and cannot be resumed; other healthy scopes still migrate. A
detailed read of the quarantined scope continues to report corruption rather
than inventing replacement values.

The database stores normalized event arguments/results, signal reasons and
diagnostics. These can contain prompts, tool output or other sensitive data:
keep the local database private and exclude it from backups or exports that do
not have the same access controls. State is retained without automatic pruning
in this release so replay remains exact.

## LangGraph integration

Install `pathlayer[langgraph]`, then connect capture and actuation around your
compiled graph:

```python
from langgraph.errors import GraphDrained

from pathlayer import build_pipeline, load_config
from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.policies import wire
from pathlayer.policies.langgraph_actuator import LangGraphActuator

config = load_config()
pipeline = build_pipeline(config)
actuator = LangGraphActuator()
actuator.attach()
sink = wire(pipeline, actuator)
adapter = LangGraphAdapter.from_config(sink, config)
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

`from_config()` applies `config.pricing.effective_prices()` to LLM calls.
`OpenClawAdapter.from_config(sink, config)` provides the same pricing wiring.
For an OpenClaw `llm_output` carrying only total tokens, Pathlayer estimates
the input/output split with `pricing.openclaw_input_ratio` (default `0.8`).
Budget signals and reports identify that cost as estimated; it is not a
provider invoice. Estimates count toward the configured budget brake, like
per-tool budget estimates, and can select STOP. The OpenClaw recorder remains
passive: this decision does not terminate the external agent. Missing token
counters or unknown prices remain unmeasured; an explicitly configured
per-call budget estimate may still apply where an event exists.
State semantics version 2 rejects older durable scopes: estimated token costs
now belong to the estimated counter, and a known zero estimate no longer
receives a fallback charge. Start a new state key for runs created with the
previous semantics; their recorded files remain available for reporting.
The recorder also pins the effective pricing in the run manifest. Resuming an
LLM-bearing run with changed or missing pricing provenance is refused explicitly;
use a new run id instead of repricing previously recorded activity.

State capture is opt-in. To bracket each real tool call with the deterministic
filesystem collector, configure the adapter with an explicit sensor:

```python
from pathlayer.state_evidence import FilesystemStateSensor, StateEvidenceProducer

state_evidence = StateEvidenceProducer(
    [FilesystemStateSensor("workspace/result.json", resource_id="result.json")]
)
adapter = LangGraphAdapter(sink, state_evidence=state_evidence)
```

`PytestStateSensor` similarly observes an explicit tuple of pytest nodeids. The
adapter embeds the validated before/after records and a `changed`, `unchanged`,
`unavailable` or `collection_failed` outcome in the corresponding `ToolEvent`.
That metadata survives the normal trace and durable-store codecs and can be read
without I/O using `state_evidence_from_event(event)`. With no producer supplied,
capture and pipeline behavior are unchanged. Capture alone does not alter policy
decisions. The opt-in `StateEvidenceProgressAnchor` lets the existing progress
mechanisms consume novelty on one explicitly selected filesystem artifact.
See [VAL-5 integration and measured results](benchmarks/val5/README.md) for the
bounded runtime GO, configuration and abstentions. Novelty is not proof of task
success or semantic progress.

### The coding state preset

Naming every resource by hand does not scale to a coding agent that touches a
different file on every call. The opt-in `coding` preset selects them from the
arguments of the call itself:

```yaml
detectors:
  state_evidence:
    enabled: true
    preset: coding
    workspace: .            # where the tools' paths are resolved
    git: true               # observe the whole working tree as one resource
    tests: []               # optional exact pytest nodeids
    build: {}               # optional {id: check, command: [python, -m, mypy, src]}
    tools:                  # optional, for tools Pathlayer does not know
      apply_patch: {effect: workspace}
      save_code: {effect: write, paths: [target]}
```

```python
adapter = LangGraphAdapter.from_config(sink, config)   # capture side
pipeline = build_pipeline(config)                      # consumer side
```

What each call is observed on is decided by explicit rules, never by guessing:

| Call | Observed |
| --- | --- |
| A known file-writing tool (`write_file`, `move_file`, `file_delete`, `copy_file`) | the named file(s), plus the configured scope |
| A text-editor call (`command` + `path`) | that `path`, plus the configured scope |
| A tool declared under `tools` | what the declaration names |
| A read, or a tool Pathlayer does not know | nothing: the call claims no state |
| A mutating call whose paths cannot be resolved | the configured scope alone, never an invented file |
| The same, with no scope configured (`git: false`, no `tests`, no `build`) | nothing |

`tests` and `build` are observed once before and once after every observed
call, inside the adapter's lock, so the agent waits for them; `timeout_seconds`
(default 60) bounds each run. Leave them empty unless the selection is fast.

Paths outside the workspace, unusable paths, calls naming more files than can
be attributed, and two tool calls whose boundaries overlap are abstentions:
they produce fewer observations, never a fabricated one. A missing observation
stays a gap and never becomes `unchanged`.

The git resource fingerprints the working tree's *content*, so committing,
staging or amending is not a change, editing a file is, and `.gitignore` decides
what counts. Observation never writes to the observed repository.

A configured `build.command` runs with the workspace as its working directory,
and the observed agent can write there. Name the executable by absolute path,
or keep it outside the agent's reach: an agent that can plant a binary the
command resolves to can make Pathlayer run it. This is the same trust boundary
an ANC-4 probe has.

Two limits worth stating plainly. An observed effect is not success: the anchor
that feeds `no_progress` stops measuring as soon as test results or ANC-4
proofs are present, so an effect can never keep a stalled objective quiet.
And novelty is gameable -- an agent writing fresh junk on every call reaches new
states -- so pair the preset with a contract when the objective can be stated.

The fragment assumes `graph` and `initial_state` already exist. For WARN to
reach the model, add `actuator.injection_node()` before the model node.
SUMMARIZE additionally requires `summary_tool()`, `summary_gate_node(sink)` and
the conditional route returned by `summary_route()`. The complete runnable
wiring lives in [`examples/demo_no_progress.py`](examples/demo_no_progress.py).

At a duration timeout, `LangGraphActuator` requests cooperative drain. It does
not forcibly cancel a provider request that is already blocked; the STOP is
applied as soon as the runtime can observe/cooperate with that request. The
OpenClaw recorder persists the timeout, STOP decision, trace and report, but is
passive toward the separate OpenClaw process and does not terminate it.

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
- [`examples/demo_coding_contract.py`](examples/demo_coding_contract.py): a
  five-line coding contract on a real LangGraph agent whose tools return plain
  text -- productive repetition left alone, stagnation stopped.

The no-progress demo relies on deterministic ground truth from tests. Detecting
slow burns with no tests, exit codes, code evidence or other objective signal is
outside the zero-config V1 claim. An operator-supplied progress-proof contract
can cover such a run conditionally; the sealed ANC-3 corpus measurement and its
remaining short-trace limitation are documented in
[`docs/anc-3-corpus-measurement.md`](docs/anc-3-corpus-measurement.md).

## License

Apache-2.0.
