# Examples

Runnable before/after demos. The key demo: an agent that loops forever
**without** Pathlayer, vs. the same agent **with** Pathlayer detecting the
loop, stopping execution, and producing a summary.

## `demo_loop.py` (DEMO-1)

Smallest complete Pathlayer story: an agent gets stuck in a loop and would run
forever, versus the same agent stopped through the **real ACT-0 runtime path**
(not a shortcut):

```text
agent -> wire(pipeline, actuator) -> LoopDetector -> Policy -> STOP
      -> DemoActuator.apply(STOP) -> the agent drains and stops
```

`DemoActuator` is the offline analogue of `LangGraphActuator`: on `STOP` it
requests a cooperative drain (a flag the agent observes). It never decides --
the `Action` arrives already arbitrated by the pipeline.

The demo prints a before/after for the looping agent (triggering detector,
detection step, STOP, work avoided) and runs a **healthy bounded counterpart**
that is deliberately *not* stopped (a false positive is worse than no
Pathlayer at all). OBS-1 signal/decision/action logs go to stderr as JSON
lines. Deterministic, offline, no LLM, no network.

```bash
python examples/demo_loop.py
```

## `demo_no_progress.py` (DEMO-2)

The flagship story: the **no-progress "slow burn"** (the "$47k incident"). An
agent looks busy and changes its actions every turn -- a genuinely different
edit, then the test suite, over and over -- but the ground-truth progress signal
(tests passing) never moves. Repetition detectors stay silent because nothing
repeats; only `NoProgressDetector` (CUSUM on the flat test result) catches it.

Unlike DEMO-1, this runs a **real compiled LangGraph `StateGraph`** (still no LLM
and no network: the model node is scripted and deterministic) through the full
runtime path, and shows the *graded* intervention rather than an immediate hard
stop:

```text
LangGraphAdapter (capture) -> wire(pipeline, actuator) -> build_pipeline()
    -> NoProgressDetector -> Policy (graded ladder) -> LangGraphActuator

warn      -> a non-interrupting advisory is injected into the agent context
summarize -> a strict gate forces a structured progress summary before any
             other tool runs (the agent complies, emitting one summary)
stop      -> progress still never resumes, so the run is drained and halted
```

`loop` and `retry` are wired **advisory** (observe-only): the demo prints their
(flat) confidence to prove they stay silent, but only `no_progress` drives the
ladder. This is the calibrated DEMO-2 separation -- with `loop`/`retry`
actionable, a healthy trace (`retry_then_succeed`) would be forced to SUMMARIZE
(a false positive), which `tests/test_demo2.py` proves in both directions.

A healthy counterpart (same activity shape, but the failing-test count strictly
decreases to zero) runs through the identical graph and is never warned,
summarized or stopped.

Scope: this demonstrates the differentiator on an agent that emits a
**deterministic ground-truth signal** (test results). It does not claim to catch
slow burns with *no* tests, git or exit code -- that native-signal-free case is
tracked separately by ANC-1 (#135).

The demo runs tools through the idiomatic `langgraph.prebuilt.ToolNode`. A
`ToolNode` wraps each result in a `ToolMessage`; the ADP-0 adapter unwraps that
wrapper (structured `artifact`, else a JSON object/array `content` parsed back to
a dict/list) so `TestProgressAnchor` still sees the structured `{passed, failed}`
result and no_progress fires -- exactly as it would on a real agent.

Needs the langgraph extra (`pip install "pathlayer[langgraph]"`):

```bash
python examples/demo_no_progress.py
```

## `demo_coding_contract.py` (ANC-4)

The same no-progress story when the agent's tools return plain text, as a shell
tool would: the default coding anchors have nothing to measure. The operator
writes a five-line coding contract instead:

```yaml
task: implement-calculator
objective: Implement the four calculator operations so their tests pass.
progress:
  - type: tests_improve
    path: tests/test_calculator.py
  - type: file_changes
    path: calculator.py
```

The demo compiles it into a progress-proof bundle, enables it through the
normal configuration path, and passes the producer to `LangGraphAdapter`. The
generated probe runs the real tests and checks the real file after every tool
call of a real compiled `StateGraph`. STOP is enforced through
`LangGraphActuator`, with production thresholds and the default detectors.
Five scripted scenarios were fixed before the first run:

| Scenario | With the contract | Without it |
| --- | --- | --- |
| productive: a wrong attempt then a fix, per function | no action, 16 calls | no action, 16 calls |
| stagnant: a different wrong edit every time | STOP at call 11 | no action, 16 calls |
| oscillating: fixes `add`, breaks it, fixes it again | STOP at call 11 | no action, 16 calls |
| objective reached, then unrelated notes | no action, 16 calls | no action, 16 calls |
| stagnant, but the probe cannot import Pathlayer | no action, 0 proofs, 16 warnings | no action, 16 calls |

Every scenario also replays identically from its recorded trace after the
workspace is deleted, and decides identically through a hand-written low-level
manifest and probe fed the same checkpoints. The file edit in the stagnant run
is an observed effect: it counts once, then the unchanged tests are a stall.
The unavailable probe is fail-open: no proof, no decision change, one explicit
warning per tool call. `tests/test_anc4_coding_demo.py` asserts this table.

Needs the `langgraph` and `yaml` extras:

```bash
python examples/demo_coding_contract.py
```

## `pathlayer.config.yaml`

Reference configuration (INF-4) documenting every tunable threshold with its
built-in default. Pathlayer is zero-config: with no file present it uses these
exact values. Copy it to `.pathlayer/config.yaml` (`.yml` and `.json` are also
discovered) at your project root and override only what you need, then:

```python
from pathlayer import load_config, build_pipeline

pipeline = build_pipeline(load_config())  # discovers .pathlayer/config.*
```

YAML support needs the optional extra: `pip install "pathlayer[yaml]"` (JSON
works with no extra dependency).
