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
