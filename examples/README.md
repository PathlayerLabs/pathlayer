# Examples

Runnable before/after demos. The key demo: an agent that loops forever
**without** Pathlayer, vs. the same agent **with** Pathlayer detecting the
loop, stopping execution, and producing a summary.

## `demo_loop.py`

Single before/after demo in one command, covering the three real Sprint 1
detectors. For each scenario it builds a trace where a naive agent would keep
going, then runs it through Pathlayer which detects the problem, stops cleanly,
and reports why (post-mortem):

- loop (`LoopDetector`): the agent replays the same actions forever;
- retry (`RetryDetector`): the agent replays the same failing call;
- budget (`BudgetDetector`): the agent burns through its dollar budget.

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
