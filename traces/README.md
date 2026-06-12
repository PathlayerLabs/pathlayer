# Traces

Labeled recordings of agent runs, used to validate the detectors offline (no
live agent, no LLM). Each file is one run. The format and the loader/writer live
in [`pathlayer/validation/traces.py`](../pathlayer/validation/traces.py) (VAL-1).

## On-disk format

A trace is a single JSON object:

```json
{
  "label": "loop",
  "description": "what makes this run loop / no_progress / healthy",
  "events": [
    { "tool": "list_dir", "args": { "path": "src" } },
    { "tool": "read_file", "args": { "path": "src/main.py" } }
  ]
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `label` | yes | One of `loop`, `no_progress`, `healthy`. |
| `description` | no | Short human note on why the run has that label. |
| `metadata` | no | Free-form object (source, seed, ...). |
| `events` | yes | Ordered list of events. |

Each event is the serialized form of a
[`ToolEvent`](../pathlayer/events.py). Only `tool` is required; every other
field is optional and falls back to the `ToolEvent` default when absent:

| Event field | Default | Notes |
| --- | --- | --- |
| `tool` | (required) | Tool / action name. |
| `args` | `{}` | Raw arguments (canonicalization happens later, in the detectors). |
| `result` | `null` | Raw result. |
| `cost_usd` | `0.0` | Cost of the call. |
| `ts` | `0.0` | Event-sourced timestamp (the detection loop never reads the wall clock). |
| `kind` | `"tool_call"` | `tool_call`, `subagent_spawn` or `run_restart`. |
| `run_id` | `null` | Run identifier (cross-run state). |
| `metadata` | `{}` | Free-form per-event data. |

Fields left at their default are omitted on write, so simple traces stay terse.

## Labels

- **`loop`** - the agent replays the same action(s); the stream compresses trivially.
- **`no_progress`** - each step looks different but nothing actually advances (the
  47k$-style slow burn).
- **`healthy`** - a normal run that must **not** trigger any intervention (guards
  against false positives).

## Loading / writing

```python
from pathlayer.validation import load_trace, dump_trace, Trace

t = load_trace("traces/loop_trivial.json")   # -> Trace(label="loop", events=[...])
dump_trace(t, "traces/copy.json")            # round-trips: load(dump(t)) == t
```

## Current corpus

These are minimal **format examples** (one per class). The curated golden set
(~5 per class, incl. the flagship slow-burn) is VAL-2; programmatically
generated/mutated traces are VAL-4/VAL-5.

| File | Label |
| --- | --- |
| `loop_trivial.json` | `loop` |
| `no_progress_trivial.json` | `no_progress` |
| `healthy_trivial.json` | `healthy` |
