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

## Layout

- The three minimal **format examples** stay flat at the root of `traces/`
  (`loop_trivial.json`, `no_progress_trivial.json`, `healthy_trivial.json`).
- The curated **golden set** (VAL-2) lives in one subfolder per class:
  `traces/loop/`, `traces/no_progress/`, `traces/healthy/`.

Tests discover **every** `*.json` under `traces/` recursively, so the folder a
trace sits in is cosmetic -- the `label` field is the source of truth. The
golden set is guarded by `tests/test_golden_traces.py` (>= 5 traces per class,
each well-formed and described).

## Current corpus

Curated golden set: VAL-2. Programmatically generated/mutated traces are
VAL-4/VAL-5 (later).

### `loop`

| File | What makes it a loop |
| --- | --- |
| `loop_trivial.json` | Two-action cycle (period 2), volatile ids per turn. |
| `loop/single_action_repeat.json` | Period-1 exact repetition of one call. |
| `loop/period3_cycle.json` | Longer cycle of period 3 replayed identically. |
| `loop/volatile_ids.json` | Same call hidden behind fresh UUIDs/timestamps/cursors; visible only after canonicalization. |
| `loop/ping_pong.json` | Analyzer <-> verifier A->B->A->B oscillation. |

### `no_progress`

| File | What makes it no-progress |
| --- | --- |
| `no_progress_trivial.json` | Re-analyze / re-plan, distinct each step, never edits. |
| `no_progress/slow_burn_47k.json` | The 47k$-style analyzer/verifier debate: distinct text + cost each turn, zero real progress. |
| `no_progress/replan_churn.json` | Plan rewritten over and over, never executed. |
| `no_progress/failing_edit_churn.json` | Genuinely different patches each turn, but the test result never moves off `3 failed`. |
| `no_progress/research_rabbit_hole.json` | Endless distinct searches/reads, no change applied. |

### `healthy` (false-positive boundary)

| File | Why it must NOT fire |
| --- | --- |
| `healthy_trivial.json` | Varied tools that each advance the task; terminates. |
| `healthy/retry_then_succeed.json` | Bounded retries that self-resolve, then progress. |
| `healthy/same_tool_different_args.json` | Same tool, genuinely different files (legit exploration). |
| `healthy/expensive_but_legitimate.json` | High cumulative cost but real progress; terminates. |
| `healthy/varied_long_run.json` | Long but progressing run; length alone is not a fault. |
