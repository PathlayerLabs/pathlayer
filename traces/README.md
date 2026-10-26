# Traces

Labeled recordings of agent runs, used to validate the detectors offline (no
live agent, no LLM). Each file is one run. The format and the loader/writer live
in [`pathlayer/validation/traces.py`](../pathlayer/validation/traces.py).

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
| `label` | yes | One of `loop`, `no_progress`, `healthy` (or `unknown` for an unclassified recorded run). |
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
| `kind` | `"tool_call"` | Includes `tool_call`, resource events, and the duration watchdog's `run_start`/`run_timeout`. |
| `run_id` | `null` | Run identifier (cross-run state). |
| `metadata` | `{}` | Free-form per-event data. |

Fields left at their default are omitted on write, so simple traces stay terse.

## Labels

- **`loop`** - the agent replays the same action(s); the stream compresses trivially.
- **`no_progress`** - each step looks different but nothing actually advances (the
  "busy but stuck" slow burn shown in [`docs/no-progress-demo.md`](../docs/no-progress-demo.md)).
- **`healthy`** - a normal run that must **not** trigger any intervention (guards
  against false positives).
- **`unknown`** - neutral, non-curated label: the run's class was never
  established. Live recordings (such as the OpenClaw recorder's) use it rather
  than claiming `healthy` just because the agent reached the end. Such traces
  never belong to the golden set; the replay harness may replay them for
  inspection but excludes them from confusion metrics. The three labels above
  are the curated classes.

## Loading / writing

```python
from pathlayer.validation import load_trace, dump_trace, Trace

t = load_trace("traces/loop_trivial.json")   # -> Trace(label="loop", events=[...])
dump_trace(t, "traces/copy.json")            # round-trips: load(dump(t)) == t
```

## Layout

- The three minimal **format examples** stay flat at the root of `traces/`
  (`loop_trivial.json`, `no_progress_trivial.json`, `healthy_trivial.json`).
- The curated **golden set** lives in one subfolder per class:
  `traces/loop/`, `traces/no_progress/`, `traces/healthy/`.

Tests discover **every** `*.json` under `traces/` recursively, so the folder a
trace sits in is cosmetic -- the `label` field is the source of truth. The
golden set is guarded by `tests/test_golden_traces.py` (>= 5 traces per class,
each well-formed and described).

## Current corpus

The tables below describe the curated golden set. Mutated variants with
machine-derived expectations are generated in memory by
[`pathlayer/validation/mutations.py`](../pathlayer/validation/mutations.py)
rather than stored here.

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
| `no_progress/slow_burn_47k.json` | An analyzer/verifier debate modeled on the publicly reported $47k incident: distinct text + cost each turn, zero real progress. |
| `no_progress/replan_churn.json` | Plan rewritten over and over, never executed. |
| `no_progress/failing_edit_churn.json` | Genuinely different patches each turn, but the test result never moves off `3 failed`. |
| `no_progress/research_rabbit_hole.json` | Endless distinct searches/reads, no change applied. |
| `no_progress/rich_flat_red.json` | Rich test results: stable per-test debts remain red while activity continues. |
| `no_progress/rich_oscillation.json` | Rich test results: one debt oscillates after its first repair; repeated recovery earns no new credit. |
| `no_progress/rich_deleted_failure.json` | Rich test results: a failing test disappears; absence is not credited as a repair. |

### `healthy` (false-positive boundary)

| File | Why it must NOT fire |
| --- | --- |
| `healthy_trivial.json` | Varied tools that each advance the task; terminates. |
| `healthy/retry_then_succeed.json` | Bounded retries that self-resolve, then progress. |
| `healthy/same_tool_different_args.json` | Same tool, genuinely different files (legit exploration). |
| `healthy/expensive_but_legitimate.json` | High cumulative cost but real progress; terminates. |
| `healthy/varied_long_run.json` | Long but progressing run; length alone is not a fault. |
| `healthy/multi_file_refactor.json` | N-gram loop adversarial: same read/edit/test shape repeats, but each occurrence targets a genuinely different file -- no n-gram of tokens ever recurs. |
| `healthy/repo_exploration.json` | N-gram loop adversarial: alternating search/read with a distinct query and file every time -- similar-looking but never a recurring n-gram. |
| `healthy/repeated_checkpoint_pair.json` | N-gram loop known-limitation fixture: real work (edit) every cycle, but a fixed routine checkpoint pair (lint, run_tests) recurs every time -- raw LoopDetector confidence is *not* zero (pure repetition-frequency cannot distinguish this from an idle interruption), but capped + suppressed by real test progress, so the default pipeline never STOPs on it. |
| `healthy/rich_fail_to_pass_progress.json` | Rich test results: stable failing test IDs convert to passed often enough to reset patience. |
| `healthy/rich_late_test_progress.json` | Rich test results: work begins before test evidence exists, then rich outcomes show real repairs. |
| `healthy/rich_new_target_progress.json` | Rich test results: a new red target progresses locally without competing with an older target's absolute score. |
| `healthy/rich_legacy_revert_recovers.json` | Rich test results: the producer drops per-test outcomes for a few runs (guarded reversion reverts the target to aggregate tracking instead of staying wedged), then resumes rich mode and ends fully green. |

### `anc4/` (coding-contract probe outage)

| File | Why it must NOT fire |
| --- | --- |
| `anc4/probe_outage_mid_run.json` | A productive agent under a compiled coding contract breaks its module's syntax for ten consecutive calls mid-refactor; the probe observes the file but no test, so each call carries a `progress_proof_unobserved` marker. With `ProofProgressAnchor` those calls are gaps and the run completes; with the markers stripped the frozen level STOPs at call 17 (`tests/test_anc4_outage_trace.py`). Kept outside `healthy/`, whose exact contents the progress-proof corpus measurement seals. |

## Optional rich test-result contract

Aggregate ``passed``/``failed`` counts remain supported unchanged. A test tool
can additionally emit stable per-test outcomes when it can guarantee identity
and collection scope:

```json
{
  "passed": 2,
  "failed": 1,
  "test_outcomes": {
    "tests/test_api.py::test_create": "passed",
    "tests/test_api.py::test_delete": "failed"
  },
  "scope_id": "pytest:tests/test_api.py",
  "collection_complete": true,
  "scope_exhaustive": true
}
```

- ``test_outcomes`` maps stable IDs to ``passed``, ``failed``, ``skipped``,
  ``error`` or ``flaky``. ``failed``, ``error`` and ``flaky`` create a debt;
  its first later ``passed`` earns one permanent progress credit. ``skipped``
  creates no debt and no credit, but—like every non-``passed`` status—prevents
  that observation from establishing satisfaction.
- ``scope_id`` identifies the executed selection but never identifies a test
  debt; debts are keyed by ``(target, test_id)``.
- ``collection_complete`` says whether omissions from this selection are
  meaningful. In a complete collection, an ID omitted from the same
  ``scope_id`` loses comparability; omission from another partial scope does
  not. Explicitly reported outcomes remain usable when it is false.
- ``scope_exhaustive`` says this selection covers the whole target and may
  establish success. A partial green run never suppresses no-progress.

The producer is responsible for these guarantees; adapters transport the raw
result without interpreting it. Malformed rich-only payloads are inert and do
not disable weaker exit-code/git anchors; valid aggregate counts in the same
event use the legacy fallback. Per-target rich mode is sticky, but
**guarded-reversible**: a lone aggregate-only event for a rich target is
inert (as above), but 3 consecutive aggregate-only events (``_LEGACY_REVERT_STREAK``)
revert the target to plain aggregate tracking, discarding its rich debt
ledger. A later rich event restarts rich mode with a fresh offset from the
latest aggregate value. This is a deliberate trade-off, not a full fix: it
bounds but does not eliminate a re-count risk -- a producer that deliberately
oscillates rich and aggregate data for the same target could still get a debt
it already fixed once credited again after a revert, because the ledger that
remembered the first fix is gone. Its immutable incremental state costs O(K)
time per rich update and O(K) memory for K unique test IDs, independent of
history length.
For safety, one payload and one target ledger are capped at 10,000 IDs;
the whole anchor is capped at 10,000 tracked IDs across 256 rich targets, with
256 remembered scopes per target, 10,000 stored ``(scope_id, test_id)`` status
entries in total, and 512 characters per ``scope_id``. Oversized payloads are
inert, while cumulative ID/target/scope overflow is sticky and can never earn
credit or establish satisfaction.

None of the above is silent. A malformed payload, a capacity cap reached, a
guarded-reversion streak progressing or committing are all reported as
human-readable notes in ``Signal.diagnostics`` (see
[`pathlayer/detectors/base.py`](../pathlayer/detectors/base.py)), surfaced by
the existing structured logging (``pathlayer/observability.py``) on
every tick -- including the common silent (confidence 0) one, so an operator
can see a degraded producer without the run first having to look stalled.
Diagnostics never feed arbitration; they are purely observational.

These fixtures are synthetic boundary cases, not production recordings. Runs
without any deterministic ground truth remain outside this contract; an
operator-supplied progress-proof contract can cover them conditionally (next
section).

## Progress-proof corpus measurement

This measurement replays the 20 healthy and 8 no-progress traces twice: unchanged under
zero-config, then on freshly loaded copies enriched by an external deterministic
producer following explicit per-trace checkpoint schedules. The source traces
remain byte-for-byte unchanged. See
[`docs/progress-proof-corpus-measurement.md`](../docs/progress-proof-corpus-measurement.md) for the
reproduction command, before/after metrics, per-trace outcomes and the one
documented short-trace false negative.
