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
| `kind` | `"tool_call"` | Includes `tool_call`, resource events, and RES-2 `run_start`/`run_timeout`. |
| `run_id` | `null` | Run identifier (cross-run state). |
| `metadata` | `{}` | Free-form per-event data. |

Fields left at their default are omitted on write, so simple traces stay terse.

## Labels

- **`loop`** - the agent replays the same action(s); the stream compresses trivially.
- **`no_progress`** - each step looks different but nothing actually advances (the
  47k$-style slow burn).
- **`healthy`** - a normal run that must **not** trigger any intervention (guards
  against false positives).
- **`unknown`** - neutral, non-curated label: the run's class was never
  established. Live recordings (ADP-2) use it rather than claiming `healthy`
  just because the agent reached the end. Such traces never belong to the golden
  set; VAL-3 may replay them for inspection but excludes them from confusion
  metrics. The three labels above are the curated classes.

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
| `loop/retry_same_error.json` | Retry storm: the agent calls the same deploy endpoint with identical args and gets the same timeout error every time, never backing off or changing approach. Caught by the retry detector (same tool + args + error) and, structurally, by the loop detector (a period-1 repetition). |

### `no_progress`

| File | What makes it no-progress |
| --- | --- |
| `no_progress_trivial.json` | Re-analyze / re-plan, distinct each step, never edits. |
| `no_progress/slow_burn_47k.json` | The 47k$-style analyzer/verifier debate: distinct text + cost each turn, zero real progress. |
| `no_progress/replan_churn.json` | Plan rewritten over and over, never executed. |
| `no_progress/failing_edit_churn.json` | Genuinely different patches each turn, but the test result never moves off `3 failed`. |
| `no_progress/research_rabbit_hole.json` | Endless distinct searches/reads, no change applied. |
| `no_progress/rich_flat_red.json` | NOP-5: stable per-test debts remain red while activity continues. |
| `no_progress/rich_oscillation.json` | NOP-5: one debt oscillates after its first repair; repeated recovery earns no new credit. |
| `no_progress/rich_deleted_failure.json` | NOP-5: a failing test disappears; absence is not credited as a repair. |

### `healthy` (false-positive boundary)

| File | Why it must NOT fire |
| --- | --- |
| `healthy_trivial.json` | Varied tools that each advance the task; terminates. |
| `healthy/retry_then_succeed.json` | Bounded retries that self-resolve, then progress. |
| `healthy/same_tool_different_args.json` | Same tool, genuinely different files (legit exploration). |
| `healthy/expensive_but_legitimate.json` | High cumulative cost but real progress; terminates. |
| `healthy/varied_long_run.json` | Long but progressing run; length alone is not a fault. |
| `healthy/multi_file_refactor.json` | LOOP-2 adversarial: same read/edit/test shape repeats, but each occurrence targets a genuinely different file -- no n-gram of tokens ever recurs. |
| `healthy/repo_exploration.json` | LOOP-2 adversarial: alternating search/read with a distinct query and file every time -- similar-looking but never a recurring n-gram. |
| `healthy/repeated_checkpoint_pair.json` | LOOP-2 known-limitation fixture: real work (edit) every cycle, but a fixed routine checkpoint pair (lint, run_tests) recurs every time -- raw LoopDetector confidence is *not* zero (pure repetition-frequency cannot distinguish this from an idle interruption), but capped + suppressed by real test progress, so the default pipeline never STOPs on it. |
| `healthy/rich_fail_to_pass_progress.json` | NOP-5: stable failing test IDs convert to passed often enough to reset patience. |
| `healthy/rich_late_test_progress.json` | NOP-5: work begins before test evidence exists, then rich outcomes show real repairs. |
| `healthy/rich_new_target_progress.json` | NOP-5: a new red target progresses locally without competing with an older target's absolute score. |
| `healthy/rich_legacy_revert_recovers.json` | NOP-5 follow-up: the producer drops per-test outcomes for a few runs (guarded reversion reverts the target to aggregate tracking instead of staying wedged), then resumes rich mode and ends fully green. |
| `healthy/finished_without_tests.json` | A research agent completes a summary task, then keeps issuing low-value follow-up searches. It carries no test or exit-code evidence, so the coding anchors abstain and NoProgressDetector stays silent -- we never stop an agent we cannot measure. Documents the F1.1 limit: 'finished' outside the test signal is handled by anchor abstention, not by is_satisfied. |
| `healthy/finishes_before_runtime_limit.json` | A long but healthy run starts lifecycle measurement and completes before the hard ceiling, so no timeout event exists and duration stays silent. |
| `healthy/noisy_progress.json` | Genuine but noisy progress: the agent fixes a suite while occasionally regressing by one test before climbing again (passed-minus-failed goes 1, 3, 2, 4, 3, 5). The trend is clearly upward and every dip is bounded at -1, so a record-based CUSUM keeps resetting and must stay silent. Guards NOP-2 against treating normal red/green churn as a stall. |
| `healthy/plateau_after_success.json` | The agent fixes a suite to all-green, then keeps polishing without breaking anything: progress plateaus at success (failed==0). A plateau here is 'nothing left to fix', not a stall, so NoProgressDetector must stay silent via anchor success suppression (is_satisfied). Guards NOP-3 against stopping an agent that has actually succeeded. |
| `healthy/repetitive_but_passing.json` | F1a precision false positive: the agent re-runs the same test suite with identical args (no `target`, so canonicalization sees an unchanging action) after every fix, and the test count climbs 1 -> 8 passed each time. LoopDetector sees only the repeated (tool, args) pair -- it has no visibility into `result` by design -- and reaches STOP by the 5th repeat even though the agent is making real, verifiable progress. Guards the loop<-progress suppression (LOOP-PRECISION): a repetition that is actually advancing must not be stopped by loop alone. |
| `healthy/retry_reset_after_progress.json` | RET-3 boundary: the agent hits a couple of transient rate-limited searches, then does genuine work (an edit, a passing test run, a commit), then resumes with a short fresh search burst. RET-2 must not merge the two search phases across the real work into one dominating run -- because only the trailing uninterrupted run of a tool counts (the counter resets on a genuinely different action), the final 3-call search burst is too short to dominate, so nothing fires. A naive bag-of-occurrences count over the window would risk a false alarm here. |
| `healthy/slow_but_real_progress.json` | Genuine but slow progress: the agent sets a new ground-truth record (passed-minus-failed climbs 0 -> 2 -> 4 -> 6) only every few edits, with flat stretches in between. Each stretch is shorter than the CUSUM patience, so confidence rises but never reaches the stop threshold and every new record resets it. Guards NOP-3 against stopping an honest-but-slow agent (false positive on legitimate work). |
| `healthy/varied_reads_by_id.json` | CAN-3 boundary: the agent reads many genuinely different files addressed by a numeric file_id (not a path). The _id suffix must NOT be treated as volatile, otherwise these distinct reads collapse into a false loop. After exploring, an edit and a passing test close the task. |

### `anc4/` (coding-contract probe outage)

| File | Why it must NOT fire |
| --- | --- |
| `anc4/probe_outage_mid_run.json` | ANC-4: a productive agent under a compiled coding contract breaks its module's syntax for ten consecutive calls mid-refactor; the probe observes the file but no test, so each call carries a `progress_proof_unobserved` marker. With `ProofProgressAnchor` those calls are gaps and the run completes; with the markers stripped the frozen level STOPs at call 17 (`tests/test_anc4_outage_trace.py`). Kept outside `healthy/`, whose exact contents the ANC-3 benchmark seals. |

### `nop4_experiment/` (net vs size git anchors)

| File | Why the label is what it is |
| --- | --- |
| `nop4_experiment/inplace_bugfix_no_tests.json` | Legit in-place bug fixing, no tests run. Each edit modifies new distinct existing lines, so the cumulative diff vs baseline keeps growing in SIZE (insertions+deletions) but its NET (insertions-deletions) stays ~0. Genuinely progressing -> label healthy. A net-based git anchor sees a flat 0 and false-stops; a size-based git anchor credits the growing churn and stays silent. Discriminates net vs size scalar. |
| `nop4_experiment/late_first_test_progressing.json` | The agent edits real code (git net staircases up) for several steps with NO test result yet, then the test suite appears late and stays red/flat while edits keep progressing. Genuinely progressing -> label healthy. A naive git anchor (always applicable) keeps crediting the rising net and prevents the false stop; a test-deferring conditional git anchor abstains once tests appear and cannot prevent the baseline false stop. Distinguishes naive vs conditional integration. |
| `nop4_experiment/same_zone_rewrite.json` | Churn: the agent rewrites the SAME zone over and over, so the cumulative diff vs baseline is constant in both NET and SIZE (insertions=4, deletions=4 every step). No tests run. Real no-progress -> label no_progress. Baseline has no applicable anchor (no test/exit/git-progress) and stays silent -> false negative; either git scalar sees a flat level and catches it. Shows git adds detection for no-test coding agents. |
| `nop4_experiment/tdd_red_progressing.json` | TDD red phase: the agent writes real, growing code (git net staircases up) while the test suite stays red and flat (passed=0, failed=3). Genuinely progressing -> label healthy. Baseline (test anchor only) sees a flat red level and false-stops; a git anchor that credits the rising net should prevent that false stop. |
| `nop4_experiment/useless_growth.json` | 47k-style slow burn: the agent keeps adding new but useless lines (git net staircases up) while the test level stays flat and not-satisfied (passed=2, failed=2). Real no-progress -> label no_progress. Baseline (test anchor) catches the flat level; a git anchor that credits the rising net MASKS it -> false negative. This is the hidden cost of the git anchor. |

### `val7/` (session-level traces)

Session-level traces, replayed with `session_stitching` enabled (see
[`docs/session-stitching.md`](../docs/session-stitching.md)). Each run owns its
pipeline; successful repetitions are never stitched.

| File | What the concluding turn freezes |
| --- | --- |
| `val7/polling.json` | concluding turn with frozen cross-run evidence: polling. The user asks again each turn and the same read-only check succeeds. |
| `val7/progress.json` | concluding turn with frozen cross-run evidence: progress. |
| `val7/repeat.json` | concluding turn with frozen cross-run evidence: repeat. |
| `val7/unavailable.json` | concluding turn with frozen cross-run evidence: unavailable. |

## Optional rich test-result contract (NOP-5)

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
the existing OBS-1 structured logging (``pathlayer/observability.py``) on
every tick -- including the common silent (confidence 0) one, so an operator
can see a degraded producer without the run first having to look stalled.
Diagnostics never feed arbitration; they are purely observational.

These fixtures are synthetic boundary cases, not production recordings. Runs
without any deterministic ground truth remain outside NOP-5 and are tracked by
ANC-1.

## ANC-3 progress-proof measurement

ANC-3 replays the 20 healthy and 8 no-progress traces twice: unchanged under
zero-config, then on freshly loaded copies enriched by an external deterministic
producer following explicit per-trace checkpoint schedules. The source traces
remain byte-for-byte unchanged. See
[`docs/anc-3-corpus-measurement.md`](../docs/anc-3-corpus-measurement.md) for the
reproduction command, before/after metrics, per-trace outcomes and the one
documented short-trace false negative.
