# ANC-3 — progress-proof corpus measurement

ANC-3 asks a narrow question: when a domain supplies explicit checkpoint
ground truth through the ANC-1 schema and ANC-2 producer, do the four previously
unmeasurable no-progress traces become detectable without stopping a healthy
trace?

## Reproduce

From a development checkout with its dependencies installed:

```bash
python scripts/anc3_corpus_benchmark.py
python scripts/anc3_corpus_benchmark.py --json
```

The benchmark targets exactly `healthy_trivial.json`,
`no_progress_trivial.json`, `traces/healthy/*.json` and
`traces/no_progress/*.json`: 20 healthy and 8 no-progress traces. Loop traces
are deliberately excluded because ANC-3 measures the progress-proof contract,
not repetition detection.

The first pass reloads the original traces and uses the normal replay defaults.
The second pass reloads them again, generates one validated manifest per trace,
and invokes the real `ProgressProofProducer` on every event. Its external probe
reads only the sealed checkpoint schedule and its own invocation counter. It
never reads the tool, arguments, result, description or other agent-authored
text. The schedules live in
`scripts/anc3_progress_scenarios.json`; all 28 traces have an explicit entry.

The three long healthy traces that would otherwise have no early applicable
anchor carry human-authored intermediate milestones. Other healthy traces
remain protected by their existing objective anchors or finish before the
stagnation threshold. The committed trace files are never edited.

## Aggregate result

| Metric | Before | With progress proof | Delta |
| --- | ---: | ---: | ---: |
| Healthy traces | 20 | 20 | 0 |
| No-progress traces | 8 | 8 | 0 |
| False positives | 0 | 0 | 0 |
| False negatives | 4 | 1 | -3 |
| True positives | 4 | 7 | +3 |
| True negatives | 20 | 20 | 0 |
| FN rate on no-progress | 50.0% | 12.5% | -37.5 points |
| Mean detection index among detected pathological traces | 11.00 | 10.57 | -0.43 |
| Events avoided | 10 | 13 | +3 |
| Cost avoided | $0.00 | $0.47 | +$0.47 |

Detection indexes are zero-based. The three newly detected runs all STOP at
index 10, on their 11th event. For each, the canonical post-mortem reason begins
with `no_progress`, its confidence is positive, and every other actionable
signal is zero at the STOP tick.

The mean latency is calculated only over detected pathological traces. Its
population therefore grows from four to seven; the value is not a paired mean
over the same runs.

## Per-trace result

| Trace | Before | With progress proof |
| --- | --- | --- |
| `healthy_trivial.json` | silent | silent |
| `healthy/expensive_but_legitimate.json` | silent | silent |
| `healthy/finished_without_tests.json` | silent | silent |
| `healthy/finishes_before_runtime_limit.json` | silent | silent |
| `healthy/multi_file_refactor.json` | silent | silent |
| `healthy/noisy_progress.json` | silent | silent |
| `healthy/plateau_after_success.json` | silent | silent |
| `healthy/repeated_checkpoint_pair.json` | silent | silent |
| `healthy/repetitive_but_passing.json` | silent | silent |
| `healthy/repo_exploration.json` | silent | silent |
| `healthy/retry_reset_after_progress.json` | silent | silent |
| `healthy/retry_then_succeed.json` | silent | silent |
| `healthy/rich_fail_to_pass_progress.json` | silent | silent |
| `healthy/rich_late_test_progress.json` | silent | silent |
| `healthy/rich_legacy_revert_recovers.json` | silent | silent |
| `healthy/rich_new_target_progress.json` | silent | silent |
| `healthy/same_tool_different_args.json` | silent | silent |
| `healthy/slow_but_real_progress.json` | silent | silent |
| `healthy/varied_long_run.json` | silent | silent |
| `healthy/varied_reads_by_id.json` | silent | silent |
| `no_progress_trivial.json` | silent (FN) | silent (FN) |
| `no_progress/failing_edit_churn.json` | STOP @ 11 | STOP @ 11 |
| `no_progress/replan_churn.json` | silent (FN) | STOP @ 10, `no_progress` |
| `no_progress/research_rabbit_hole.json` | silent (FN) | STOP @ 10, `no_progress` |
| `no_progress/rich_deleted_failure.json` | STOP @ 10 | STOP @ 10 |
| `no_progress/rich_flat_red.json` | STOP @ 11 | STOP @ 11 |
| `no_progress/rich_oscillation.json` | STOP @ 12 | STOP @ 12 |
| `no_progress/slow_burn_47k.json` | silent (FN) | STOP @ 10, `no_progress` |

## Honest limit: the remaining false negative

`no_progress_trivial.json` has 10 events. Its first applicable proof establishes
the anchor baseline without accruing stagnation, leaving only nine subsequent
flat observations. The unchanged default configuration requires ten stagnant
steps to reach STOP, so a hypothetical 11th unchanged event would be the first
one eligible to stop.

ANC-3 does not lower a threshold, append an event or relabel the trace to force
a perfect score. Those would change detector calibration or corpus evidence and
need their own healthy counterexamples. The measured V1 result is therefore an
honest reduction from four false negatives to one, not a claim of universal
no-progress detection.

This experiment also does not prove that Pathlayer can author the customer's
business truth. The client still owns each checkpoint's meaning and must run
the probe outside the observed agent's control. Contract libraries, imports,
automatic selection, semantic inference and LLM-generated checks remain out of
scope.
