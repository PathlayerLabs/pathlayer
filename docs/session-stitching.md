# Session-level repetitions

Set `pipeline.session_stitching: true` in the Pathlayer configuration to recognize equivalent actions across several related runs. The feature is disabled by default.

Existing thresholds and signals are unchanged. Only the retry detector uses this context; loops, budgets, watchdogs, policy state, no-progress, lifecycle and STOP remain specific to each run.

## Capture

OpenClaw uses only the `ctx.sessionKey` supplied by the source. LangGraph requires a session identifier and a logical run identifier, both supplied explicitly by the caller. A missing or ambiguous identity produces no cross-run conclusion. Pathlayer does not detect that the same identity was reused for another task: the source or the caller must supply a task-specific identity. An identical failure repeated under a reused identity would therefore be stitched.

Each invocation therefore keeps its own independent pipeline and lifecycle. The end of a run does not close the source session.

Only identical failures are stitched: same canonical action and same error. A successful call is never stitched, because the same successful call repeated from one turn to the next most often comes from a user retrying (waiting for CI, asking a question again), not from a stuck agent. A different error or a success therefore breaks continuity.

The error alone is not enough: an unchanged and valid state observation, or a complete proof of checkpoints that are still unsatisfied, is also required to stitch two actions. A changed state, a partial or satisfied proof, an invalid or unavailable proof, a different action, a missing link or a test result breaks continuity. Tool output alone is never proof.

## Durability and replay

The append-only journal lives in `<store.path>.sessions`. Each capture is isolated by source, session, logical run and event identifier. Identical captures are idempotent; a different payload with the same identity is rejected.

The bounded context used for the decision is attached to the captured event. Replay therefore reproduces the decision without rereading the current journal or probing the workspace again. Raw payloads and canonical decisions are not rewritten. Historical configuration fingerprints remain compatible; enabling this feature has its own semantics version.

The labeled traces in `traces/val7/` carry this frozen context, which only counts with the option enabled. `tests/test_session_stitching.py` replays them with `session_stitching`, and they are excluded from the aggregate score of `python -m pathlayer.validation.replay traces`, which uses the default configuration.

A missed observation inserts a window break into the journal: no repetition is stitched across it, and continuity resumes with the observations captured afterwards. Three cases produce a break: a capture error, an interrupted OpenClaw run (its unobserved suffix is unknown) and a configuration change (observations are no longer comparable, including when switching back to the previous configuration). The session is never disabled permanently, and there is no need to change the session identity.

If the journal is unavailable, the break is kept in a marker synced to disk, one file per missed observation, then written to the journal before that session's next capture. Local processing continues normally. If even the marker cannot be written, the error is propagated rather than claiming a continuity that would survive a restart. A series of capture failures produces a single warning, until the next successful capture.

## Changes active even without the option

Three capture changes shipped with this feature apply even when `session_stitching` is disabled:

- `LangGraphAdapter.from_config` wires the coding-contract progress-proof producer (`producer_from_config`) as soon as `detectors.progress_proof` is enabled: the probe then runs after every tool call. To do without it, construct the adapter directly, without `progress_proof`.
- If that producer raises an exception on the LangGraph side, the event carries `progress_proof_unobserved: {"reason": "producer_failed"}`. This marker deliberately falls outside the closed progress-proof schema: `ProofProgressAnchor` ignores it and behaves as if there were no marker, but its presence is enough to prevent any cross-run stitching.
- The OpenClaw manifest keeps the `session_key` set when the run was created; the start hook no longer overwrites it with its own `ctx.sessionKey`.

## Read-only report

```bash
python -m pathlayer.session --source openclaw --session-key my-session \
  runs/turn-one/trace.json runs/turn-two/trace.json runs/turn-three/trace.json
```

Add `--json` for structured output and `--config` to use custom canonicalization masks. The same grouping is available through `SessionJournal.report()` and `project_session(...)`.

The report describes a comparable repeated failure that was observed: canonical identity, bounded tool, number of occurrences, run identifiers and observation index. It points to each run's own events and decisions, without raising a new alert or inventing a detector attribution.

## Limits

This version stitches only consecutive, equivalent failures, according to the existing canonicalization. A successful repetition across several turns is never reported at the session level, even if it reflects a real loop; per-run detection is unchanged. It does not yet stitch cycles that alternate between several tools, and it does not carry the general no-progress state across runs. An unknown proof makes the cross-session signal abstain; per-run detection continues normally.

The context is limited to 64 observations and 128 KiB of encoded events. Exceeding a limit creates a window break. These are storage and replay bounds, not thresholds calibrated on a new metric.

The journal contains sensitive trace data and must receive the same access protections as existing traces. Retention and compaction are outside the scope of this feature.
