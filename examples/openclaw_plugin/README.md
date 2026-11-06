# OpenClaw integration and per-run recording

Pathlayer observes OpenClaw through a **loopback HTTP bridge** (not a sidecar
daemon, not in-process Python inside the Node gateway). A small JavaScript
plugin forwards hook payloads; Python translates them and runs detectors. With
`record.py --enforce`, a STOP also blocks the stopped run's later tool calls
(see [Enforce STOP](#enforce-stop-opt-in)).

## Architecture

```
OpenClaw Gateway (Node)
  └── pathlayer-plugin
        ├── before_agent_run, llm_output, after_tool_call, agent_end
        │     └── POST http://127.0.0.1:8787/ingest  (fire-and-forget)
        └── before_tool_call
              └── POST .../ingest, awaited (fails open)  ->  allow, or block after a STOP
                    └── record.py  OR  listen.py (demo)
                          └── OpenClawRunRecorder  OR  print ToolEvents
```

- **`listen.py`** — print-only demo listener (unchanged behaviour).
- **`record.py`** — production per-run recorder (durable directories).
- **`pathlayer/adapters/openclaw.py`** — translates `after_tool_call` → `ToolEvent`,
  and `llm_output` → a separate `LLM_CALL` event carrying an estimated cost.
- **`pathlayer/adapters/openclaw_recorder.py`** — one pipeline per `ctx.runId`.

Run identity: **`ctx.runId`** (fallback: `event.runId`). `event.toolCallId` is
stored in metadata only and is **never** used as `ToolEvent.run_id`.

## Required OpenClaw configuration

Conversation hooks (`before_agent_run`, `agent_end`) require explicit opt-in:

```json
{
  "plugins": {
    "entries": {
      "pathlayer-plugin": {
        "hooks": {
          "allowConversationAccess": true
        }
      }
    }
  }
}
```

## Installation

Install Pathlayer on the same host as the OpenClaw gateway: the plugin posts to
a loopback address.

```bash
git clone https://github.com/PathlayerLabs/pathlayer.git
cd pathlayer
python3 -m venv .venv
.venv/bin/pip install -e ".[yaml]"

# Recording storage, outside the git checkout (the default location is shown;
# any directory you own works)
export PATHLAYER_RUNS_DIR=/var/lib/pathlayer/runs
sudo mkdir -p "$PATHLAYER_RUNS_DIR"
sudo chown "$USER" "$PATHLAYER_RUNS_DIR"
```

Install the OpenClaw plugin from the checkout:

```bash
cp -r examples/openclaw_plugin/pathlayer-plugin ~/.openclaw/extensions/pathlayer-plugin
openclaw gateway restart
```

## Start the recording listener

From the checkout:

```bash
source .venv/bin/activate
export PATHLAYER_RUNS_DIR=/var/lib/pathlayer/runs
python examples/openclaw_plugin/record.py
```

Expected line: `Ingest URL : http://127.0.0.1:8787/ingest`. To keep the
listener running after you log out, run it under your service manager, or in
the background with
`nohup python examples/openclaw_plugin/record.py > record.log 2>&1 &`.

### Detector configuration

The listener uses the standard Pathlayer config -- no listener-specific
format. In priority order: `--config <path>`, `$PATHLAYER_CONFIG`, then
discovery of `.pathlayer/config.{yaml,yml,json}` in the working directory. With
no config file the built-in defaults apply.

```bash
python examples/openclaw_plugin/record.py --config path/to/config.yaml
```

An unreadable, malformed or out-of-range config **fails at startup** with a
`invalid Pathlayer config: ...` message and a non-zero exit -- it never falls
back to defaults silently.

## Enforce STOP (opt-in)

By default the listener only observes: a STOP is recorded and analysis ends,
but OpenClaw keeps running the agent's tool calls. Start the listener with
`--enforce` to act on it:

```bash
python examples/openclaw_plugin/record.py --enforce
```

Before each tool call, the plugin's `before_tool_call` hook asks the listener
for a decision. Once Pathlayer has stopped a run, every later tool call of that
run is blocked, and the agent receives the reason as the tool's error:

```text
Pathlayer stopped this run (loop: 5x repeat of 1-action block). Do not call
more tools: report your progress and what blocked you.
```

What this does and does not do:

- **Only STOP is enforced.** WARN and SUMMARIZE stay observed on OpenClaw:
  `before_tool_call` can allow, block or ask for approval, not advise.
- **One run at a time.** Other runs and later requests are not affected; each
  OpenClaw run has its own pipeline. A blocked run is not ended: the agent can
  still answer without tools.
- **Nothing extra is recorded.** The query carries identifiers only (run,
  session, agent, tool name), never tool arguments. It is never journaled and
  never reaches a detector, so recordings and their replay are the same with or
  without `--enforce`. Blocked calls that OpenClaw still reports through
  `after_tool_call` are recorded like any event after a STOP.
- **It fails open.** OpenClaw blocks a tool call when a `before_tool_call`
  handler throws or times out (15 seconds by default). The plugin never throws,
  gives up on Pathlayer after at most 0.8 s, and then lets the call run: a
  listener that is down, slow or broken never blocks the agent. Keep any
  `hooks.timeouts.before_tool_call` you set for this plugin well above 1 s.
- **Timing.** The plugin waits (at most 0.3 s) for its previous forward of the
  run to be answered before asking, so the tool call that follows the event
  that triggered the STOP is the first one blocked. If that forward times out,
  one more call can run.
- **Several attempts.** OpenClaw can run one `runId` in several attempts
  (after a compaction, a timeout or a provider failover), each ending with its
  own `agent_end`. A run stopped in one attempt stays blocked in the next ones,
  as long as no other run of its session starts in between (see *Runs with
  several attempts* below).
- **Restarts.** A listener restart marks the runs in progress as interrupted:
  they are no longer analyzed, so a run stopped before the restart is no
  longer blocked after it.

The ingest endpoint listens on loopback only and has no authentication: any
local process can post to it. Use `--enforce` on a host whose local users you
trust.

For a quick smoke test without persistence, use the print-only demo (it never
blocks anything):

```bash
python examples/openclaw_plugin/listen.py
```

## Run one test task

1. Open the OpenClaw dashboard and send a harmless task (e.g. “read README.md
   and report the line count”).
2. Confirm OpenClaw logs show `[pathlayer] forwarded before_agent_run` (etc.).
3. Confirm a run directory appears under `$PATHLAYER_RUNS_DIR/<runId>/`.

## Inspect a recorded run

Each run directory contains:

```
<run_id>/
├── manifest.json      # schema version, execution status, observed outcome, metadata
├── request.json       # before_agent_run snapshot (prompt, raw payload)
├── events.raw.jsonl   # every hook payload (verbatim)
├── trace.json         # replayable ToolEvent trace (format: traces/README.md)
├── decisions.jsonl    # Pathlayer's per-tool-event tick: signals, score, action
├── final.json         # agent_end snapshot
└── report.json        # end-of-session report
```

The recorder also keeps `state.db` at the runs root by default. It journals
normalized events, detector signals, actions and cumulative costs under the
OpenClaw `runId`, allowing an unfinished run to rebuild its exact pipeline
state after a listener restart. The raw JSONL journal and SQLite use the same
stable sequence-derived event IDs, so reattachment does not double-count old
events. A configured non-default `store.path` is honored instead.

SQLite cost counters describe only the analyzed and committed prefix, ending
at Pathlayer's STOP. `events.raw.jsonl` remains the source for the separate
recorder/reporting fold over every event, including the post-STOP suffix; the
post-session report must use that fold when reporting the real session spend
rather than presenting the SQLite snapshot as a total.

If SQLite becomes unavailable, OpenClaw keeps the raw JSONL recording and
continues analysis in memory. The manifest then records
`state_durable: false` and a `persistence_error` category. In that state the
SQLite journal is only a prefix and is never treated as a complete resumable
run; the raw journal remains the recovery source. A decision-affecting config
change similarly interrupts the old run instead of silently opening another
scope under the same `runId`. During reattachment, a store outage rebuilds a
fresh memory pipeline from the complete raw journal; a disagreement between a
stable event ID and its translated payload instead interrupts and poisons the
scope rather than mixing two histories.

Persistent state requires SQLite WAL mode. If WAL cannot be activated, the
recorder keeps operating from a fresh memory pipeline and records the loss of
durability in the manifest. During a v2-to-v3 migration, an undecodable event
quarantines only its own scope as non-durable; healthy scopes remain available.

Summary JSON files are written atomically (temp file, `fsync`, `os.replace`),
while the JSONL journals are appended and synced after each record. Run files
and SQLite contain raw or normalized tool data and may therefore contain
secrets. On POSIX, Pathlayer applies owner-only permissions (`0600`) to the
database, WAL and SHM files on a best-effort basis without changing an existing
parent directory; exported files are explicitly created as `0600`. Other
platforms and pre-existing directory permissions remain the operator's
responsibility. No automatic retention or compaction is performed in this
release.

**Execution status vs health.** `manifest.json` separates the two:

| field               | meaning                                                       |
| ------------------- | ------------------------------------------------------------- |
| `status`            | `started` / `orphan` / `awaiting_retry` / `completed` / `interrupted` |
| `stopped`           | whether Pathlayer emitted a STOP during the run               |
| `state_durable`     | whether SQLite contains every analyzed tick recorded so far   |
| `persistence_error` | storage error category after fail-open, otherwise `null`      |
| `health`            | curated verdict; live recordings remain `unknown`             |
| `health_reason`     | observed OpenClaw/Pathlayer outcome awaiting human curation   |
| `failed_attempts`   | attempts that ended with `success: false` (absent when none)  |

A run reaching `agent_end` is `completed`, which says nothing about whether it
was sound. Live capture records OpenClaw's reported outcome and Pathlayer's
decision in `health_reason`, but always keeps `health` and the trace label
at `unknown` until an explicit human curation step assigns ground truth. An
absence of intervention is not evidence that the run was healthy, and a STOP
is a detector outcome rather than a curated failure class.

**Runs with several attempts.** An OpenClaw run is a loop of attempts sharing
one `runId`: after a context-overflow compaction, a timeout compaction or a
provider failover, OpenClaw starts another attempt, and every attempt sends its
own `agent_end` and then its own `llm_output`. An `agent_end` whose `success` is
exactly `false` therefore ends an attempt, not the run:

- The run becomes `awaiting_retry`. This status is not terminal: no `final.json`,
  no `trace.json`. A provisional `report.json` (outcome `in_progress`) is written
  from the journals, and rebuilt when the failed attempt's `llm_output` arrives.
- The next attempt's `before_agent_run` resumes the run (back to `started`):
  same pipeline, same sequence, same cost, same STOP. A usage report does not
  resume it, since OpenClaw sends one after every attempt, and neither does a
  tool call: the plugin posts hooks without ordering them, so the failed
  attempt's last tool call can arrive after its end. Such a call is recorded
  and analyzed like any other. Usage received between two attempts counts
  in-run and reaches the budget.
- The run ends for good at an `agent_end` with any other `success` value (an
  absent value included, as before), or when the next run of its session starts:
  OpenClaw runs one run at a time per session, so the waiting run's failed
  attempt was its end (`ended_by: next_run_in_session` in the manifest). A run
  still waiting, or cut during a later attempt, at the next listener startup is
  marked `interrupted`; its `trace.json` is then written with every recorded
  event.
- A failed `agent_end` delivered again before any new attempt starts is
  dropped, like a duplicated final one. After a new attempt's start, an
  identical failed end is that attempt's own and counts.
- The duration watchdog counts from the first attempt. It pauses while the run
  waits: OpenClaw sends no signal when a failed attempt was the run's last, and
  a live timer would otherwise fire on a run that has already ended. If the
  deadline has passed, it fires at the next attempt's start.
- The report's duration of a multi-attempt run is the sum of the attempts'
  durations as OpenClaw measured them; the time between attempts (compaction,
  backoff) is in no measurement. A run left waiting has no duration until it
  ends.

This behavior is derived from OpenClaw's source (2026.7.1 and 2026.10.10) and
tested with synthetic payloads; a real compaction or provider failover has not
been recorded yet. Not covered: OpenClaw's model fallback runs each model's
attempt as its own task on the session lane, so another run of the session
could start between two attempts of one run. That start ends the waiting run,
and the next model's attempt is then neither recorded (only its usage report
reaches the ended run's `late.jsonl`) nor blocked (whether a run can actually
be queued in that gap is unconfirmed). And if an attempt's `before_agent_run`
is lost, that attempt runs with the duration watchdog paused, and a failed end
identical to the previous one is taken for a repeated delivery.

**A STOP does not end recording.** Pathlayer is a passive observer in V1: after
a STOP the detectors and arbitration stop running, but events keep being
normalized, kept in order and appended to `trace.json`. Those events appear in
`decisions.jsonl` with `"analyzed": false`.

**Every analyzed decision carries the tick that produced it.** Each
`"analyzed": true` line of `decisions.jsonl` records what Pathlayer computed for
that event, so the run's detector history is durable rather than derived from a
live process:

| column              | meaning                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `signals`           | every detector that ran, at its raw confidence, with its reason    |
| `fused`             | the arbitration score actually compared to the thresholds          |
| `arbitrated_action` | the action arbitration decided (`action` stays the *emitted* one)   |
| `dry_run`           | true when that decision was observed only and never applied        |
| `reason`            | why the action was taken; empty when nothing was decided           |

Silent detectors are kept in `signals` on purpose: "wired and reported 0.00" and
"not wired at all" are different facts, and a report has to tell them apart. A
line with `"analyzed": false` carries none of these columns — no detector ran for
that event, so asserting a confidence for it would be an invention.

`fused` and `arbitrated_action` arrived with the post-session report, which
makes them the marker of the journal's format. A run recorded by an earlier listener is still read back:
its `action` column gives the action Pathlayer emitted, so a STOP and its reason
survive, while the fused score it never recorded is reported as unknown rather
than as `0.00`. The report names how many lines are in that older format.

Terminal runs (`completed`, `interrupted`) are immutable: a duplicated
`agent_end` or a late tool call for such a run is logged and dropped, never
replayed onto the run directory. The one exception is a late `llm_output`.
OpenClaw 2026.7.1 reports model usage aggregated per attempt after `agent_end`;
post-run usage is therefore expected, not just a delivery race. That payload
is appended to the run's `late.jsonl` annex and `report.json` is rebuilt; it
never reaches a detector, and every other file stays byte-for-byte unchanged.
The headline cost includes the priced usage, separating estimates recorded
during the run (`estimated_usd`) from estimates received after the end
(`post_run_estimated_usd`). Unknown usage stays unmeasured; a priced zero is
shown as zero. A usage report can cover several model calls.

The **per-run budget** only sees events received while the run is active.
Model usage received after the end cannot trigger a budget STOP for that run.
Applying this spend to subsequent runs is planned in the
[session budget (#29)](https://github.com/PathlayerLabs/pathlayer/issues/29).

```bash
RUN=run-abc123   # original ctx.runId from manifest.json
cat "/var/lib/pathlayer/runs/$RUN/manifest.json"
cat "/var/lib/pathlayer/runs/$RUN/trace.json"
cat "/var/lib/pathlayer/runs/$RUN/report.json"
```

## Read the post-session report

A finalized run normally carries a `report.json`: what it cost, which detectors
reacted and how strongly, what Pathlayer did, and how the run ended. It is
written at `agent_end` (provisionally at the end of a failed attempt, then
rewritten when the run ends), and for a run that was cut short during the next
listener startup — nothing else will ever speak for it. That startup sweep only
reads the directory; it invents no cost figure, because the process that knew the
run's budget configuration is the one that disappeared. A run whose journal was
killed mid-line is still reported, with a note saying the truncated line was
dropped; a journal corrupted elsewhere is refused instead of summarised, and that
run keeps no `report.json` (the failure is logged and the sweep continues).

Repeated alerts are grouped into findings. A finding is one unbroken
run of non-zero readings from the same detector, shown once with its
occurrence count, event span, and peak confidence, and labelled with the tool
calls it covered, for example `loop / shell_exec: 15 occurrences (events 3-17,
peak 1.00)`. A model turn between two tool calls does not split it. A silent or
missing reading ends it, so the detector firing again later is reported as a
second finding. A finding does not mean Pathlayer acted: a reading below every
action threshold still forms one, and `actions` lists what was actually
arbitrated. This grouping affects only how the report presents the run:
`decisions.jsonl` and `trace.json` still hold every event and every reading.

`pathlayer-report` prints it back, rebuilding it from the run directory
(`manifest.json`, `decisions.jsonl`, `trace.json`, `final.json`) rather than from
any live state, so an `interrupted` run is reportable too:

```bash
pathlayer-report "$RUN"     # one named run, in full
pathlayer-report --last     # the most recently started run, in full
pathlayer-report --today    # one line per run started today (UTC)
pathlayer-report --all      # one line per recorded run
pathlayer-report --last --json   # the structured report instead of the summary
```

### Captured, attributed and unattributed cost

`--cost-attribution` replaces the per-run output with one cost split over the
selected runs. It needs a run selection like the other modes:

```bash
pathlayer-report --all --cost-attribution          # every recorded run
pathlayer-report --today --cost-attribution --json # structured, today's runs
```

```
cost attribution over 12 run(s), from llm_output hooks
captured: $0.4210 over 61 event(s), 2 unmeasured
attributed to a run: $0.3468 over 50 event(s), 2 unmeasured
  of which received after the run ended: $0.0081 over 1 event(s)
unattributed: $0.0742 over 11 event(s)
  missing_run_id_in_source (trigger cron): $0.0611 over 9 event(s)
  missing_run_id_in_source (trigger absent): $0.0131 over 2 event(s)
```

- **Captured** is every `llm_output` Pathlayer journaled, priced at the table
  pinned in the run's manifest when it was recorded, never at today's config. It
  always equals attributed plus unattributed. `llm_output` is the only OpenClaw
  hook that carries a cost; per-tool budget estimates are configuration, not
  captured cost, and are left out.
- **Attributed** means the source named the run (`ctx.runId`, else
  `event.runId`). Two subsets are shown as evidence and are never moved
  elsewhere: calls *received after the run ended* (see above), and calls in a run
  whose `before_agent_run` was never recorded.
- **Unattributed** has a single reason, `missing_run_id_in_source`: the hook
  named no run. Pathlayer never attaches such a cost to a run by session or by
  timing, because that would invent a causality the source did not report. It is
  grouped by the source's own `ctx.trigger` (`cron`, `heartbeat`, ...), or
  `absent` when the source gave none.
- **Unmeasured** calls (unknown model, no usable token counter, no pinned price
  table) are counted and never priced.

What the split cannot see: a payload the plugin failed to forward within its
300 ms timeout, or one sent while no listener was running, never reaches
Pathlayer, so no figure here accounts for it.

A single-run report also prints its own attribution, and a run that never wrote a
`trace.json` (an `orphan-*` directory, or a run cut before `agent_end`) now
reports the LLM cost read back from its raw hook journal instead of reporting it
as unmeasured.

Runs are listed in chronological order of their recorded start, ties broken by
run id, so the output is deterministic. Add `--runs-root` to point at another
recordings root (default: `$PATHLAYER_RUNS_DIR`, else `/var/lib/pathlayer/runs`).
An unknown or ambiguous run id, and `--last` with nothing recorded, fail with a
message on stderr and a non-zero exit; `--today` or `--all` matching nothing is
not an error.

```
run run-abc123 (session sess-1, agent agent-1)
outcome: stopped -- Pathlayer halted the run
status: completed   2026-08-10T12:30:58+00:00 -> 2026-08-10T12:30:59+00:00, 4.2s
events: 8 recorded, 5 analyzed
cost: $0.0123 effective = $0.0123 estimated [... $0.012300 estimated from the
  reported input, output and cache token counters at local rates]
cost attribution:
  captured: $0.0123 over 4 event(s)
  attributed to a run: $0.0123 over 4 event(s)
  unattributed: no costed event
detectors: 3 wired, 1 reacted over 5 recorded tick(s)
  loop         peak 0.94 on event 4 (t=4.0), non-zero on 4 of 5 tick(s) -- 5x repeat
  no_progress  silent on all 5 tick(s)
  retry        silent on all 5 tick(s)
fused score: peaked at 0.94 on event 4 (t=4.0)
actions: 1 arbitrated
  event 4 (t=4.0): STOP -- loop: 5x repeat of 1-action block
stop reason: loop: 5x repeat of 1-action block
```

The five outcomes are `completed` (reached its end without being halted — the
actions section says what was emitted), `stopped`, `escalated` (reached the
ESCALATE tier without stopping), `interrupted` (never reached its end) and
`in_progress` (a provisional report of a run waiting for another attempt; it is
rewritten when the run ends). A run nobody saw finish is never filed as
`completed`.

**What the report will not claim.** A figure nothing computed is reported as
absent rather than as zero — and, symmetrically, a figure that *was* computed is
reported even when it comes out at zero:

- **Cost.** Tool hooks carry no cost. Model spend comes from `llm_output`: when
  it reports a model and its input, output, cache-read and cache-write tokens,
  each counter is priced at its own rate from the `pricing` table (cache tokens
  of a model without cache rates stay unpriced, counted in
  `unpriced_cache_tokens`). When it reports only a non-negative `usage.total`,
  the total is split into input and output with `pricing.openclaw_input_ratio`
  (default 0.8). Either way that is an *estimate* at local rates, never an
  invoice-grade measurement, and the report labels it as such. A run recorded
  before split pricing keeps the total split it was priced with. An unknown model or a missing
  counter leaves that call unmeasured rather than free. The other substitute is
  `detectors.budget.tool_costs_usd`: a known price per tool times the
  number of calls, also reported as an estimate, separately from captured cost.
  That estimate is persisted on every tool call, not only at `agent_end`, so a run
  killed mid-flight keeps what it had already spent. A per-tool estimate of `$0.00`
  is a measured zero and is reported as such; only a run that recorded no estimate
  at all is reported as unmeasured, and if that run never reached its end the
  report calls the figure indeterminate rather than blaming a disabled detector.
- **Cross-run spend.** `detectors.budget.baseline_usd` is the seed this run's
  detection actually started from. The cumulative total across runs belongs to
  the SQLite state store, which this report does not read: it is the engine's
  inter-run state, while a run directory is the adapter's own auditable capture.
  The report keeps one distinct, still-empty field as the seam between the two.
- **Detector activity.** A run recorded by an older listener, before this
  report existed, has no per-tick confidence and no fused score in its journal. The report says both are unknown, not that
  no detector reacted and not that the score was zero.

The most actionable line is often the fused score: a run that peaked at 0.85 and
never reached the stop threshold is reported as exactly that, which is the signal
you need to retune a threshold.

## Export one run

```bash
pathlayer-export-run "$RUN" --output "/tmp/$RUN.tar.gz"
cat "/tmp/$RUN.tar.gz.sha256.json"
```

Refuses unknown or ambiguous run IDs. The archive and its checksum manifest are
created owner-only (`0600`) whatever the umask, and moved into place only once
complete -- an exported run carries the same prompts and tool results as the run
directory it comes from.

## Replay `trace.json`

```bash
python - <<'PY'
from pathlayer.validation.replay import replay_trace
from pathlayer.validation.traces import load_trace
report = replay_trace(load_trace("/var/lib/pathlayer/runs/RUN_ID/trace.json"))
print(report)
PY
```

To replay a trace file or a folder of traces from the command line, run
`pathlayer-replay <path>`.

## Privacy warning

Recorded runs contain **prompts, tool arguments, and tool results** as sent by
OpenClaw. Treat `$PATHLAYER_RUNS_DIR` as sensitive data. Restrict filesystem
permissions (owner-only; Pathlayer's SQLite chmod is POSIX best-effort). Do not
commit run directories into git.

## Limitations

- Only STOP is enforced, and only with `--enforce` (see
  [Enforce STOP](#enforce-stop-opt-in)). Without it, a STOP is recorded and
  stops analysis without affecting the OpenClaw run.
- No retention: run directories accumulate until you archive or delete them.
- Only the four forwarded hooks are recorded; the `before_tool_call` query is
  answered, never recorded. Model cost is an estimate from the reported token
  counters (or `usage.total` and a configured input/output ratio) at local
  rates, not a provider invoice.
- Hook payload shapes follow the public OpenClaw docs and may differ slightly
  on your gateway version — inspect `events.raw.jsonl` when in doubt.
- `ts` on ToolEvents reflects HTTP arrival order, not wall-clock time.

## Rollback

```bash
rm -rf ~/.openclaw/extensions/pathlayer-plugin && openclaw gateway restart
pkill -f "examples/openclaw_plugin/record.py" || true
```
