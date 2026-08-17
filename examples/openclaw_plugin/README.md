# OpenClaw integration (ADP-1) + per-run recording

Pathlayer observes OpenClaw through a **loopback HTTP bridge** (not a sidecar
daemon, not in-process Python inside the Node gateway). A small JavaScript
plugin forwards hook payloads; Python translates them and runs detectors.

## Architecture

```
OpenClaw Gateway (Node)
  └── pathlayer-plugin (before_agent_run, after_tool_call, agent_end)
        └── POST http://127.0.0.1:8787/ingest  (fire-and-forget)
              └── record.py  OR  listen.py (demo)
                    └── OpenClawRunRecorder  OR  print ToolEvents
```

- **`listen.py`** — print-only demo listener (unchanged behaviour).
- **`record.py`** — production per-run recorder (durable directories).
- **`pathlayer/adapters/openclaw.py`** — translates `after_tool_call` → `ToolEvent`.
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

## Installation (Lightsail / same host as OpenClaw)

```bash
# Pathlayer release layout (example)
sudo mkdir -p /opt/pathlayer/releases /opt/pathlayer/shared
cd /opt/pathlayer/releases
git clone git@github.com:Kropemann/Pathlayer.git "$(date +%Y%m%d%H%M%S)"
cd "$(ls -1t | head -1)"
python3 -m venv .venv
.venv/bin/pip install -e ".[yaml]"
sudo ln -sfn "$PWD" /opt/pathlayer/current

# Recording storage (outside the git checkout)
sudo mkdir -p /var/lib/pathlayer/runs
sudo chown "$USER":"$USER" /var/lib/pathlayer/runs
export PATHLAYER_RUNS_DIR=/var/lib/pathlayer/runs
```

Install the OpenClaw plugin:

```bash
cp -r /opt/pathlayer/current/examples/openclaw_plugin/pathlayer-plugin \
  ~/.openclaw/extensions/pathlayer-plugin
openclaw gateway restart
```

## Start the recording listener

```bash
cd /opt/pathlayer/current
source .venv/bin/activate
export PATHLAYER_RUNS_DIR=/var/lib/pathlayer/runs
nohup python examples/openclaw_plugin/record.py > /opt/pathlayer/shared/record.log 2>&1 &
tail -f /opt/pathlayer/shared/record.log
```

Expected line: `Ingest URL : http://127.0.0.1:8787/ingest`.

### Detector configuration

The listener uses the standard Pathlayer config (INF-4) -- no listener-specific
format. In priority order: `--config <path>`, `$PATHLAYER_CONFIG`, then
discovery of `.pathlayer/config.{yaml,yml,json}` in the working directory. With
no config file the built-in defaults apply.

```bash
python examples/openclaw_plugin/record.py --config /opt/pathlayer/shared/config.yaml
```

An unreadable, malformed or out-of-range config **fails at startup** with a
`invalid Pathlayer config: ...` message and a non-zero exit -- it never falls
back to defaults silently.

For a quick smoke test without persistence, use the print-only demo:

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
├── trace.json         # replayable ToolEvent trace (VAL-1)
├── decisions.jsonl    # Pathlayer actions per tool event
└── final.json         # agent_end snapshot
```

The recorder also keeps `state.db` at the runs root by default. It journals
normalized events, detector signals, actions and cumulative costs under the
OpenClaw `runId`, allowing an unfinished run to rebuild its exact pipeline
state after a listener restart. The raw JSONL journal and SQLite use the same
stable sequence-derived event IDs, so reattachment does not double-count old
events. A configured non-default `store.path` is honored instead.

SQLite cost counters describe only the analyzed and committed prefix, ending
at Pathlayer's STOP. `events.raw.jsonl` remains the source for the separate
recorder/reporting fold over every event, including the post-STOP suffix; OBS-2
must use that fold when reporting the real session spend rather than presenting
the SQLite snapshot as a total.

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
| `status`            | `started` / `orphan` / `completed` / `interrupted`             |
| `stopped`           | whether Pathlayer emitted a STOP during the run               |
| `state_durable`     | whether SQLite contains every analyzed tick recorded so far   |
| `persistence_error` | storage error category after fail-open, otherwise `null`      |
| `health`            | curated verdict; live recordings remain `unknown`             |
| `health_reason`     | observed OpenClaw/Pathlayer outcome awaiting human curation   |

A run reaching `agent_end` is `completed`, which says nothing about whether it
was sound. Live capture records OpenClaw's reported outcome and Pathlayer's
decision in `health_reason`, but always keeps `health` and the VAL-1 trace label
at `unknown` until an explicit human curation step assigns ground truth. An
absence of intervention is not evidence that the run was healthy, and a STOP
is a detector outcome rather than a curated failure class.

**A STOP does not end recording.** Pathlayer is a passive observer in V1: after
a STOP the detectors and arbitration stop running, but events keep being
normalized, kept in order and appended to `trace.json`. Those events appear in
`decisions.jsonl` with `"analyzed": false`.

Terminal runs (`completed`, `interrupted`) are immutable: a duplicated
`agent_end` or a late tool call for such a run is logged and dropped, never
replayed onto the run directory.

```bash
RUN=run-abc123   # original ctx.runId from manifest.json
cat "/var/lib/pathlayer/runs/$RUN/manifest.json"
cat "/var/lib/pathlayer/runs/$RUN/trace.json"
```

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

Or use the offline replay CLI documented in `traces/README.md`.

## Privacy warning

Recorded runs contain **prompts, tool arguments, and tool results** as sent by
OpenClaw. Treat `$PATHLAYER_RUNS_DIR` as sensitive data. Restrict filesystem
permissions (owner-only; Pathlayer's SQLite chmod is POSIX best-effort). Do not
commit run directories into git.

## Limitations (V1)

- Passive observer only — Pathlayer does **not** block OpenClaw tool calls yet,
  so a STOP is recorded and stops analysis without ending the OpenClaw run.
- No retention: run directories accumulate until you archive or delete them.
- Only the three forwarded hooks are recorded; LLM cost hooks are not wired.
- Hook payload shapes follow the public OpenClaw docs and may differ slightly
  on your gateway version — inspect `events.raw.jsonl` when in doubt.
- `ts` on ToolEvents reflects HTTP arrival order, not wall-clock time.

## Rollback

```bash
rm -rf ~/.openclaw/extensions/pathlayer-plugin && openclaw gateway restart
pkill -f "examples/openclaw_plugin/record.py" || true
```

## Demo listener runbook (legacy)

The step-by-step French runbook for first-time manual validation with
`listen.py` remains valid for smoke testing; prefer `record.py` for production
recording on `/var/lib/pathlayer/runs`.
