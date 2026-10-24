"""Production OpenClaw recording listener (one run directory per ``ctx.runId``).

Unlike ``listen.py`` (print-only demo), this listener persists every forwarded
hook under ``$PATHLAYER_RUNS_DIR`` (default ``/var/lib/pathlayer/runs``) and
runs an independent Pathlayer pipeline per run.

Detector thresholds come from the standard Pathlayer config: pass
``--config path`` (or set ``$PATHLAYER_CONFIG``), or drop a
``.pathlayer/config.yaml`` next to where the listener runs and it is discovered
automatically. With no config file at all the built-in defaults apply, which is
exactly how the listener behaved before. An unreadable or invalid config is a
startup error, never a silent fallback to defaults.

Usage (on the same host as the OpenClaw gateway)::

    export PATHLAYER_RUNS_DIR=/var/lib/pathlayer/runs
    sudo mkdir -p "$PATHLAYER_RUNS_DIR"
    python examples/openclaw_plugin/record.py [--config .pathlayer/config.yaml] [--enforce]

``--enforce`` makes a STOP block the stopped run's later tool calls, through the
plugin's ``before_tool_call`` hook. Without it the listener only observes.

See ``README.md`` in this directory for OpenClaw plugin installation and config.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import types
from collections.abc import Sequence
from pathlib import Path

from pathlayer.adapters.openclaw import DEFAULT_PORT
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.config import ConfigError, PathlayerConfig, load_config

DEFAULT_RUNS_ROOT = Path("/var/lib/pathlayer/runs")
PORT = int(os.environ.get("PATHLAYER_PORT", DEFAULT_PORT))
RUNS_ROOT = Path(os.environ.get("PATHLAYER_RUNS_DIR", DEFAULT_RUNS_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pathlayer OpenClaw recording listener.")
    parser.add_argument(
        "--config",
        default=os.environ.get("PATHLAYER_CONFIG"),
        help=(
            "Pathlayer config file (default: $PATHLAYER_CONFIG, else discovery "
            "of ./.pathlayer/config.{yaml,yml,json}, else built-in defaults)"
        ),
    )
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT, help="Recorded runs root")
    parser.add_argument("--port", type=int, default=PORT, help="Loopback ingest port")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="block the tool calls of a run after Pathlayer stops it (default: observe only)",
    )
    return parser.parse_args(argv)


def load_listener_config(path: str | None) -> PathlayerConfig:
    """Load the Pathlayer config, or exit with a readable message.

    ``None`` keeps the documented discovery behaviour (``.pathlayer/config.*``
    in the working directory, built-in defaults when absent).
    """
    try:
        return load_config(path)
    except ConfigError as exc:
        raise SystemExit(f"invalid Pathlayer config: {exc}") from exc


def build_recorder(args: argparse.Namespace) -> OpenClawRunRecorder:
    """Build the recorder for parsed CLI args, failing fast on a bad config.

    Value-range violations (a window smaller than its own k, ...) are raised by
    the detectors themselves when the first pipeline is built, which the
    recorder does eagerly at construction -- so they surface here, at startup,
    like a malformed file does.
    """
    config = load_listener_config(args.config)
    try:
        return OpenClawRunRecorder(args.runs_root, config=config, enforce=args.enforce)
    except ConfigError as exc:
        raise SystemExit(f"invalid Pathlayer config: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    recorder = build_recorder(args)
    recorder.startup()
    adapter = recorder.attach_http(port=args.port)

    print("=" * 60)
    print("Pathlayer OpenClaw recorder -- ready.")
    print("=" * 60)
    print(f"Ingest URL : http://127.0.0.1:{adapter.port}/ingest")
    print(f"Runs root  : {recorder.runs_root}")
    print(f"Config     : {args.config or 'discovery (.pathlayer/config.*) or defaults'}")
    print("Hooks      : before_agent_run, llm_output, after_tool_call, agent_end, before_tool_call")
    if args.enforce:
        print("Enforcement: on -- a STOP blocks the run's later tool calls")
    else:
        print("Enforcement: off -- observe only (start with --enforce to block after a STOP)")
    print("Ctrl+C to stop.\n", flush=True)

    def _stop(_signum: int, _frame: types.FrameType | None) -> None:
        print("\nStopping recorder.")
        adapter.detach()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    signal.pause()


if __name__ == "__main__":
    main()
