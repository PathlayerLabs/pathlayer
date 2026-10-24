"""Ready-to-run listener to try the OpenClaw adapter against a real gateway.

This script needs NO Python to be written or changed: it starts
`OpenClawAdapter`, prints the port it listens on, then prints each event live,
as OpenClaw sends them through the reference plugin (`pathlayer-plugin/`, in
this same folder).

Run it on the SAME machine as the OpenClaw gateway (see this folder's
README.md for the installation steps):

    python examples/openclaw_plugin/listen.py

To keep it running after an SSH disconnection, start it with nohup:

    nohup python examples/openclaw_plugin/listen.py > pathlayer.log 2>&1 &
    tail -f pathlayer.log

Stop: Ctrl+C in the foreground, or `pkill -f listen.py` when started in the
background with nohup.
"""

from __future__ import annotations

import os
import signal
import sys
import types
from datetime import datetime, timezone

from pathlayer.adapters.openclaw import DEFAULT_PORT, OpenClawAdapter
from pathlayer.events import ToolEvent

# Changes the port without touching the code, if 8787 is already taken on the
# host -- then set the same port in PATHLAYER_URL on the JavaScript plugin side
# (see pathlayer-plugin/index.js).
PORT = int(os.environ.get("PATHLAYER_PORT", DEFAULT_PORT))


def _print_event(event: ToolEvent) -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{now}] tool   = {event.tool!r}")
    print(f"           args   = {event.args}")
    if event.result is not None:
        print(f"           result = {event.result!r}")
    if event.metadata.get("error") is not None:
        print(f"           ERROR  = {event.metadata['error']!r}")
    print(flush=True)


def main() -> None:
    adapter = OpenClawAdapter(_print_event, port=PORT)
    adapter.attach()

    print("=" * 60)
    print("Pathlayer listener for OpenClaw -- ready.")
    print("=" * 60)
    print(f"Listening on: http://127.0.0.1:{adapter.port}/ingest")
    print("The OpenClaw plugin must post to this same URL: if you changed the")
    print("port, set PATHLAYER_URL for the gateway (see pathlayer-plugin/index.js).")
    print()
    print("Now give OpenClaw a simple task (e.g. read a file) and")
    print("watch the events appear below.")
    print("Ctrl+C to stop.\n", flush=True)

    def _stop(_signum: int, _frame: types.FrameType | None) -> None:
        print("\nListener stopped.")
        adapter.detach()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    signal.pause()


if __name__ == "__main__":
    main()
