"""Recording listener, kept for setups started from a source checkout.

Same options and environment variables as ``pathlayer-openclaw record``, the
command ``pip install pathlayer`` installs; prefer it::

    pathlayer-openclaw record [--config .pathlayer/config.yaml] [--enforce]

See ``README.md`` in this directory for OpenClaw plugin installation and config.
"""

from __future__ import annotations

import sys

from pathlayer.adapters.openclaw_cli import main

if __name__ == "__main__":
    raise SystemExit(main(["record", *sys.argv[1:]]))
