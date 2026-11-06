"""``pathlayer-openclaw``: run Pathlayer next to an OpenClaw gateway.

``pip install pathlayer`` is all an OpenClaw host needs. Three subcommands:

- ``record`` starts the recording listener: one durable directory per OpenClaw
  ``runId`` under the runs root and an independent pipeline per run. With
  ``--enforce``, a STOP blocks the stopped run's later tool calls through the
  plugin's ``before_tool_call`` hook; without it the listener only observes.
- ``plugin-path`` prints the directory of the OpenClaw plugin shipped in this
  package.
- ``install-plugin`` copies that plugin into OpenClaw's extensions directory.

Detector thresholds come from the standard Pathlayer config: ``--config path``,
else ``$PATHLAYER_CONFIG``, else a ``.pathlayer/config.{yaml,yml,json}`` in the
working directory, else the built-in defaults. An unreadable or invalid config
is a startup error, never a silent fallback to defaults.

See ``examples/openclaw_plugin/README.md`` in the repository for the OpenClaw
configuration and the files a run leaves behind.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import sys
import threading
import types
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from pathlayer.adapters.openclaw import DEFAULT_PORT
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_run_store import runs_root_from_env
from pathlayer.config import ConfigError, PathlayerConfig, load_config

__all__ = [
    "DEFAULT_PLUGIN_DEST",
    "PLUGIN_DIR",
    "PLUGIN_FILES",
    "PluginInstallError",
    "build_parser",
    "build_recorder",
    "install_plugin",
    "load_listener_config",
    "main",
    "parse_args",
    "run_record",
]

#: The OpenClaw plugin shipped as package data, next to this module.
PLUGIN_DIR = Path(__file__).resolve().parent / "openclaw_plugin"
PLUGIN_FILES = ("index.js", "openclaw.plugin.json", "package.json")
#: OpenClaw scans this directory for plugins when its gateway starts.
DEFAULT_PLUGIN_DEST = Path("~/.openclaw/extensions/pathlayer-plugin")

CONFIG_ENV = "PATHLAYER_CONFIG"
PORT_ENV = "PATHLAYER_PORT"
#: Held by the listener recording into a runs root, for as long as it runs.
RUNS_ROOT_LOCK = ".listener.lock"
_HOST = "127.0.0.1"
# A timed wait, so a Ctrl+C reaches its handler on Windows as well, where an
# untimed Event.wait() cannot be interrupted.
_STOP_POLL_SECONDS = 0.5


class PluginInstallError(Exception):
    """The bundled plugin cannot be installed where it was asked to go."""


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        port = -1
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"invalid port {value!r} (from --port or ${PORT_ENV}): expected 0 to 65535"
        )
    return port


def build_parser() -> argparse.ArgumentParser:
    """The ``pathlayer-openclaw`` parser. Environment defaults are read here."""
    parser = argparse.ArgumentParser(
        prog="pathlayer-openclaw",
        description="Run Pathlayer next to an OpenClaw gateway.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    record = commands.add_parser(
        "record",
        help="start the recording listener",
        description=(
            "Record every OpenClaw run forwarded by the Pathlayer plugin, one "
            "directory per runId, and analyze each run with its own pipeline."
        ),
    )
    record.add_argument(
        "--config",
        default=os.environ.get(CONFIG_ENV),
        help=(
            f"Pathlayer config file (default: ${CONFIG_ENV}, else discovery "
            "of ./.pathlayer/config.{yaml,yml,json}, else built-in defaults)"
        ),
    )
    record.add_argument(
        "--runs-root",
        type=Path,
        default=runs_root_from_env(),
        help="recorded runs root (default: $PATHLAYER_RUNS_DIR, else /var/lib/pathlayer/runs)",
    )
    record.add_argument(
        "--port",
        type=_port,
        # A string default goes through ``type`` too: a bad $PATHLAYER_PORT
        # gets the same readable error as a bad --port.
        default=os.environ.get(PORT_ENV, str(DEFAULT_PORT)),
        help=f"loopback ingest port (default: ${PORT_ENV}, else {DEFAULT_PORT}; 0: any free port)",
    )
    record.add_argument(
        "--enforce",
        action="store_true",
        help="block the tool calls of a run after Pathlayer stops it (default: observe only)",
    )

    commands.add_parser(
        "plugin-path",
        help="print the directory of the OpenClaw plugin shipped with Pathlayer",
        description=(
            "Print the directory of the bundled OpenClaw plugin, for instance for "
            '`openclaw plugins install -l "$(pathlayer-openclaw plugin-path)"`.'
        ),
    )

    install = commands.add_parser(
        "install-plugin",
        help="copy the bundled plugin into OpenClaw's extensions directory",
        description=(
            "Copy the bundled OpenClaw plugin's files into --dest. Files already "
            "there are never overwritten without --force, and nothing is deleted."
        ),
    )
    install.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_PLUGIN_DEST,
        help=f"plugin directory to create (default: {DEFAULT_PLUGIN_DEST})",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="overwrite the plugin files already in --dest",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_listener_config(path: str | None) -> PathlayerConfig:
    """Load the Pathlayer config, or exit with a readable message.

    ``None`` keeps the documented discovery behaviour (``.pathlayer/config.*``
    in the working directory, built-in defaults when absent).
    """
    try:
        return load_config(path)
    except ConfigError as exc:
        raise SystemExit(f"invalid Pathlayer config: {exc}") from exc


def _concerns(error: OSError, root: Path) -> bool:
    """Whether ``error`` is about ``root``, one of its parents or a file inside it."""
    if not isinstance(error.filename, str | bytes | os.PathLike):
        return False
    path = Path(os.fsdecode(error.filename)).absolute()
    root = root.absolute()
    return path == root or path in root.parents or root in path.parents


def build_recorder(args: argparse.Namespace) -> OpenClawRunRecorder:
    """Build the recorder for parsed ``record`` args, failing fast on a bad setup.

    Value-range violations (a window smaller than its own k, ...) are raised by
    the detectors themselves when the first pipeline is built, which the
    recorder does eagerly at construction -- so they surface here, at startup,
    like a malformed file does. So does a runs root the listener cannot use.
    """
    config = load_listener_config(args.config)
    try:
        return OpenClawRunRecorder(args.runs_root, config=config, enforce=args.enforce)
    except ConfigError as exc:
        raise SystemExit(f"invalid Pathlayer config: {exc}") from exc
    except OSError as exc:
        message = f"cannot start the OpenClaw recorder: {exc}"
        if _concerns(exc, args.runs_root):
            root = args.runs_root
            message += (
                f"\nRuns are recorded under {root}, which must be a directory you own. "
                f'Create it (sudo mkdir -p "{root}" && sudo chown "$USER" "{root}") '
                "or choose another one with --runs-root or $PATHLAYER_RUNS_DIR."
            )
        raise SystemExit(message) from exc


def _port_in_use(port: int, error: OSError) -> SystemExit:
    return SystemExit(
        f"cannot listen on {_HOST}:{port}: {error}\n"
        "Is another Pathlayer listener already running? Stop it, or choose another "
        f"port with --port or ${PORT_ENV} and give the gateway the same one "
        "(PATHLAYER_URL in its environment)."
    )


def _check_port_free(port: int) -> None:
    """Fail before the startup sweep if the port is taken.

    The sweep marks every unfinished run under the runs root as interrupted. If
    the port is taken, it is most likely by a listener already recording into
    the same root, whose live runs the sweep would close.
    """
    if port == 0:
        return
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name != "nt":
            # Same option as the ingest server: a port left in TIME_WAIT by the
            # previous listener is free; one with a listener on it is not.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((_HOST, port))
    except OSError as exc:
        raise _port_in_use(port, exc) from exc
    finally:
        probe.close()


def _try_lock(handle: IO[str]) -> None:
    """Take an exclusive lock on ``handle`` without waiting; ``OSError`` if held."""
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def lock_runs_root(root: Path) -> IO[str]:
    """Hold ``root`` for this listener, or exit if another listener holds it.

    The startup sweep marks every unfinished run under the root as interrupted,
    so a second listener on the same root would close the first one's live
    runs, whatever its port. The lock is the open file returned: closing it, or
    the process ending in any way, crash included, releases it, so a lock file
    left behind never blocks a restart.
    """
    path = root / RUNS_ROOT_LOCK
    handle = path.open("a+", encoding="utf-8")
    try:
        _try_lock(handle)
    except OSError as exc:
        try:
            handle.seek(0)
            holder = handle.read().strip()
        except OSError:  # Windows: the holder's locked byte cannot be read
            holder = ""
        handle.close()
        process = f" (process {holder})" if holder.isdigit() else ""
        raise SystemExit(
            f"another Pathlayer listener{process} is already recording into {root}.\n"
            "A second one would mark its live runs interrupted. Stop it, or give this "
            "listener its own runs root with --runs-root or $PATHLAYER_RUNS_DIR."
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _wait_for_stop(stop: threading.Event) -> None:
    """Block until SIGINT or SIGTERM (or ``stop`` is set), then restore the handlers."""

    def _handler(_signum: int, _frame: types.FrameType | None) -> None:
        stop.set()

    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in signals}
    for signum in signals:
        signal.signal(signum, _handler)
    try:
        while not stop.wait(_STOP_POLL_SECONDS):
            pass
    finally:
        for signum, handler in previous.items():
            # None: a handler not installed from Python; the default is the
            # closest one Python can put back.
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)


def run_record(args: argparse.Namespace, *, stop: threading.Event | None = None) -> int:
    """Run the recording listener until SIGINT or SIGTERM (or ``stop`` is set)."""
    _check_port_free(args.port)
    recorder = build_recorder(args)
    # Held before the startup sweep and until the listener stops: see lock_runs_root.
    with lock_runs_root(recorder.runs_root):
        recorder.startup()
        try:
            adapter = recorder.attach_http(port=args.port)
        except OSError as exc:
            raise _port_in_use(args.port, exc) from exc

        try:
            print("=" * 60)
            print("Pathlayer OpenClaw recorder -- ready.")
            print("=" * 60)
            print(f"Ingest URL : http://{_HOST}:{adapter.port}/ingest")
            print(f"Runs root  : {recorder.runs_root}")
            print(f"Config     : {args.config or 'discovery (.pathlayer/config.*) or defaults'}")
            print(
                "Hooks      : before_agent_run, llm_output, after_tool_call, agent_end, "
                "before_tool_call"
            )
            if args.enforce:
                print("Enforcement: on -- a STOP blocks the run's later tool calls")
            else:
                print(
                    "Enforcement: off -- observe only (start with --enforce to block after a STOP)"
                )
            print("Ctrl+C to stop.\n", flush=True)
            _wait_for_stop(stop or threading.Event())
        finally:
            adapter.detach()
    print("Stopping recorder.", flush=True)
    return 0


def _missing_plugin_files() -> list[str]:
    return [name for name in PLUGIN_FILES if not (PLUGIN_DIR / name).is_file()]


def _incomplete_plugin(missing: list[str]) -> str:
    return (
        f"the OpenClaw plugin shipped with Pathlayer is incomplete in {PLUGIN_DIR} "
        f"(missing: {', '.join(missing)}); reinstall pathlayer"
    )


def install_plugin(dest: Path, *, force: bool = False) -> list[Path]:
    """Copy the bundled plugin's files into ``dest``; return the files written.

    Without ``force``, refuses when any of the plugin's files is already in
    ``dest``, before writing anything. With it, overwrites those files only:
    nothing in ``dest`` is ever deleted.
    """
    missing = _missing_plugin_files()
    if missing:
        raise PluginInstallError(_incomplete_plugin(missing))
    dest = dest.expanduser()
    if os.path.lexists(dest) and not dest.is_dir():
        raise PluginInstallError(f"{dest} exists and is not a directory")
    present = [name for name in PLUGIN_FILES if os.path.lexists(dest / name)]
    if present and not force:
        raise PluginInstallError(
            f"{dest} already holds plugin files ({', '.join(present)}); "
            "pass --force to overwrite them"
        )
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in PLUGIN_FILES:
        target = dest / name
        shutil.copyfile(PLUGIN_DIR / name, target)
        written.append(target)
    return written


def _install(dest: Path, *, force: bool) -> int:
    try:
        written = install_plugin(dest, force=force)
    except (PluginInstallError, OSError) as exc:
        print(f"cannot install the OpenClaw plugin: {exc}", file=sys.stderr)
        return 1
    directory = written[0].parent
    print(f"Pathlayer plugin installed in {directory}")
    print("Next steps:")
    print("  1. Allow its conversation hooks in your OpenClaw config:")
    print(
        '     {"plugins": {"entries": {"pathlayer-plugin": '
        '{"hooks": {"allowConversationAccess": true}}}}}'
    )
    print("  2. Restart the gateway: openclaw gateway restart")
    print(
        "  3. Start the listener: pathlayer-openclaw record "
        "(add --enforce to block a stopped run's tool calls)"
    )
    return 0


def _plugin_path() -> int:
    missing = _missing_plugin_files()
    if missing:
        print(_incomplete_plugin(missing), file=sys.stderr)
        return 1
    print(PLUGIN_DIR)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "record":
        return run_record(args)
    if args.command == "plugin-path":
        return _plugin_path()
    return _install(args.dest, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
