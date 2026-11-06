"""The ``pathlayer-openclaw`` command: what ``pip install pathlayer`` gives an OpenClaw host.

Three promises are pinned here. ``record`` behaves like the listener script it
replaces and stops cleanly on a signal, without ``signal.pause()``. The plugin
ships inside the package, complete and at the package's version.
``install-plugin`` copies it without ever overwriting a file unasked or deleting
one.
"""

from __future__ import annotations

import errno
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import pathlayer
from pathlayer.adapters import openclaw_cli
from pathlayer.adapters.openclaw import DEFAULT_PORT
from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_run_store import DEFAULT_RUNS_ROOT, RunStore

_ROOT = Path(__file__).resolve().parents[1]
_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX signals and permissions")


def _ctx(run_id: str) -> dict[str, Any]:
    return {"runId": run_id, "sessionKey": "sess-cli", "agentId": "agent-1"}


def _payloads(run_id: str) -> list[dict[str, Any]]:
    return [
        {"hook": "before_agent_run", "event": {"prompt": "fix the bug"}, "ctx": _ctx(run_id)},
        {
            "hook": "after_tool_call",
            "event": {
                "toolName": "read",
                "params": {"path": "a.py"},
                "result": "ok",
                "toolCallId": "call-1",
                "runId": run_id,
            },
            "ctx": _ctx(run_id),
        },
        {"hook": "agent_end", "event": {"success": True}, "ctx": _ctx(run_id)},
    ]


def _post(port: int, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/ingest",
        data=json.dumps({"source": "openclaw", "payload": payload}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200


def _post_when_listening(port: int, payloads: list[dict[str, Any]]) -> None:
    deadline = time.monotonic() + 10
    while True:
        try:
            _post(port, payloads[0])
            break
        except urllib.error.URLError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    for payload in payloads[1:]:
        _post(port, payload)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def _status(runs_root: Path, run_id: str) -> str:
    run = RunStore(runs_root).resolve_run(run_id)
    assert run is not None
    status: str = run.read_manifest()["status"]
    return status


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PATHLAYER_CONFIG", "PATHLAYER_RUNS_DIR", "PATHLAYER_PORT"):
        monkeypatch.delenv(name, raising=False)


# --- options -----------------------------------------------------------------


def test_record_options_default_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_env(monkeypatch)
    args = openclaw_cli.parse_args(["record"])
    assert (args.config, args.runs_root, args.port, args.enforce) == (
        None,
        DEFAULT_RUNS_ROOT,
        DEFAULT_PORT,
        False,
    )

    monkeypatch.setenv("PATHLAYER_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("PATHLAYER_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("PATHLAYER_PORT", "9911")
    args = openclaw_cli.parse_args(["record", "--enforce"])
    assert (args.config, args.runs_root, args.port, args.enforce) == (
        str(tmp_path / "config.yaml"),
        tmp_path / "runs",
        9911,
        True,
    )
    assert openclaw_cli.parse_args(["record", "--port", "0"]).port == 0


@pytest.mark.parametrize(("env", "argv"), [("abc", []), (None, ["--port", "70000"])])
def test_an_invalid_port_is_a_readable_usage_error(
    env: str | None,
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clean_env(monkeypatch)
    if env is not None:
        monkeypatch.setenv("PATHLAYER_PORT", env)
    with pytest.raises(SystemExit) as exit_info:
        openclaw_cli.parse_args(["record", *argv])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "invalid port" in err
    assert "(from --port or $PATHLAYER_PORT): expected 0 to 65535" in err


def test_a_subcommand_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        openclaw_cli.main([])
    assert exit_info.value.code == 2
    assert "COMMAND" in capsys.readouterr().err


# --- the plugin shipped in the package ----------------------------------------


def test_the_package_ships_exactly_the_plugin_files() -> None:
    # Dotfiles (.DS_Store) are git-ignored, so never built into the package.
    shipped = sorted(
        path.name for path in openclaw_cli.PLUGIN_DIR.iterdir() if not path.name.startswith(".")
    )
    assert shipped == sorted(openclaw_cli.PLUGIN_FILES)


def test_the_plugin_version_is_the_package_version() -> None:
    plugin = json.loads((openclaw_cli.PLUGIN_DIR / "package.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert plugin["version"] == pyproject["project"]["version"] == pathlayer.__version__


def test_the_plugin_loads_when_the_gateway_starts() -> None:
    # OpenClaw no longer loads at startup a plugin that does not say so.
    manifest_path = openclaw_cli.PLUGIN_DIR / "openclaw.plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["activation"] == {"onStartup": True}
    assert manifest["id"] == "pathlayer-plugin"


def test_plugin_path_prints_the_bundled_plugin(capsys: pytest.CaptureFixture[str]) -> None:
    assert openclaw_cli.main(["plugin-path"]) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed == openclaw_cli.PLUGIN_DIR
    assert all((printed / name).is_file() for name in openclaw_cli.PLUGIN_FILES)


def test_plugin_path_fails_on_an_incomplete_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "index.js").write_text("// partial", encoding="utf-8")
    monkeypatch.setattr(openclaw_cli, "PLUGIN_DIR", tmp_path)

    assert openclaw_cli.main(["plugin-path"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing: openclaw.plugin.json, package.json" in captured.err
    assert "reinstall pathlayer" in captured.err


# --- install-plugin -------------------------------------------------------------


def _bundled(name: str) -> bytes:
    return (openclaw_cli.PLUGIN_DIR / name).read_bytes()


def test_install_plugin_copies_the_plugin_and_says_what_comes_next(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "extensions" / "pathlayer-plugin"

    assert openclaw_cli.main(["install-plugin", "--dest", str(dest)]) == 0

    assert sorted(path.name for path in dest.iterdir()) == sorted(openclaw_cli.PLUGIN_FILES)
    for name in openclaw_cli.PLUGIN_FILES:
        assert (dest / name).read_bytes() == _bundled(name)
    out = capsys.readouterr().out
    assert f"installed in {dest}" in out
    assert '"allowConversationAccess": true' in out
    assert "openclaw gateway restart" in out
    assert "pathlayer-openclaw record" in out


def test_install_plugin_defaults_to_the_openclaw_extensions_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert openclaw_cli.main(["install-plugin"]) == 0

    installed = tmp_path / ".openclaw" / "extensions" / "pathlayer-plugin" / "index.js"
    assert installed.read_bytes() == _bundled("index.js")


def test_install_plugin_never_overwrites_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "pathlayer-plugin"
    dest.mkdir()
    (dest / "index.js").write_text("// local edit", encoding="utf-8")

    assert openclaw_cli.main(["install-plugin", "--dest", str(dest)]) == 1

    err = capsys.readouterr().err
    assert "already holds plugin files (index.js)" in err
    assert "--force" in err
    # Refused before writing anything.
    assert sorted(path.name for path in dest.iterdir()) == ["index.js"]
    assert (dest / "index.js").read_text(encoding="utf-8") == "// local edit"

    assert openclaw_cli.main(["install-plugin", "--dest", str(dest), "--force"]) == 0
    assert (dest / "index.js").read_bytes() == _bundled("index.js")


def test_install_plugin_never_deletes_a_file(tmp_path: Path) -> None:
    dest = tmp_path / "pathlayer-plugin"
    dest.mkdir()
    (dest / "notes.txt").write_text("mine", encoding="utf-8")
    openclaw_cli.install_plugin(dest)

    openclaw_cli.install_plugin(dest, force=True)

    assert (dest / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_install_plugin_refuses_a_destination_that_is_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "pathlayer-plugin"
    dest.write_text("not a directory", encoding="utf-8")

    assert openclaw_cli.main(["install-plugin", "--dest", str(dest), "--force"]) == 1

    assert "is not a directory" in capsys.readouterr().err
    assert dest.read_text(encoding="utf-8") == "not a directory"


def test_install_plugin_refuses_an_incomplete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(openclaw_cli, "PLUGIN_DIR", tmp_path / "empty")
    dest = tmp_path / "pathlayer-plugin"

    assert openclaw_cli.main(["install-plugin", "--dest", str(dest)]) == 1

    assert "reinstall pathlayer" in capsys.readouterr().err
    assert not dest.exists()


def test_install_plugin_reports_a_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _denied(_src: Path, _dst: Path) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", str(_dst))

    monkeypatch.setattr(openclaw_cli.shutil, "copyfile", _denied)

    assert openclaw_cli.main(["install-plugin", "--dest", str(tmp_path / "plugin")]) == 1

    assert "cannot install the OpenClaw plugin: [Errno 13]" in capsys.readouterr().err


# --- record -------------------------------------------------------------------


def test_record_records_runs_until_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    port = _free_port()
    runs_root = tmp_path / "runs"
    args = openclaw_cli.parse_args(["record", "--port", str(port), "--runs-root", str(runs_root)])
    handlers = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    stop = threading.Event()
    errors: list[BaseException] = []

    def _drive() -> None:
        try:
            _post_when_listening(port, _payloads("run-cli"))
        except BaseException as exc:  # reported by the main thread
            errors.append(exc)
        finally:
            stop.set()

    driver = threading.Thread(target=_drive)
    driver.start()
    assert openclaw_cli.run_record(args, stop=stop) == 0
    driver.join(timeout=10)

    assert errors == []
    assert _status(runs_root, "run-cli") == "completed"
    out = capsys.readouterr().out
    assert f"Ingest URL : http://127.0.0.1:{port}/ingest" in out
    assert "Enforcement: off" in out
    assert out.rstrip().endswith("Stopping recorder.")
    # The listener let go of its port and of the signal handlers.
    openclaw_cli._check_port_free(port)
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == handlers


@_POSIX_ONLY
def test_a_stop_signal_ends_the_wait_and_restores_the_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_int = signal.getsignal(signal.SIGINT)
    original_term = signal.getsignal(signal.SIGTERM)
    real_getsignal = signal.getsignal
    # A handler installed outside Python reads as None; SIG_DFL is put back.
    monkeypatch.setattr(
        signal,
        "getsignal",
        lambda signum: None if signum == signal.SIGINT else real_getsignal(signum),
    )
    stop = threading.Event()
    sent: list[int] = []

    def _send_sigterm_once_handled() -> None:
        deadline = time.monotonic() + 10
        while real_getsignal(signal.SIGTERM) is original_term:
            if time.monotonic() > deadline:
                stop.set()  # never SIGTERM the test process without the handler
                return
            time.sleep(0.01)
        sent.append(signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)

    sender = threading.Thread(target=_send_sigterm_once_handled)
    sender.start()
    try:
        openclaw_cli._wait_for_stop(stop)
        sender.join(timeout=10)
        assert sent == [signal.SIGTERM], "the handler was never installed"
        assert stop.is_set()
        assert real_getsignal(signal.SIGTERM) is original_term
        assert real_getsignal(signal.SIGINT) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGINT, original_int)


def _seed_started_run(runs_root: Path, run_id: str) -> None:
    OpenClawRunRecorder(runs_root).process_payload(_payloads(run_id)[0])
    assert _status(runs_root, run_id) == "started"


def test_record_refuses_a_taken_port_before_sweeping_the_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    # A run that a listener already on this port could still be recording.
    _seed_started_run(runs_root, "run-live")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        args = openclaw_cli.parse_args(
            ["record", "--port", str(port), "--runs-root", str(runs_root)]
        )

        with pytest.raises(SystemExit, match=rf"cannot listen on 127\.0\.0\.1:{port}") as info:
            openclaw_cli.run_record(args)

    assert "--port or $PATHLAYER_PORT" in str(info.value)
    assert _status(runs_root, "run-live") == "started"
    openclaw_cli._check_port_free(0)  # port 0 is always free


def test_a_second_listener_on_the_same_runs_root_never_sweeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two listeners, two ports, one runs root: the second one's startup sweep
    # would mark the first one's live run interrupted.
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    first_port, second_port = _free_port(), _free_port()
    first = openclaw_cli.parse_args(
        ["record", "--port", str(first_port), "--runs-root", str(runs_root)]
    )
    second = openclaw_cli.parse_args(
        ["record", "--port", str(second_port), "--runs-root", str(runs_root)]
    )
    stop = threading.Event()
    refusals: list[str] = []
    statuses: list[str] = []
    errors: list[BaseException] = []

    def _start_a_second_listener() -> None:
        try:
            _post_when_listening(first_port, _payloads("run-live")[:1])
            statuses.append(_status(runs_root, "run-live"))
            try:
                openclaw_cli.run_record(second)
            except SystemExit as exc:
                refusals.append(str(exc))
            statuses.append(_status(runs_root, "run-live"))
            openclaw_cli._check_port_free(second_port)  # the second one never listened
        except BaseException as exc:  # reported by the main thread
            errors.append(exc)
        finally:
            stop.set()

    driver = threading.Thread(target=_start_a_second_listener)
    driver.start()
    assert openclaw_cli.run_record(first, stop=stop) == 0
    driver.join(timeout=10)

    assert errors == []
    assert statuses == ["started", "started"]
    [refusal] = refusals
    assert f"another Pathlayer listener (process {os.getpid()}) is already recording" in refusal
    assert "--runs-root or $PATHLAYER_RUNS_DIR" in refusal
    # Once the first listener has stopped, the root is free again.
    openclaw_cli.lock_runs_root(runs_root).close()


def test_a_lock_file_left_by_a_dead_listener_never_blocks_a_restart(tmp_path: Path) -> None:
    (tmp_path / openclaw_cli.RUNS_ROOT_LOCK).write_text("12345\n", encoding="utf-8")

    with openclaw_cli.lock_runs_root(tmp_path) as lock:
        assert not lock.closed
        with pytest.raises(SystemExit, match="already recording into"):
            openclaw_cli.lock_runs_root(tmp_path)
    assert (tmp_path / openclaw_cli.RUNS_ROOT_LOCK).read_text(encoding="utf-8") == (
        f"{os.getpid()}\n"
    )
    openclaw_cli.lock_runs_root(tmp_path).close()


def test_a_port_taken_after_the_check_is_still_a_readable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(openclaw_cli, "_check_port_free", lambda _port: None)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        args = openclaw_cli.parse_args(
            ["record", "--port", str(port), "--runs-root", str(tmp_path / "runs")]
        )

        with pytest.raises(SystemExit, match="cannot listen on"):
            openclaw_cli.run_record(args)


@_POSIX_ONLY
def test_record_names_the_runs_root_it_cannot_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can write anywhere")
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    runs_root = locked / "runs"
    try:
        args = openclaw_cli.parse_args(["record", "--runs-root", str(runs_root)])
        with pytest.raises(SystemExit) as info:
            openclaw_cli.build_recorder(args)
    finally:
        locked.chmod(0o700)

    message = str(info.value)
    assert message.startswith("cannot start the OpenClaw recorder: [Errno 13]")
    assert f"Runs are recorded under {runs_root}" in message
    assert "--runs-root or $PATHLAYER_RUNS_DIR" in message


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EACCES, "Permission denied", "/elsewhere/state.db"),
        OSError(errno.EIO, "I/O error"),
    ],
)
def test_an_unrelated_startup_error_gets_no_runs_root_hint(
    error: OSError, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    def _failing(*_args: object, **_kwargs: object) -> OpenClawRunRecorder:
        raise error

    monkeypatch.setattr(openclaw_cli, "OpenClawRunRecorder", _failing)
    args = openclaw_cli.parse_args(["record", "--runs-root", str(tmp_path / "runs")])

    with pytest.raises(SystemExit) as info:
        openclaw_cli.build_recorder(args)

    assert str(info.value) == f"cannot start the OpenClaw recorder: {error}"


def _subprocess_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PATHLAYER_CONFIG", "PATHLAYER_RUNS_DIR", "PATHLAYER_PORT"}
    }
    # The checkout's package, whatever the interpreter has installed.
    env["PYTHONPATH"] = str(_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHON_COLORS"] = "0"  # plain argparse output on 3.14 whatever the CI sets
    return env


@_POSIX_ONLY
def test_the_record_command_stops_cleanly_on_sigterm(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pathlayer.adapters.openclaw_cli",
            "record",
            "--port",
            "0",
            "--runs-root",
            str(runs_root),
        ],
        cwd=tmp_path,
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: queue.Queue[str] = queue.Queue()
    assert process.stdout is not None
    reader = threading.Thread(
        target=lambda: [lines.put(line) for line in process.stdout or ()], daemon=True
    )
    reader.start()
    output: list[str] = []
    try:
        port = None
        deadline = time.monotonic() + 30
        while port is None:
            line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
            output.append(line)
            match = re.search(r"Ingest URL : http://127\.0\.0\.1:(\d+)/ingest", line)
            if match:
                port = int(match.group(1))
        _post_when_listening(port, _payloads("run-sub"))
        assert _status(runs_root, "run-sub") == "completed"

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    reader.join(timeout=5)
    while not lines.empty():
        output.append(lines.get())
    assert "Stopping recorder." in "".join(output)


def test_the_record_script_is_the_record_command(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(_ROOT / "examples" / "openclaw_plugin" / "record.py"), "--help"],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("usage: pathlayer-openclaw record")
    assert "--enforce" in completed.stdout
