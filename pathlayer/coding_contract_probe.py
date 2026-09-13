"""Capture-time evaluator for compiled ANC-4 coding contracts.

``pathlayer.coding_contracts`` compiles a declarative contract into an ANC
bundle whose ``probe.py`` only carries the compiled contract as data and calls
:func:`run_compiled_contract`. This module is that probe's body: it observes the
workspace once and prints the ANC-2 wire shape ``{"checkpoints": {id: bool}}``.

It runs at *capture* time, in the probe's own process, never inside the pure
detection loop, and it never raises: like
:meth:`~pathlayer.detectors.proof_producer.ProgressProofProducer.probe`, a
failure degrades to less evidence, not to an error in the observed run.

Each check reports either a **fact** or **nothing**:

- a fact is a deterministic property of the current workspace: a required test
  did not pass (failed, skipped, deleted or renamed), a command exited
  non-zero, a file is missing or differs from its baseline;
- a check that cannot observe its resource -- a directory or symlink where a
  regular file is expected, a path resolving outside the workspace, a test
  module that cannot be imported (possibly a dependency missing from this
  interpreter, not the agent's doing), pytest or a command that cannot start or
  times out, an unreadable report, an exhausted time budget -- is *omitted*
  from the proof and its reason is printed on standard error from a fixed
  vocabulary (never exception text).

An omitted checkpoint can neither create nor erase progress: ANC-2 names it in
a ``progress_proof_unobserved`` marker, the ANC-1 anchor keeps its last known
value for later, and ``no_progress`` stays silent on that tool call (a gap).
When nothing at all can be observed the probe exits non-zero and ANC-2
attaches the marker alone.

Observation units are isolated from one another: every file check and every
command stands alone, and all test checkpoints share one pytest session per
tick (one session per check would multiply the per-tool-call cost). When that
session cannot complete, every test checkpoint is omitted for that tick.

Tests run with the contract's ``python`` interpreter -- typically the
project's own virtualenv -- or, by default, the interpreter running the probe.
The private pytest plugin is copied next to each session under a random module
name, so it works where Pathlayer is not installed and cannot be shadowed by a
file the agent planted in the workspace.

The filesystem semantics are those of the ARCH-1B collector
(:func:`~pathlayer.state_collectors.collect_filesystem_evidence`): only regular
files are defined. Subprocesses reuse its process-group termination so a timed
out pytest or command cannot outlive the probe.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TextIO

from pathlayer import _contract_pytest_plugin as _plugin
from pathlayer._contract_pytest_plugin import (
    COLLECT_ENV,
    COLLECT_ERRORS_ENV,
    MAX_NODEID_LENGTH,
    MAX_OUTCOMES,
    OUTPUT_ENV,
    SELECT_ENV,
)
from pathlayer.state_collectors import _wait_for_pytest, collect_filesystem_evidence

PAYLOAD_FORMAT = "pathlayer-coding-contract"
PAYLOAD_VERSION = 1

EXIT_OBSERVED = 0
EXIT_PATHLAYER_UNAVAILABLE = 2  # returned by the generated probe itself
EXIT_INVALID_CONTRACT = 3
EXIT_NOTHING_OBSERVED = 4

MISSING_FINGERPRINT = "missing"

# 0: all selected tests passed; 1: some failed or a module could not be
# collected (--continue-on-collection-errors); 5: nothing selected ran, e.g.
# every required test was deleted. Anything else (interrupted, internal error,
# usage error such as a broken conftest.py) is not a statement about the tests.
_PYTEST_FACT_STATUSES = frozenset({0, 1, 5})
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_MAX_CHECKS = 256
_SHA256_DIGEST = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT = re.compile(r"missing|sha256:[0-9a-f]{64}")
_PYTEST_MODULE = "pytest"
_MINIMUM_PYTHON = (3, 8)  # the private plugin's own floor

# Reasons are a fixed vocabulary: exception text can carry secrets and would
# make the probe's diagnostics depend on the host.
REASON_NOT_REGULAR = "path is not a regular file (directory, symlink or unreadable)"
REASON_OUTSIDE = "path resolves outside the workspace"
REASON_UNSAFE_PATH = "path is not a safe workspace-relative path"
REASON_WORKSPACE = "workspace is not an accessible directory"
REASON_BUDGET = "probe time budget exhausted"
REASON_PYTEST_TIMEOUT = "pytest timed out"
REASON_PYTEST_UNREADABLE = "pytest could not run or reported unreadable outcomes"
REASON_TEST_UNCOLLECTABLE = "test module could not be collected (import or syntax error)"
REASON_TEST_INTERPRETER = "test interpreter could not be launched"
REASON_COMMAND_TIMEOUT = "command timed out"
REASON_COMMAND_LAUNCH = "command could not be launched"
REASON_UNEXPECTED = "unexpected probe error"


class Unobservable(Exception):
    """A check could not observe its resource this tick (never a negative fact)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PytestCollectionError(Exception):
    """A compile-time ``tests_improve`` path selection could not be expanded."""


class InterpreterCheckError(Exception):
    """A compile-time test interpreter cannot run the probe's pytest session."""


# --------------------------------------------------------------------------- #
# Workspace paths and files
# --------------------------------------------------------------------------- #


def is_safe_relative_path(value: object) -> bool:
    """Whether ``value`` is a normalized, workspace-relative POSIX path."""
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).anchor:
        return False
    parts = value.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def contained_path(workspace: Path, relative: str) -> Path:
    """Return ``workspace/relative``, or raise :class:`Unobservable` if it escapes.

    ``workspace`` must already be resolved. Intermediate symlinks are allowed
    while they stay inside the workspace; a final symlink is rejected later by
    the filesystem collector.
    """
    if not is_safe_relative_path(relative):
        raise Unobservable(REASON_UNSAFE_PATH)
    target = workspace.joinpath(*relative.split("/"))
    try:
        resolved = target.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:  # RuntimeError: symlink loop
        raise Unobservable(REASON_NOT_REGULAR) from error
    if resolved != workspace and workspace not in resolved.parents:
        raise Unobservable(REASON_OUTSIDE)
    return target


def file_fingerprint(workspace: Path, relative: str) -> str:
    """``"missing"`` or ``"sha256:<hex>"`` for one workspace file.

    The same function captures a ``file_changes`` baseline at compile time and
    the current state at capture time, so the two are always comparable.
    """
    target = contained_path(workspace, relative)
    # A standalone observation: "before" is the collector's unpaired phase.
    evidence = collect_filesystem_evidence(
        target, tool_event_ref="coding-contract", resource_id="coding-contract", phase="before"
    )
    fingerprint = evidence.canonical_fingerprint
    if evidence.status != "observed" or not isinstance(fingerprint, str):
        raise Unobservable(REASON_NOT_REGULAR)
    return fingerprint


def _regular_file(workspace: Path, relative: str) -> os.stat_result | None:
    """``lstat`` of a regular workspace file, ``None`` if missing, else unobservable."""
    target = contained_path(workspace, relative)
    try:
        metadata = target.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, ValueError) as error:
        raise Unobservable(REASON_NOT_REGULAR) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise Unobservable(REASON_NOT_REGULAR)
    return metadata


# --------------------------------------------------------------------------- #
# Subprocesses (pytest and commands)
# --------------------------------------------------------------------------- #


def _launch(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None, timeout: float
) -> int | None:
    """Run ``argv`` without a shell in its own process group; ``None`` on timeout.

    ``_wait_for_pytest`` is the ARCH-1B helper that kills the whole process
    tree on timeout (``killpg`` on POSIX, ``taskkill /T`` on Windows); despite
    its name nothing in it is pytest-specific.
    """
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=None if env is None else dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=sys.platform != "win32",
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform == "win32" else 0
        ),
    )
    return _wait_for_pytest(process, timeout)


@contextlib.contextmanager
def _pytest_scratch(prefix: str) -> Iterator[tuple[Path, str]]:
    """A private temporary directory holding a fresh copy of the plugin.

    Yields the directory and the plugin's module name. The name is random so a
    module the agent writes into the workspace (first on ``sys.path`` under
    ``python -m``) cannot impersonate it.
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as name:
        scratch = Path(name)
        module = f"_pathlayer_contract_plugin_{secrets.token_hex(8)}"
        (scratch / f"{module}.py").write_bytes(Path(_plugin.__file__).read_bytes())
        yield scratch, module


def _pytest_command(python: str, plugin: str, *arguments: str) -> list[str]:
    return [
        python,
        "-m",
        _PYTEST_MODULE,
        "-q",
        "--tb=no",
        "--rootdir=.",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-p",
        plugin,
        *arguments,
    ]


def _pytest_env(scratch: Path, extra: Mapping[str, str]) -> dict[str, str]:
    """The project's own pytest behaviour, minus what could hide or write things.

    Installed plugins stay enabled (unlike the ARCH-1B benchmark collector): a
    suite that needs pytest-asyncio or a plugin fixture must not read as
    failing. ``addopts``/``PYTEST_ADDOPTS`` are dropped so options such as
    ``-x``, ``-k`` or ``--lf`` cannot keep a required test from running, the
    cache is disabled (:func:`_pytest_command`) and no bytecode is written.
    ``scratch`` goes first on ``PYTHONPATH`` so ``-p`` finds the plugin copy.
    """
    env = os.environ.copy()
    for key in (OUTPUT_ENV, SELECT_ENV, COLLECT_ENV, COLLECT_ERRORS_ENV, "PYTEST_ADDOPTS"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(scratch) + (os.pathsep + existing if existing else "")
    env.update(extra)
    return env


def _read_bounded_json(path: Path) -> Any:
    with path.open("rb") as handle:
        encoded = handle.read(_MAX_REPORT_BYTES + 1)
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("pytest report is too large")
    return json.loads(encoded)


@dataclass(frozen=True, slots=True)
class TestsObservation:
    """Per-nodeid result of one pytest session: facts and omissions."""

    __test__ = False

    passed: Mapping[str, bool]
    unobserved: Mapping[str, str]


def observe_tests(
    workspace: Path, nodeids: Sequence[str], timeout: float, python: str | None = None
) -> TestsObservation:
    """Run the required nodeids in one pytest session and report each one.

    ``python`` is the interpreter to run pytest with (default: this one).

    Test *files* are passed on the command line and the private plugin keeps
    only the required nodeids. A nodeid whose file is missing, or that pytest
    did not run (deleted or renamed), is a ``False`` fact. A nodeid below a
    collector that failed (a module or package that cannot be imported) is
    unobserved: the import error may come from the environment -- a project
    dependency missing from this interpreter -- rather than from the agent.
    """
    by_file: dict[str, list[str]] = {}
    for nodeid in nodeids:
        by_file.setdefault(nodeid.split("::", 1)[0], []).append(nodeid)
    passed: dict[str, bool] = {}
    unobserved: dict[str, str] = {}
    files: list[str] = []
    runnable: list[str] = []
    for relative, file_nodeids in sorted(by_file.items()):
        try:
            present = _regular_file(workspace, relative) is not None
        except Unobservable as error:
            unobserved.update(dict.fromkeys(file_nodeids, error.reason))
            continue
        if present:
            files.append(relative)
            runnable.extend(file_nodeids)
        else:
            passed.update(dict.fromkeys(file_nodeids, False))
    if files:
        result = _pytest_outcomes(workspace, files, runnable, timeout, python or sys.executable)
        if isinstance(result, str):
            unobserved.update(dict.fromkeys(runnable, result))
        else:
            outcomes, failed_collectors = result
            for nodeid in runnable:
                if nodeid not in outcomes and _below_any(nodeid, failed_collectors):
                    unobserved[nodeid] = REASON_TEST_UNCOLLECTABLE
                else:
                    passed[nodeid] = outcomes.get(nodeid) == "passed"
    return TestsObservation(passed=passed, unobserved=unobserved)


def _below_any(nodeid: str, collectors: Sequence[str]) -> bool:
    return any(
        not collector or nodeid.startswith((collector + "::", collector + "/"))
        for collector in collectors
    )


def _pytest_outcomes(
    workspace: Path, files: Sequence[str], nodeids: Sequence[str], timeout: float, python: str
) -> tuple[dict[str, str], tuple[str, ...]] | str:
    """Outcomes of the selected nodeids that ran and the failed collectors, or a reason."""
    try:
        with _pytest_scratch("pathlayer-contract-pytest-") as (scratch, plugin):
            report = scratch / "outcomes.json"
            errors_report = scratch / "collect-errors.json"
            selection = scratch / "select.json"
            selection.write_text(json.dumps(sorted(nodeids), ensure_ascii=False), "utf-8")
            env = {
                OUTPUT_ENV: str(report),
                SELECT_ENV: str(selection),
                COLLECT_ERRORS_ENV: str(errors_report),
            }
            command = _pytest_command(
                python, plugin, "--continue-on-collection-errors", "--", *files
            )
            try:
                status = _launch(
                    command, cwd=workspace, env=_pytest_env(scratch, env), timeout=timeout
                )
            except OSError:
                return REASON_TEST_INTERPRETER
            if status is None:
                return REASON_PYTEST_TIMEOUT
            if (
                status not in _PYTEST_FACT_STATUSES
                or not report.is_file()
                or not errors_report.is_file()
            ):
                return REASON_PYTEST_UNREADABLE
            raw = _read_bounded_json(report)
            failed_collectors = _read_bounded_json(errors_report)
    except (OSError, ValueError, RecursionError, subprocess.SubprocessError):
        return REASON_PYTEST_UNREADABLE
    if not isinstance(failed_collectors, list) or not all(
        isinstance(collector, str) for collector in failed_collectors
    ):
        return REASON_PYTEST_UNREADABLE
    if not isinstance(raw, Mapping) or raw.keys() != {"exit_code", "outcomes"}:
        return REASON_PYTEST_UNREADABLE
    outcomes = raw["outcomes"]
    if raw["exit_code"] != status or not isinstance(outcomes, Mapping):
        return REASON_PYTEST_UNREADABLE
    selected = set(nodeids)
    if any(
        nodeid not in selected or result not in ("passed", "failed")
        for nodeid, result in outcomes.items()
    ):
        return REASON_PYTEST_UNREADABLE
    return dict(outcomes), tuple(failed_collectors)


def collect_pytest_nodeids(
    workspace: Path, target: str, timeout: float, python: str | None = None
) -> tuple[str, ...]:
    """Expand a workspace-relative file or directory to its collected nodeids.

    Compile-time only. Any collection error fails the expansion: a partial
    selection would silently shrink the contract.
    """
    try:
        with _pytest_scratch("pathlayer-contract-collect-") as (scratch, plugin):
            report = scratch / "collected.json"
            status = _launch(
                _pytest_command(python or sys.executable, plugin, "--collect-only", "--", target),
                cwd=workspace,
                env=_pytest_env(scratch, {COLLECT_ENV: str(report)}),
                timeout=timeout,
            )
            if status is None:
                raise PytestCollectionError(f"could not collect tests under {target!r}: timed out")
            if status == 5:
                raise PytestCollectionError(f"selection {target!r} contains no test")
            if status != 0 or not report.is_file():
                raise PytestCollectionError(
                    f"could not collect tests under {target!r} (pytest exited with status "
                    f"{status}); fix the collection error or list exact nodeids instead"
                )
            raw = _read_bounded_json(report)
    except (OSError, ValueError, RecursionError, subprocess.SubprocessError) as error:
        raise PytestCollectionError(f"could not collect tests under {target!r}") from error
    if raw is None:
        raise PytestCollectionError(
            f"selection {target!r} is too large (more than {MAX_OUTCOMES} tests) or unusable"
        )
    if not isinstance(raw, list) or not all(isinstance(nodeid, str) for nodeid in raw):
        raise PytestCollectionError(f"could not collect tests under {target!r}")
    if not raw:
        raise PytestCollectionError(f"selection {target!r} contains no test")
    return tuple(sorted(set(raw)))


_VERIFIED_INTERPRETERS: set[str] = set()


def check_test_interpreter(python: str, timeout: float) -> None:
    """Raise :class:`InterpreterCheckError` unless ``python`` can run the probe's pytest.

    Compile-time only: a Python at least 3.8 that loads the private plugin and
    collects an empty directory, exactly as a probe session would. Successful
    checks are remembered for the life of the process.
    """
    if python in _VERIFIED_INTERPRETERS:
        return
    version_check = f"import sys; sys.exit(0 if sys.version_info >= {_MINIMUM_PYTHON!r} else 3)"
    minimum = ".".join(map(str, _MINIMUM_PYTHON))
    try:
        with _pytest_scratch("pathlayer-contract-check-") as (scratch, plugin):
            try:
                status = _launch(
                    [python, "-c", version_check], cwd=scratch, env=None, timeout=timeout
                )
            except OSError as error:
                raise InterpreterCheckError(f"{python} could not be launched") from error
            if status is None:
                raise InterpreterCheckError(f"{python} did not start within {timeout:g}s")
            if status == 3:
                raise InterpreterCheckError(f"{python} is older than Python {minimum}")
            if status != 0:
                raise InterpreterCheckError(f"{python} cannot run Python code")
            empty = scratch / "empty"
            empty.mkdir()
            ini = scratch / "pytest.ini"
            ini.write_text("[pytest]\n", encoding="utf-8")
            collected = scratch / "collected.json"
            status = _launch(
                _pytest_command(python, plugin, "-c", str(ini), "--collect-only", "--", str(empty)),
                cwd=empty,
                env=_pytest_env(scratch, {COLLECT_ENV: str(collected)}),
                timeout=timeout,
            )
            healthy = status == 5 and collected.is_file() and _read_bounded_json(collected) == []
    except InterpreterCheckError:
        raise
    except (OSError, ValueError, RecursionError, subprocess.SubprocessError) as error:
        raise InterpreterCheckError(f"{python} cannot run pytest") from error
    if not healthy:
        raise InterpreterCheckError(
            f"{python} cannot run pytest (install pytest in that environment)"
        )
    _VERIFIED_INTERPRETERS.add(python)


def _command_succeeds(workspace: Path, argv: Sequence[str], timeout: float) -> bool:
    try:
        status = _launch(argv, cwd=workspace, env=None, timeout=timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise Unobservable(REASON_COMMAND_LAUNCH) from error
    if status is None:
        raise Unobservable(REASON_COMMAND_TIMEOUT)
    return status == 0


# --------------------------------------------------------------------------- #
# Compiled payload (strict: it is read back from an operator-owned file)
# --------------------------------------------------------------------------- #


class _InvalidPayload(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _FileCheck:
    checkpoint: str
    kind: str
    path: str
    baseline: str | None = None
    min_bytes: int = 0
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _CommandCheck:
    checkpoint: str
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Compiled:
    workspace: Path
    timeout: float
    python: str | None
    files: tuple[_FileCheck, ...]
    tests: tuple[tuple[str, str], ...]
    commands: tuple[_CommandCheck, ...]


_FILE_KEYS = {
    "file_exists": {"checkpoint", "kind", "path"},
    "file_changes": {"checkpoint", "kind", "path", "baseline"},
    "artifact_produced": {"checkpoint", "kind", "path", "min_bytes", "sha256"},
}


def _checkpoint_id(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise _InvalidPayload
    return value


def _parse_file_check(raw: object) -> _FileCheck:
    if not isinstance(raw, Mapping) or raw.get("kind") not in _FILE_KEYS:
        raise _InvalidPayload
    kind = raw["kind"]
    if set(raw) != _FILE_KEYS[kind] or not is_safe_relative_path(raw["path"]):
        raise _InvalidPayload
    check = _FileCheck(checkpoint=_checkpoint_id(raw["checkpoint"]), kind=kind, path=raw["path"])
    if kind == "file_changes":
        baseline = raw["baseline"]
        if not isinstance(baseline, str) or not _FINGERPRINT.fullmatch(baseline):
            raise _InvalidPayload
        return _FileCheck(check.checkpoint, kind, check.path, baseline=baseline)
    if kind == "artifact_produced":
        min_bytes, digest = raw["min_bytes"], raw["sha256"]
        if isinstance(min_bytes, bool) or not isinstance(min_bytes, int) or min_bytes < 0:
            raise _InvalidPayload
        if digest is not None and (
            not isinstance(digest, str) or not _SHA256_DIGEST.fullmatch(digest)
        ):
            raise _InvalidPayload
        return _FileCheck(check.checkpoint, kind, check.path, min_bytes=min_bytes, sha256=digest)
    return check


def _parse_payload(raw: object) -> _Compiled:
    expected = {
        "format",
        "schema_version",
        "workspace",
        "python",
        "timeout_seconds",
        "files",
        "tests",
        "commands",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise _InvalidPayload
    version = raw["schema_version"]
    if raw["format"] != PAYLOAD_FORMAT or isinstance(version, bool) or version != PAYLOAD_VERSION:
        raise _InvalidPayload
    workspace, timeout, python = raw["workspace"], raw["timeout_seconds"], raw["python"]
    if not isinstance(workspace, str) or not workspace or "\x00" in workspace:
        raise _InvalidPayload
    if python is not None and (
        not isinstance(python, str) or "\x00" in python or not os.path.isabs(python)
    ):
        raise _InvalidPayload
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise _InvalidPayload
    if not math.isfinite(timeout) or timeout <= 0:
        raise _InvalidPayload
    files, tests, commands = raw["files"], raw["tests"], raw["commands"]
    if (
        not isinstance(files, list)
        or not isinstance(tests, Mapping)
        or not isinstance(commands, list)
    ):
        raise _InvalidPayload
    parsed_tests: list[tuple[str, str]] = []
    for nodeid, checkpoint in tests.items():
        if (
            not isinstance(nodeid, str)
            or len(nodeid) > MAX_NODEID_LENGTH
            or nodeid.startswith("-")
            or "::" not in nodeid
            or not is_safe_relative_path(nodeid.split("::", 1)[0])
        ):
            raise _InvalidPayload
        parsed_tests.append((nodeid, _checkpoint_id(checkpoint)))
    parsed_commands: list[_CommandCheck] = []
    for command in commands:
        if not isinstance(command, Mapping) or set(command) != {"checkpoint", "argv"}:
            raise _InvalidPayload
        argv = command["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(arg, str) and arg and "\x00" not in arg for arg in argv)
        ):
            raise _InvalidPayload
        parsed_commands.append(_CommandCheck(_checkpoint_id(command["checkpoint"]), tuple(argv)))
    compiled = _Compiled(
        workspace=Path(workspace),
        timeout=float(timeout),
        python=python,
        files=tuple(_parse_file_check(check) for check in files),
        tests=tuple(sorted(parsed_tests)),
        commands=tuple(parsed_commands),
    )
    checkpoints = [
        *(check.checkpoint for check in compiled.files),
        *(checkpoint for _, checkpoint in compiled.tests),
        *(check.checkpoint for check in compiled.commands),
    ]
    if (
        not checkpoints
        or len(checkpoints) > _MAX_CHECKS
        or len(set(checkpoints)) != len(checkpoints)
    ):
        raise _InvalidPayload
    return compiled


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _evaluate_file(workspace: Path, check: _FileCheck) -> bool:
    if check.kind == "file_changes":
        return file_fingerprint(workspace, check.path) != check.baseline
    metadata = _regular_file(workspace, check.path)
    if check.kind == "file_exists":
        return metadata is not None
    if metadata is None or metadata.st_size < check.min_bytes:
        return False
    if check.sha256 is None:
        return True
    return file_fingerprint(workspace, check.path) == f"sha256:{check.sha256}"


def _observe(
    compiled: _Compiled, clock: Callable[[], float]
) -> tuple[dict[str, bool], dict[str, str]]:
    observed: dict[str, bool] = {}
    unobserved: dict[str, str] = {}
    deadline = clock() + compiled.timeout
    try:
        workspace = compiled.workspace.resolve(strict=True)
        usable = workspace.is_dir()
    except (OSError, RuntimeError):
        usable = False
    if not usable:
        every = [
            *(check.checkpoint for check in compiled.files),
            *(checkpoint for _, checkpoint in compiled.tests),
            *(check.checkpoint for check in compiled.commands),
        ]
        return observed, dict.fromkeys(every, REASON_WORKSPACE)

    def remaining() -> float:
        return deadline - clock()

    for file_check in compiled.files:
        try:
            observed[file_check.checkpoint] = _evaluate_file(workspace, file_check)
        except Unobservable as error:
            unobserved[file_check.checkpoint] = error.reason
        except Exception:
            unobserved[file_check.checkpoint] = REASON_UNEXPECTED

    if compiled.tests:
        checkpoints = dict(compiled.tests)
        budget = remaining()
        if budget <= 0:
            unobserved.update(dict.fromkeys(checkpoints.values(), REASON_BUDGET))
        else:
            try:
                result = observe_tests(workspace, list(checkpoints), budget, compiled.python)
            except Exception:
                unobserved.update(dict.fromkeys(checkpoints.values(), REASON_UNEXPECTED))
            else:
                for nodeid, value in result.passed.items():
                    observed[checkpoints[nodeid]] = value
                for nodeid, reason in result.unobserved.items():
                    unobserved[checkpoints[nodeid]] = reason

    for command in compiled.commands:
        budget = remaining()
        if budget <= 0:
            unobserved[command.checkpoint] = REASON_BUDGET
            continue
        try:
            observed[command.checkpoint] = _command_succeeds(workspace, command.argv, budget)
        except Unobservable as error:
            unobserved[command.checkpoint] = error.reason
        except Exception:
            unobserved[command.checkpoint] = REASON_UNEXPECTED
    return observed, unobserved


def run_compiled_contract(
    payload: object,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Observe the workspace once; print the ANC-2 proof shape. Never raises.

    Exit status: ``0`` with ``{"checkpoints": ...}`` on stdout when at least
    one checkpoint was observed, ``3`` for a payload this version cannot read,
    ``4`` when nothing could be observed. Omitted checkpoints are listed on
    stderr as ``unobserved <checkpoint>: <reason>``.
    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        compiled = _parse_payload(payload)
    except _InvalidPayload:
        err.write("invalid compiled coding contract; recompile it with this Pathlayer version\n")
        return EXIT_INVALID_CONTRACT
    try:
        observed, unobserved = _observe(compiled, clock)
    except Exception:
        err.write(f"{REASON_UNEXPECTED}\n")
        return EXIT_NOTHING_OBSERVED
    for checkpoint, reason in sorted(unobserved.items()):
        err.write(f"unobserved {checkpoint}: {reason}\n")
    if not observed:
        err.write("no checkpoint could be observed; no proof this tick\n")
        return EXIT_NOTHING_OBSERVED
    json.dump({"checkpoints": dict(sorted(observed.items()))}, out, sort_keys=True)
    out.write("\n")
    return EXIT_OBSERVED


__all__ = [
    "EXIT_INVALID_CONTRACT",
    "EXIT_NOTHING_OBSERVED",
    "EXIT_OBSERVED",
    "EXIT_PATHLAYER_UNAVAILABLE",
    "MISSING_FINGERPRINT",
    "PAYLOAD_FORMAT",
    "PAYLOAD_VERSION",
    "InterpreterCheckError",
    "PytestCollectionError",
    "TestsObservation",
    "Unobservable",
    "check_test_interpreter",
    "collect_pytest_nodeids",
    "contained_path",
    "file_fingerprint",
    "is_safe_relative_path",
    "observe_tests",
    "run_compiled_contract",
]
