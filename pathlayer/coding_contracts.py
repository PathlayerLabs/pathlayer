"""Declarative coding progress contracts compiled to progress-proof artifacts.

ANC-1/2/3 made progress proofs deterministic, but an operator still had to
hand-write a ``ProofSpec`` manifest and an executable probe. This module gives
coding agents a small declarative vocabulary instead::

    task: fix-login
    objective: Make the login flow pass its test.
    progress:
      - type: test_passes
        test: tests/test_login.py::test_login
      - type: file_changes
        path: src/auth.py

It compiles such a contract -- from YAML/JSON or from the equivalent Python
objects -- into exactly the bundle ANC-3 already defines (``proof_spec.json``,
``probe.py``, ``README.md``), installed by the same writer. Nothing new decides:
the ANC-2 producer runs the probe, the ANC-1 ``ProofProgressAnchor`` folds the
proofs, ``NoProgressDetector`` and the policy act on them unchanged.

Every primitive compiles to one or more ``checkpoint_set`` checkpoints of equal
weight; no priority between them is inferred. Pathlayer keeps three levels
apart and never infers one from another:

- an **observed effect** (``file_changes``) says something changed. It can mark
  one step of progress, but never reaches the objective on its own, so a
  contract made only of effects is rejected: once every checkpoint is true the
  anchor is satisfied and ``no_progress`` goes silent for the rest of the run;
- **progress toward the objective** is the number of checkpoints currently true;
  ``NoProgressDetector`` only credits a new high, so a checkpoint that flips
  back and forth, is deleted or is skipped never manufactures progress;
- the **objective is reached** when every checkpoint is true at once.

Two failure philosophies, as in ANC-2: everything here runs at build time and
raises :class:`CodingContractError` (a :class:`~pathlayer.config.ConfigError`)
before a run starts -- unknown keys, unsafe paths, duplicate or too many
checkpoints, an uncollectable test selection, an unreadable ``file_changes``
baseline. The generated probe runs at capture time and is fail-soft (see
:mod:`pathlayer.coding_contract_probe`).

Trust boundary: the manifest and the probe (which carries the compiled
contract) must stay outside the observed agent's write boundary. The test
primitives execute the workspace's own tests and ``conftest.py`` files, so an
agent that can edit them can forge a passing test; the generated README says so.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, TypeAlias

from pathlayer._contract_pytest_plugin import MAX_NODEID_LENGTH
from pathlayer.coding_contract_probe import (
    PAYLOAD_FORMAT,
    PAYLOAD_VERSION,
    InterpreterCheckError,
    PytestCollectionError,
    Unobservable,
    check_test_interpreter,
    collect_pytest_nodeids,
    contained_path,
    file_fingerprint,
)
from pathlayer.config import ConfigError, _as_mapping, _parse_file, _Section
from pathlayer.detectors.proof import _MAX_CHECKPOINT_ID_LENGTH, _MAX_REQUIRED_CHECKPOINTS
from pathlayer.proof_scaffold import (
    _MANIFEST_NAME,
    _PROBE_NAME,
    ScaffoldBundle,
    ScaffoldError,
    ScaffoldPaths,
    _blockquote,
    _install_bundle,
    _manifest_json,
    _quoted,
    _validated_spec,
)

_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_TIMEOUT_SECONDS = 3600.0
# The ANC-2 producer's own timeout must outlast the probe's budget, or it kills
# the probe before the probe can stop its pytest/command process group.
_PRODUCER_MARGIN_SECONDS = 10.0
_MAX_PATH_LENGTH = 1024
_COMMAND_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


class CodingContractError(ConfigError):
    """Raised when a coding contract is malformed or cannot be compiled."""


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FileExists:
    """Objective: a regular file exists at ``path`` (workspace-relative)."""

    path: str


@dataclass(frozen=True, slots=True)
class FileChanges:
    """Observed effect: ``path`` differs from its state when the contract was compiled.

    Missing-then-created and deleted both count as a change. Restoring the
    exact baseline content makes the checkpoint false again. It never reaches
    the objective on its own.
    """

    path: str


@dataclass(frozen=True, slots=True)
class ArtifactProduced:
    """Objective: a regular file of at least ``min_bytes``, optionally with this SHA-256."""

    path: str
    min_bytes: int = 1
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class TestPasses:
    """Objective: the exact pytest nodeid ``test`` passes."""

    __test__ = False  # importing the vocabulary into a test module must not collect it

    test: str


@dataclass(frozen=True, slots=True)
class TestsImprove:
    """Progress toward the objective: one checkpoint per selected test.

    Give exactly one of ``tests`` (exact nodeids) or ``path`` (a workspace file
    or directory expanded to its collected tests when the contract is
    compiled). Tests added later by the agent are not part of the contract.
    """

    __test__ = False

    tests: tuple[str, ...] = ()
    path: str | None = None


@dataclass(frozen=True, slots=True)
class CommandSucceeds:
    """Objective: ``command`` (an argument list, no shell) exits with status 0.

    ``id`` names the checkpoint (``command:<id>``), so the checkpoint never
    embeds an absolute path or depends on how arguments were split.
    """

    id: str
    command: tuple[str, ...]


Primitive: TypeAlias = (
    FileExists | FileChanges | ArtifactProduced | TestPasses | TestsImprove | CommandSucceeds
)


@dataclass(frozen=True, slots=True)
class CodingContract:
    """A declarative coding contract. Validated by :func:`compile_coding_contract`.

    ``workspace`` is resolved against the current directory at compile time
    (a contract *file* resolves it against its own directory instead).
    ``python`` is the interpreter the test primitives run under -- typically
    the project's virtualenv, e.g. ``.venv/bin/python``, relative to the
    workspace or absolute; by default, the interpreter running the probe.
    ``timeout_seconds`` is the probe's total time budget per tool call.
    ``proof_id`` and ``scope_id`` default to ``task``.
    """

    task: str
    objective: str
    progress: Sequence[Primitive]
    workspace: str | Path = "."
    python: str | Path | None = None
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    proof_id: str | None = None
    scope_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompiledCodingContract:
    """The progress-proof bundle plus the compiled data its probe carries."""

    bundle: ScaffoldBundle
    payload: Mapping[str, Any]
    producer_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class InstalledCodingContract:
    """A compiled contract written to disk by :func:`write_coding_contract`."""

    paths: ScaffoldPaths
    compiled: CompiledCodingContract

    def progress_proof_config(self) -> dict[str, Any]:
        """The ``detectors.progress_proof`` section that enables this bundle.

        Absolute paths, and the interpreter running this code: the probe
        imports ``pathlayer``, so it must run where Pathlayer is installed.
        """
        return {
            "enabled": True,
            "manifest_path": str(self.paths.manifest.resolve()),
            "probe_command": [sys.executable, str(self.paths.probe.resolve())],
            "timeout_seconds": self.compiled.producer_timeout_seconds,
        }


# --------------------------------------------------------------------------- #
# Loading (YAML / JSON / mapping)
# --------------------------------------------------------------------------- #

_ABSENT = object()


@contextlib.contextmanager
def _as_contract_errors() -> Iterator[None]:
    """Report the config section readers' errors as contract errors."""
    try:
        yield
    except CodingContractError:
        raise
    except ConfigError as error:
        raise CodingContractError(str(error)) from error


def _required_str(section: _Section, key: str, where: str) -> str:
    value = section._pop(key, None)
    if value is None:
        raise CodingContractError(f"'{where}.{key}' is required")
    if not isinstance(value, str):
        raise CodingContractError(
            f"invalid value for '{where}.{key}': expected string, got {type(value).__name__}"
        )
    return value


def _optional_str(section: _Section, key: str, where: str) -> str | None:
    value = section._pop(key, None)
    if value is not None and not isinstance(value, str):
        raise CodingContractError(
            f"invalid value for '{where}.{key}': expected string, got {type(value).__name__}"
        )
    return value


def _string_list(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise CodingContractError(
            f"invalid value for '{where}': expected list of strings, got {type(value).__name__}"
        )
    return tuple(value)


def _parse_tests_improve(section: _Section, where: str) -> TestsImprove:
    tests = section._pop("tests", _ABSENT)
    path = section._pop("path", _ABSENT)
    if (tests is _ABSENT) == (path is _ABSENT):
        raise CodingContractError(f"'{where}' needs exactly one of 'tests' or 'path'")
    if tests is not _ABSENT:
        nodeids = _string_list(tests, f"{where}.tests")
        if not nodeids:
            raise CodingContractError(f"'{where}.tests' must list at least one nodeid")
        return TestsImprove(tests=nodeids)
    if not isinstance(path, str):
        raise CodingContractError(
            f"invalid value for '{where}.path': expected string, got {type(path).__name__}"
        )
    return TestsImprove(path=path)


def _parse_artifact(section: _Section, where: str) -> ArtifactProduced:
    path = _required_str(section, "path", where)
    min_bytes = section.int_("min_bytes", 1)
    digest = section._pop("sha256", None)
    if digest is not None and not isinstance(digest, str):
        raise CodingContractError(
            f"invalid value for '{where}.sha256': expected string, got {type(digest).__name__}"
        )
    return ArtifactProduced(path=path, min_bytes=min_bytes, sha256=digest)


def _parse_command(section: _Section, where: str) -> CommandSucceeds:
    command_id = _required_str(section, "id", where)
    command = section._pop("command", None)
    if command is None:
        raise CodingContractError(f"'{where}.command' is required")
    return CommandSucceeds(id=command_id, command=_string_list(command, f"{where}.command"))


_PARSERS: dict[str, Any] = {
    "file_exists": lambda section, where: FileExists(_required_str(section, "path", where)),
    "file_changes": lambda section, where: FileChanges(_required_str(section, "path", where)),
    "artifact_produced": _parse_artifact,
    "test_passes": lambda section, where: TestPasses(_required_str(section, "test", where)),
    "tests_improve": _parse_tests_improve,
    "command_succeeds": _parse_command,
}
# Checked before any field is read, so a typo ("pth") is reported as such
# rather than as a missing required field.
_KEYS: dict[str, frozenset[str]] = {
    "file_exists": frozenset({"path"}),
    "file_changes": frozenset({"path"}),
    "artifact_produced": frozenset({"path", "min_bytes", "sha256"}),
    "test_passes": frozenset({"test"}),
    "tests_improve": frozenset({"tests", "path"}),
    "command_succeeds": frozenset({"id", "command"}),
}


def _primitive_from_mapping(raw: object, where: str) -> Primitive:
    if not isinstance(raw, Mapping):
        raise CodingContractError(f"'{where}' must be a mapping with a 'type'")
    section = _Section(_as_mapping(raw, where), where)
    kind = section._pop("type", None)
    if kind is None:
        raise CodingContractError(f"'{where}.type' is required")
    if not isinstance(kind, str) or kind not in _PARSERS:
        expected = ", ".join(sorted(_PARSERS))
        raise CodingContractError(
            f"'{where}': unknown progress type {kind!r}; expected one of {expected}"
        )
    unknown = sorted(set(section._data) - _KEYS[kind])
    if unknown:
        allowed = ", ".join(sorted(_KEYS[kind]))
        raise CodingContractError(
            f"unknown key(s) in '{where}' ({kind}): {', '.join(map(repr, unknown))}; "
            f"allowed: {allowed}"
        )
    primitive: Primitive = _PARSERS[kind](section, where)
    section.finish()
    return primitive


def coding_contract_from_mapping(
    data: Mapping[str, Any], *, base_dir: str | Path
) -> CodingContract:
    """Parse an already-loaded contract mapping; unknown keys are rejected.

    A relative ``workspace`` is resolved against ``base_dir``; an absent one
    means ``base_dir`` itself.
    """
    if not isinstance(data, Mapping):
        raise CodingContractError(f"a coding contract must be a mapping, got {type(data).__name__}")
    with _as_contract_errors():
        root = _Section(_as_mapping(data, "contract"), "contract")
        task = _required_str(root, "task", "contract")
        objective = _required_str(root, "objective", "contract")
        workspace = root.str_("workspace", "")
        python = _optional_str(root, "python", "contract")
        timeout = root.float_("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
        proof_id = _optional_str(root, "proof_id", "contract")
        scope_id = _optional_str(root, "scope_id", "contract")
        raw_progress = root._pop("progress", None)
        root.finish()
        if not isinstance(raw_progress, list):
            raise CodingContractError("'contract.progress' must be a list of progress entries")
        progress = tuple(
            _primitive_from_mapping(item, f"progress[{index}]")
            for index, item in enumerate(raw_progress)
        )
    base = Path(base_dir)
    workspace_path = Path(workspace) if workspace else Path()
    return CodingContract(
        task=task,
        objective=objective,
        progress=progress,
        workspace=workspace_path if workspace_path.is_absolute() else base / workspace_path,
        python=python,
        timeout_seconds=timeout,
        proof_id=proof_id,
        scope_id=scope_id,
    )


def load_coding_contract(path: str | Path) -> CodingContract:
    """Load a ``.yaml``/``.yml``/``.json`` contract file (YAML needs ``pathlayer[yaml]``)."""
    contract_path = Path(path)
    if not contract_path.is_file():
        raise CodingContractError(f"coding contract not found: {contract_path}")
    try:
        data = _parse_file(contract_path)
    except ConfigError as error:
        raise CodingContractError(str(error)) from error
    except (OSError, UnicodeError) as error:
        raise CodingContractError(
            f"cannot read coding contract {contract_path}: {error}"
        ) from error
    return coding_contract_from_mapping(data, base_dir=contract_path.parent)


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #


def _nonempty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CodingContractError(f"'{name}' must be a non-empty string")
    return value


def _probe_budget(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= _MAX_TIMEOUT_SECONDS
    ):
        raise CodingContractError(
            f"'timeout_seconds' must be a finite number of seconds in (0, {_MAX_TIMEOUT_SECONDS:g}]"
        )
    return float(value)


def _resolve_workspace(value: object) -> Path:
    if not isinstance(value, (str, PurePath)):
        raise CodingContractError("'workspace' must be a path")
    try:
        workspace = Path(value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise CodingContractError(f"workspace {value!s} cannot be resolved: {error}") from error
    if not workspace.is_dir():
        raise CodingContractError(f"workspace must be an existing directory: {workspace}")
    return workspace


def _interpreter_path(raw: object, workspace: Path) -> str | None:
    """Absolute path of the contract's test interpreter, deliberately not resolved.

    A virtualenv's ``bin/python`` is usually a symlink to a base interpreter:
    invoking the link target instead would silently bypass the virtualenv.
    """
    if raw is None:
        return None
    text = str(raw) if isinstance(raw, (str, PurePath)) else ""
    if not text or "\x00" in text or len(text) > _MAX_PATH_LENGTH:
        raise CodingContractError("'python' must be the path of a Python interpreter")
    candidate = Path(text).expanduser()
    path = os.path.abspath(candidate if candidate.is_absolute() else workspace / candidate)
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise CodingContractError(f"'python' {text!r} is not an executable file ({path})")
    return path


def _workspace_path(raw: object, workspace: Path, where: str) -> str:
    """Normalize a contract path to a workspace-relative POSIX path, or raise."""
    if not isinstance(raw, str):
        raise CodingContractError(f"'{where}' must be a string, got {type(raw).__name__}")
    if not raw:
        raise CodingContractError(f"'{where}' must not be empty")
    if "\x00" in raw:
        raise CodingContractError(f"'{where}' contains a NUL byte")
    if len(raw) > _MAX_PATH_LENGTH:
        raise CodingContractError(f"'{where}' is longer than {_MAX_PATH_LENGTH} characters")
    pure = PurePath(raw)
    if pure.is_absolute() or pure.anchor or raw.startswith("/"):
        raise CodingContractError(f"'{where}' {raw!r} must be relative to the workspace")
    parts = [part for part in pure.parts if part not in ("", ".")]
    if ".." in parts:
        raise CodingContractError(
            f"'{where}' {raw!r} must not contain '..' (it could point outside the workspace)"
        )
    if not parts:
        raise CodingContractError(f"'{where}' {raw!r} must name something inside the workspace")
    relative = "/".join(parts)
    try:
        contained_path(workspace, relative)
    except Unobservable as error:
        raise CodingContractError(
            f"'{where}' {raw!r} resolves outside the workspace through a symlink"
        ) from error
    return relative


def _nodeid(raw: object, workspace: Path, where: str) -> str:
    if not isinstance(raw, str):
        raise CodingContractError(f"'{where}' must be a string, got {type(raw).__name__}")
    if "\x00" in raw:
        raise CodingContractError(f"'{where}' contains a NUL byte")
    if raw.startswith("-"):
        raise CodingContractError(f"'{where}' {raw!r} must not start with '-'")
    if len(raw) > MAX_NODEID_LENGTH:
        raise CodingContractError(f"'{where}' is longer than {MAX_NODEID_LENGTH} characters")
    file_part, separator, rest = raw.partition("::")
    if not separator or not rest:
        raise CodingContractError(
            f"'{where}' {raw!r} must be an exact pytest nodeid such as "
            "'tests/test_login.py::test_login'"
        )
    return f"{_workspace_path(file_part, workspace, where)}::{rest}"


def _checkpoint(prefix: str, value: str) -> str:
    """``prefix:value``, deterministically shortened to fit ANC-1's id cap."""
    candidate = f"{prefix}:{value}"
    if len(candidate) <= _MAX_CHECKPOINT_ID_LENGTH:
        return candidate
    digest = hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    keep = _MAX_CHECKPOINT_ID_LENGTH - len(prefix) - len(digest) - 2
    return f"{prefix}:{value[:keep]}#{digest}"


@dataclass(slots=True)
class _Plan:
    """Mutable accumulator for one compilation."""

    workspace: Path
    timeout: float
    python: str  # the interpreter the test primitives will run under
    files: list[dict[str, Any]]
    tests: dict[str, str]
    commands: list[dict[str, Any]]
    rows: list[tuple[str, str, str]]  # (checkpoint, what it checks, level)
    seen: set[str]
    interpreter_checked: bool = False

    def check_interpreter(self) -> None:
        """Fail the build now if the test interpreter cannot run a probe session."""
        if self.interpreter_checked:
            return
        try:
            check_test_interpreter(self.python, self.timeout)
        except InterpreterCheckError as error:
            raise CodingContractError(f"test primitives cannot run: {error}") from error
        self.interpreter_checked = True

    def add(self, checkpoint: str, what: str, level: str) -> None:
        if checkpoint in self.seen:
            raise CodingContractError(
                f"duplicate checkpoint {checkpoint!r}: two progress entries observe the same thing"
            )
        self.seen.add(checkpoint)
        self.rows.append((checkpoint, what, level))


def _compile_file(plan: _Plan, primitive: FileExists | FileChanges | ArtifactProduced) -> None:
    if isinstance(primitive, FileExists):
        relative = _workspace_path(primitive.path, plan.workspace, "file_exists.path")
        checkpoint = _checkpoint("file_exists", relative)
        plan.add(checkpoint, f"a regular file exists at `{relative}`", "objective")
        plan.files.append({"checkpoint": checkpoint, "kind": "file_exists", "path": relative})
        return
    if isinstance(primitive, FileChanges):
        relative = _workspace_path(primitive.path, plan.workspace, "file_changes.path")
        checkpoint = _checkpoint("file_changes", relative)
        try:
            baseline = file_fingerprint(plan.workspace, relative)
        except Unobservable as error:
            raise CodingContractError(
                f"cannot capture the file_changes baseline for {relative!r}: {error.reason}"
            ) from error
        plan.add(
            checkpoint,
            f"`{relative}` differs from its content when this contract was compiled",
            "observed effect",
        )
        plan.files.append(
            {
                "checkpoint": checkpoint,
                "kind": "file_changes",
                "path": relative,
                "baseline": baseline,
            }
        )
        return
    relative = _workspace_path(primitive.path, plan.workspace, "artifact_produced.path")
    min_bytes = primitive.min_bytes
    if isinstance(min_bytes, bool) or not isinstance(min_bytes, int) or min_bytes < 0:
        raise CodingContractError("'artifact_produced.min_bytes' must be a non-negative integer")
    digest = primitive.sha256
    if digest is not None and (not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
        raise CodingContractError("'artifact_produced.sha256' must be 64 hexadecimal characters")
    digest = None if digest is None else digest.lower()
    checkpoint = _checkpoint("artifact_produced", relative)
    what = f"a regular file of at least {min_bytes} byte(s) exists at `{relative}`"
    if digest is not None:
        what += f" with SHA-256 `{digest}`"
    plan.add(checkpoint, what, "objective")
    plan.files.append(
        {
            "checkpoint": checkpoint,
            "kind": "artifact_produced",
            "path": relative,
            "min_bytes": min_bytes,
            "sha256": digest,
        }
    )


def _add_test(plan: _Plan, nodeid: str, level: str) -> None:
    checkpoint = _checkpoint("test", nodeid)
    plan.add(checkpoint, f"pytest nodeid `{nodeid}` passes", level)
    plan.tests[nodeid] = checkpoint


def _compile_tests(plan: _Plan, primitive: TestPasses | TestsImprove) -> None:
    if isinstance(primitive, TestPasses):
        _add_test(plan, _nodeid(primitive.test, plan.workspace, "test_passes.test"), "objective")
        return
    tests, path = primitive.tests, primitive.path
    if isinstance(tests, (str, bytes)) or not isinstance(tests, Sequence):
        raise CodingContractError("'tests_improve.tests' must be a list of exact pytest nodeids")
    if bool(tests) == (path is not None):
        raise CodingContractError("'tests_improve' needs exactly one of 'tests' or 'path'")
    level = "progress toward the objective"
    if path is None:
        nodeids = [_nodeid(test, plan.workspace, "tests_improve.tests") for test in tests]
        duplicates = sorted({nodeid for nodeid in nodeids if nodeids.count(nodeid) > 1})
        if duplicates:
            raise CodingContractError(
                f"'tests_improve.tests' lists duplicate nodeids: {duplicates}"
            )
        for nodeid in nodeids:
            _add_test(plan, nodeid, level)
        return
    relative = _workspace_path(path, plan.workspace, "tests_improve.path")
    if not plan.workspace.joinpath(*relative.split("/")).exists():
        raise CodingContractError(f"'tests_improve.path' {relative!r} does not exist")
    plan.check_interpreter()
    try:
        collected = collect_pytest_nodeids(plan.workspace, relative, plan.timeout, plan.python)
    except PytestCollectionError as error:
        raise CodingContractError(f"'tests_improve.path': {error}") from error
    for nodeid in collected:
        _add_test(plan, _nodeid(nodeid, plan.workspace, "tests_improve.path"), level)


def _compile_command(plan: _Plan, primitive: CommandSucceeds) -> None:
    command_id = primitive.id
    if not isinstance(command_id, str) or not _COMMAND_ID.fullmatch(command_id):
        raise CodingContractError(
            "'command_succeeds.id' must be 1-64 characters from [A-Za-z0-9._-], "
            "starting with a letter or a digit"
        )
    argv = primitive.command
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise CodingContractError(
            "'command_succeeds.command' must be a non-empty list of arguments (no shell)"
        )
    for argument in argv:
        if not isinstance(argument, str) or not argument:
            raise CodingContractError(
                "'command_succeeds.command' arguments must be non-empty strings"
            )
        if "\x00" in argument:
            raise CodingContractError("'command_succeeds.command' contains a NUL byte")
    checkpoint = _checkpoint("command", command_id)
    plan.add(checkpoint, f"command `{command_id}` exits with status 0", "objective")
    plan.commands.append({"checkpoint": checkpoint, "argv": list(argv)})


def compile_coding_contract(contract: CodingContract) -> CompiledCodingContract:
    """Validate a contract and render its progress-proof bundle. Build time: raises on any problem.

    Reads the workspace to capture ``file_changes`` baselines, checks that the
    test interpreter can run a probe session, and runs pytest collection to
    expand ``tests_improve`` path selections; writes nothing.
    """
    task = _nonempty_str(contract.task, "task")
    objective = _nonempty_str(contract.objective, "objective")
    proof_id = task if contract.proof_id is None else _nonempty_str(contract.proof_id, "proof_id")
    scope_id = task if contract.scope_id is None else _nonempty_str(contract.scope_id, "scope_id")
    timeout = _probe_budget(contract.timeout_seconds)
    progress = contract.progress
    if isinstance(progress, (str, bytes)) or not isinstance(progress, Sequence):
        raise CodingContractError("'progress' must be a sequence of progress primitives")
    if not progress:
        raise CodingContractError("a coding contract needs at least one progress entry")
    workspace = _resolve_workspace(contract.workspace)
    python = _interpreter_path(contract.python, workspace)
    if python is not None and not any(isinstance(p, (TestPasses, TestsImprove)) for p in progress):
        raise CodingContractError(
            "'python' only selects the interpreter of test primitives, and this contract has none"
        )
    plan = _Plan(
        workspace,
        timeout,
        python=python or sys.executable,
        files=[],
        tests={},
        commands=[],
        rows=[],
        seen=set(),
    )
    for primitive in progress:
        if isinstance(primitive, (FileExists, FileChanges, ArtifactProduced)):
            _compile_file(plan, primitive)
        elif isinstance(primitive, (TestPasses, TestsImprove)):
            _compile_tests(plan, primitive)
        elif isinstance(primitive, CommandSucceeds):
            _compile_command(plan, primitive)
        else:
            raise CodingContractError(f"unsupported progress primitive: {type(primitive).__name__}")
    if all(isinstance(primitive, FileChanges) for primitive in progress):
        raise CodingContractError(
            "a coding contract needs at least one objective primitive (file_exists, "
            "artifact_produced, test_passes, tests_improve or command_succeeds): "
            "file_changes only records an observed effect, and a contract made only of "
            "effects would silence no-progress detection as soon as they are observed"
        )
    if plan.tests:
        plan.check_interpreter()
    if len(plan.seen) > _MAX_REQUIRED_CHECKPOINTS:
        raise CodingContractError(
            f"the contract compiles to {len(plan.seen)} checkpoints; a proof spec allows at most "
            f"{_MAX_REQUIRED_CHECKPOINTS}, so narrow the test selection"
        )
    try:
        spec = _validated_spec(
            task_id=task, proof_id=proof_id, scope_id=scope_id, checkpoints=sorted(plan.seen)
        )
    except ScaffoldError as error:
        raise CodingContractError(f"{error} (check the task, proof_id and scope_id)") from error
    payload: dict[str, Any] = {
        "format": PAYLOAD_FORMAT,
        "schema_version": PAYLOAD_VERSION,
        "workspace": str(workspace),
        "python": python,
        "timeout_seconds": timeout,
        "files": sorted(plan.files, key=lambda check: str(check["checkpoint"])),
        "tests": dict(sorted(plan.tests.items())),
        "commands": sorted(plan.commands, key=lambda check: str(check["checkpoint"])),
    }
    producer_timeout = timeout + _PRODUCER_MARGIN_SECONDS
    bundle = ScaffoldBundle(
        proof_spec=spec,
        manifest_json=_manifest_json(spec),
        probe_python=_render_probe(task=task, objective=objective, payload=payload),
        readme_markdown=_render_readme(
            task=task,
            objective=objective,
            rows=sorted(plan.rows),
            python=python,
            timeout=timeout,
            producer_timeout=producer_timeout,
        ),
    )
    return CompiledCodingContract(
        bundle=bundle, payload=payload, producer_timeout_seconds=producer_timeout
    )


def write_coding_contract(
    contract: CodingContract, *, output_dir: str | Path
) -> InstalledCodingContract:
    """Compile, then write the three bundle files, refusing to overwrite any.

    Compilation errors raise :class:`CodingContractError` before anything is
    written; write errors raise :class:`~pathlayer.proof_scaffold.ScaffoldError`
    from the ANC-3 writer, which removes whatever it installed.
    """
    compiled = compile_coding_contract(contract)
    return InstalledCodingContract(
        paths=_install_bundle(output_dir, compiled.bundle), compiled=compiled
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_probe(*, task: str, objective: str, payload: Mapping[str, Any]) -> str:
    # Double encoding: a JSON string literal is also a valid Python string
    # literal, and ensure_ascii keeps the source free of raw control characters
    # and lone surrogates whatever the task, objective, paths or nodeids.
    literal = json.dumps(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False), ensure_ascii=True
    )
    return f'''#!/usr/bin/env python3
"""Pathlayer coding-contract probe for {_quoted(task)} (generated; do not edit).

Objective (descriptive only): {_quoted(objective)}

This file only carries the compiled contract as data. The checks run in the
installed ``pathlayer.coding_contract_probe`` module, so run it with the Python
interpreter Pathlayer is installed in. Keep this file and the manifest beside
it outside the observed agent's write boundary.
"""

from __future__ import annotations

import json
import sys

COMPILED_CONTRACT = json.loads({literal})


def main() -> int:
    try:
        from pathlayer.coding_contract_probe import run_compiled_contract
    except ImportError as error:
        sys.stderr.write(
            "pathlayer is not importable by " + sys.executable
            + " (" + type(error).__name__ + "); no proof this tick\\n"
        )
        return 2
    return run_compiled_contract(COMPILED_CONTRACT)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _code(text: str) -> str:
    fence = "``" if "`" in text else "`"
    padding = " " if "`" in text else ""
    return f"{fence}{padding}{text}{padding}{fence}"


def _render_readme(
    *,
    task: str,
    objective: str,
    rows: Sequence[tuple[str, str, str]],
    python: str | None,
    timeout: float,
    producer_timeout: float,
) -> str:
    table = "\n".join(
        f"| {_table_cell(_code(checkpoint))} | {_table_cell(what)} | {level} |"
        for checkpoint, what, level in rows
    )
    interpreter = (
        _code(python)
        if python is not None
        else "the interpreter running this probe (set the contract's `python` key to use "
        "the project's own)"
    )
    config = json.dumps(
        {
            "detectors": {
                "progress_proof": {
                    "enabled": True,
                    "manifest_path": f"./{_MANIFEST_NAME}",
                    "probe_command": [sys.executable, f"./{_PROBE_NAME}"],
                    "timeout_seconds": producer_timeout,
                }
            }
        },
        indent=2,
    )
    return f"""# Pathlayer coding progress contract

Generated by `pathlayer-compile-contract` from a declarative coding contract.
This is an ordinary progress-proof bundle: `{_MANIFEST_NAME}` is the
validated proof manifest and `{_PROBE_NAME}` is the progress probe.

## Task

{_blockquote(task)}

## Objective

{_blockquote(objective)}

The task and objective are descriptive only. They never determine progress.

## Checkpoints

| Checkpoint | True when | Level |
| --- | --- | --- |
{table}

## How Pathlayer reads this contract

Pathlayer keeps three levels apart and never infers one from another. Every
checkpoint has the same weight; no priority between them is inferred.

- **Observed effect** (`file_changes`): something changed in the workspace. It
  can mark one step of progress, but it never reaches the objective on its own:
  a contract made only of effects is rejected when it is compiled.
- **Progress toward the objective**: the number of checkpoints currently true.
  Each new high restarts `no_progress`'s patience. A checkpoint that becomes
  false again, or flips back and forth, never counts as new progress.
- **Objective reached**: every checkpoint is true at the same time. The
  `no_progress` detector then stays silent: an agent that has finished is
  never stopped for having nothing left to do.

## When a check cannot observe its resource

| Situation | Reported |
| --- | --- |
| test failed, was skipped, deleted or renamed | `false` |
| command exited with a non-zero status | `false` |
| file missing (for `file_changes`: compared with the compile-time baseline) | `false` |
| test module cannot be imported (agent error, or dependency missing here) | omitted |
| path is a directory or symlink, resolves outside the workspace, or is unreadable | omitted |
| pytest or the command could not start, timed out, or reported unreadable results | omitted |
| the probe's time budget ran out | omitted |

An omitted checkpoint can neither create nor erase progress: its last known
value is kept for later, and the tool call itself is a *gap*: `no_progress`
stays silent on it and restarts its patience afterwards. When nothing can be
observed the probe exits with a non-zero status and Pathlayer attaches only a
`progress_proof_unobserved` marker for that tool call. Reasons are printed on
the probe's standard error: run `python {_PROBE_NAME}` to see them. Pathlayer
logs a warning when a probe produces no proof or omits a required checkpoint.

All test checkpoints share one pytest session per tool call, run with
{interpreter}, the plugins installed there and the project's ini options --
except `addopts`, ignored so that options such as `-x` cannot hide a required
test. The project's test dependencies must be importable by that interpreter;
Pathlayer itself need not be installed there.

Known limit: a probe that keeps failing never causes a STOP on its own, even
when Pathlayer's other progress signals are flat. An agent that stays stuck
while its probe keeps failing is left to the other detectors.

## Trust boundary

Keep `{_MANIFEST_NAME}` and `{_PROBE_NAME}` outside the observed agent's write
boundary: an agent that can edit its own probe can forge success or
stagnation. The checks themselves run in the installed `pathlayer` package.

Test checkpoints run pytest inside the workspace, with the workspace on the
import path, so pytest imports the workspace's `conftest.py` files, test
modules and code under test. Any of those an agent can edit -- or a module it
creates to shadow one pytest imports -- can forge a passing test: keep the test
code outside its write boundary when test checkpoints must be trusted. The same
holds for a test interpreter inside the workspace, such as `.venv`.
`command_succeeds` runs the configured command in the workspace, without a
shell.

## Cost

Pathlayer runs this probe synchronously after every tool call, with a
{timeout:g}-second budget. Keep test selections small.

## Configure Pathlayer

Run Pathlayer from this directory or replace the relative paths
(`pathlayer-compile-contract` prints the same section with absolute paths).
The interpreter must be the one Pathlayer is installed in, because the probe
imports `pathlayer`:

```json
{config}
```

The OpenClaw recorder runs the probe automatically. With LangGraph, pass the
producer to the adapter:

```python
from pathlayer.adapters.langgraph import LangGraphAdapter
from pathlayer.config import build_pipeline, load_config
from pathlayer.detectors.proof_producer import producer_from_config

config = load_config()
pipeline = build_pipeline(config)
adapter = LangGraphAdapter(pipeline.sink, progress_proof=producer_from_config(config))
```
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_current_state(installed: InstalledCodingContract) -> None:
    """Run the installed probe once, exactly as the ANC-2 producer would."""
    command = installed.progress_proof_config()["probe_command"]
    required = installed.compiled.bundle.proof_spec.required
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=installed.compiled.producer_timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        print(
            "\nwarning: the generated probe could not be run; "
            "Pathlayer would measure nothing on a tool call"
        )
        return
    observed: dict[str, bool] = {}
    if completed.returncode == 0:
        with contextlib.suppress(ValueError, TypeError, KeyError):
            observed = dict(json.loads(completed.stdout)["checkpoints"])
    reasons: dict[str, str] = {}
    for line in completed.stderr.splitlines():
        for checkpoint in required:
            prefix = f"unobserved {checkpoint}: "
            if line.startswith(prefix):
                reasons[checkpoint] = line[len(prefix) :]
    print("\nCurrent state (the generated probe was run once):")
    for checkpoint in required:
        if checkpoint in observed:
            print(f"  {checkpoint}: {'true' if observed[checkpoint] else 'false'}")
        else:
            print(f"  {checkpoint}: unobserved ({reasons.get(checkpoint, 'no result')})")
    if completed.returncode != 0:
        print(
            f"warning: the probe exited with status {completed.returncode}; "
            "Pathlayer would measure nothing on such a tool call"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pathlayer-compile-contract",
        description=(
            "Compile a declarative coding contract into a Pathlayer progress-proof bundle."
        ),
    )
    parser.add_argument("contract", help="contract file (.yaml, .yml or .json)")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new bundle directory, outside the observed agent's write boundary",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="do not run the generated probe once after writing it",
    )
    args = parser.parse_args(argv)
    try:
        installed = write_coding_contract(
            load_coding_contract(args.contract), output_dir=args.output_dir
        )
    except (ConfigError, ScaffoldError) as error:
        parser.error(str(error))
    for path in (installed.paths.manifest, installed.paths.probe, installed.paths.readme):
        print(f"created {path}")
    print("\nEnable it in your Pathlayer config (JSON is also valid YAML):")
    print(
        json.dumps({"detectors": {"progress_proof": installed.progress_proof_config()}}, indent=2)
    )
    if not args.no_check:
        _print_current_state(installed)
    return 0


__all__ = [
    "ArtifactProduced",
    "CodingContract",
    "CodingContractError",
    "CommandSucceeds",
    "CompiledCodingContract",
    "FileChanges",
    "FileExists",
    "InstalledCodingContract",
    "Primitive",
    "TestPasses",
    "TestsImprove",
    "coding_contract_from_mapping",
    "compile_coding_contract",
    "load_coding_contract",
    "write_coding_contract",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
