"""Git workspace and command ``StateEvidence`` collectors (STATE-1).

ARCH-1B collects one file and one pytest selection. A coding preset needs two
more deterministic observations, kept here rather than in
:mod:`pathlayer.state_collectors` because that module's bytes are one of the
sealed inputs of the COMP-1 benchmark: growing it for
an unrelated feature churns that manifest for no benefit, while a new module
costs nothing:

- **git workspace**: the content of the whole working tree, as the tree object
  ``git`` itself would write for it. This is content-addressed, so it is
  independent of ``HEAD``, of the index and of how a change was produced: the
  agent committing, staging or amending does not move the fingerprint, while
  editing a file does. Ignored files are excluded (``git`` decides, not us).
- **command**: whether an explicitly configured validation command succeeds.
  Only the success/failure of the exit status is canonical -- never its output,
  which is rarely reproducible.

Both are collectors, so they run at the adapter's tool boundary and do I/O; the
detection loop only ever reads the recorded fingerprints.

Two limits an operator must know. The configured command runs with the
workspace as its working directory, which the observed agent can write to: name
it by absolute path, or keep the executable outside the agent's reach -- an
agent that can plant a binary the command resolves to can make Pathlayer run
it, exactly as ANC-4 says of its probe. And the git probe's scratch object
directory accumulates one blob per distinct file content the run produces; it
lives in the system temporary directory and is removed when the probe is
collected or the process exits, but a very long run editing large files pays
for it in temporary space meanwhile.

The git probe never writes to the observed repository. It keeps its own index
and object directory in a run-scoped temporary directory, with the repository's
object store as a read-only alternate, and reads with ``--no-optional-locks``.
That matters beyond tidiness: the observed agent's own ``git`` commands must not
see a lock, a staged change or a refreshed index that Pathlayer caused.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pathlayer.detectors.state_evidence import StateEvidence, validate_state_evidence

GIT_COLLECTOR_VERSION = "git-collector@1"
COMMAND_COLLECTOR_VERSION = "command-collector@1"

Phase = Literal["before", "after"]

_TREE_PREFIX = "tree:"
_SUCCESS = "success"
_FAILURE = "failure"
# A delta is a diagnostic, not evidence: bound the path list so a wide refactor
# cannot push an unbounded payload through the event metadata and the store.
_MAX_DELTA_PATHS = 64
_MAX_COMMAND_LENGTH = 64
_MAX_COMMAND_ARGUMENT_LENGTH = 4096
_GIT_TIMEOUT_SECONDS = 120.0
_SHA1_LENGTH = 40
_SHA256_LENGTH = 64  # git repositories can use either object format


class GitWorkspaceError(RuntimeError):
    """Raised at build time when a workspace cannot be observed through git."""


def _run_git(
    arguments: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None
) -> str | None:
    """Run one git command, returning its stdout, or ``None`` on any failure."""
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", *arguments],
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def git_directory(workspace: Path) -> str:
    """The resolved ``.git`` directory of ``workspace``.

    Raises :class:`GitWorkspaceError` when the workspace is not a git working
    tree, so a preset fails at build time instead of collecting nothing for a
    whole run.
    """
    output = _run_git(["rev-parse", "--absolute-git-dir"], cwd=workspace)
    if output is None or not output.strip():
        raise GitWorkspaceError(
            f"{workspace} is not a git repository; set 'git: false' to observe files only"
        )
    return output.strip()


@dataclass(slots=True)
class GitWorkspaceProbe:
    """Fingerprints the working tree without touching the observed repository.

    One probe per run: its scratch index keeps git's stat cache warm, so a
    capture rehashes only the files that actually changed, and its scratch
    object directory holds the blobs and trees written while fingerprinting --
    which is also what lets a later capture name the paths that changed.
    """

    workspace: Path
    _scratch: tempfile.TemporaryDirectory[str] | None = field(default=None, init=False)
    _env: dict[str, str] | None = field(default=None, init=False)
    # The scratch index is real mutable state, unlike every other sensor. An
    # adapter serializes its own captures, but one producer can legitimately be
    # handed to two adapters; two `git add` runs against the same index file
    # would corrupt it. This keeps that impossible without the caller knowing.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _environment(self) -> dict[str, str] | None:
        if self._env is not None:
            return self._env
        try:
            git_dir = Path(git_directory(self.workspace))
        except GitWorkspaceError:
            return None
        scratch = tempfile.TemporaryDirectory(prefix="pathlayer-git-state-")
        objects = Path(scratch.name) / "objects"
        index = Path(scratch.name) / "index"
        try:
            objects.mkdir(parents=True, exist_ok=True)
            # Start from the repository's own index so git can reuse its stat
            # cache; a fresh index would rehash every tracked file per capture.
            # Its absence (a repository with no commit yet) is normal.
            source_index = git_dir / "index"
            if source_index.is_file():
                shutil.copyfile(source_index, index)
        except OSError:
            scratch.cleanup()
            return None
        alternates = [str(git_dir / "objects"), *_repository_alternates(git_dir)]
        self._scratch = scratch
        self._env = {
            **os.environ,
            "GIT_INDEX_FILE": str(index),
            "GIT_OBJECT_DIRECTORY": str(objects),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": os.pathsep.join(alternates),
            "GIT_OPTIONAL_LOCKS": "0",
        }
        return self._env

    def tree(self) -> str | None:
        """The tree object id of the current working tree, or ``None`` if unusable."""
        with self._lock:
            env = self._environment()
            if env is None:
                return None
            # ``add -A`` applies the repository's own ignore rules and handles
            # deletions, symlinks and submodules with git's semantics rather
            # than a reimplementation of them.
            if _run_git(["add", "-A", "--"], cwd=self.workspace, env=env) is None:
                return None
            output = _run_git(["write-tree"], cwd=self.workspace, env=env)
        if output is None:
            return None
        tree = output.strip()
        return tree if _is_object_id(tree) else None

    def changed_paths(self, before: str, after: str) -> tuple[tuple[str, ...], bool] | None:
        """Paths differing between two observed trees, and whether the list was cut."""
        with self._lock:
            env = self._environment()
            if env is None:
                return None
            output = _run_git(
                ["diff-tree", "-r", "--name-only", "-z", "--no-renames", before, after],
                cwd=self.workspace,
                env=env,
            )
        if output is None:
            return None
        paths = tuple(sorted(entry for entry in output.split("\0") if entry))
        if len(paths) > _MAX_DELTA_PATHS:
            return paths[:_MAX_DELTA_PATHS], True
        return paths, False


def _repository_alternates(git_dir: Path) -> list[str]:
    """Object directories the repository itself borrows from (shared clones)."""
    alternates = git_dir / "objects" / "info" / "alternates"
    try:
        content = alternates.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    return [line.strip() for line in content.splitlines() if line.strip()]


def _is_object_id(value: str) -> bool:
    return len(value) in (_SHA1_LENGTH, _SHA256_LENGTH) and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class GitWorkspaceStateSensor:
    """Observe the whole working tree of one configured workspace."""

    probe: GitWorkspaceProbe
    resource_id: str = "workspace"
    resource_type: str = "git"
    collector_version: str = GIT_COLLECTOR_VERSION

    def capture(
        self, *, tool_event_ref: str, phase: str, before: StateEvidence | None = None
    ) -> StateEvidence:
        return collect_git_evidence(
            self.probe,
            tool_event_ref=tool_event_ref,
            resource_id=self.resource_id,
            phase=phase,  # type: ignore[arg-type]
            before=before,
        )


def collect_git_evidence(
    probe: GitWorkspaceProbe,
    *,
    tool_event_ref: str,
    resource_id: str,
    phase: Phase,
    before: StateEvidence | None = None,
) -> StateEvidence:
    """Capture the working tree's content fingerprint at one boundary."""
    prior = _validate_boundary(
        before,
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type="git",
        resource_id=resource_id,
        collector_version=GIT_COLLECTOR_VERSION,
    )
    tree = probe.tree()
    if tree is None:
        return _evidence(
            tool_event_ref=tool_event_ref,
            phase=phase,
            resource_type="git",
            resource_id=resource_id,
            status="collection_failed",
            raw_fingerprint=None,
            canonical_fingerprint=None,
            delta=None,
            collector_version=GIT_COLLECTOR_VERSION,
        )
    delta: dict[str, Any] | None = None
    if prior is not None and prior.status == "observed":
        previous = git_tree_of(prior)
        delta = {"changed": previous != tree}
        if previous != tree:
            listed = probe.changed_paths(previous, tree)
            if listed is not None:
                delta["paths"], delta["truncated"] = list(listed[0]), listed[1]
    return _evidence(
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type="git",
        resource_id=resource_id,
        status="observed",
        raw_fingerprint={"tree": tree},
        canonical_fingerprint=_TREE_PREFIX + tree,
        delta=delta,
        collector_version=GIT_COLLECTOR_VERSION,
    )


def git_tree_of(evidence: StateEvidence) -> str:
    """The tree id a git record carries; raises ``ValueError`` on any mismatch.

    Consumers must validate the collector's stronger contract before trusting a
    fingerprint: the generic schema only guarantees bounded JSON.
    """
    raw = evidence.raw_fingerprint
    if not isinstance(raw, Mapping) or raw.keys() != {"tree"}:
        raise ValueError("git evidence has an invalid fingerprint")
    tree = raw["tree"]
    if not isinstance(tree, str) or not _is_object_id(tree):
        raise ValueError("git evidence has an invalid fingerprint")
    if evidence.canonical_fingerprint != _TREE_PREFIX + tree:
        raise ValueError("git evidence has an invalid fingerprint")
    return tree


@dataclass(frozen=True, slots=True)
class CommandStateSensor:
    """Observe whether one explicitly configured command still succeeds."""

    command: tuple[str, ...]
    cwd: str | Path
    resource_id: str
    timeout_seconds: float = 60.0
    resource_type: str = "command"
    collector_version: str = COMMAND_COLLECTOR_VERSION

    def capture(
        self, *, tool_event_ref: str, phase: str, before: StateEvidence | None = None
    ) -> StateEvidence:
        return collect_command_evidence(
            self.command,
            cwd=self.cwd,
            tool_event_ref=tool_event_ref,
            resource_id=self.resource_id,
            phase=phase,  # type: ignore[arg-type]
            before=before,
            timeout_seconds=self.timeout_seconds,
        )


def collect_command_evidence(
    command: Sequence[str],
    *,
    cwd: str | Path,
    tool_event_ref: str,
    resource_id: str,
    phase: Phase,
    before: StateEvidence | None = None,
    timeout_seconds: float = 60.0,
) -> StateEvidence:
    """Run one configured command without a shell and capture success/failure.

    Only ``success`` or ``failure`` is canonical. Exit statuses are not
    distinguished: an agent churning between two error codes would otherwise
    look like it was reaching new states.
    """
    validate_command(command)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    prior = _validate_boundary(
        before,
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type="command",
        resource_id=resource_id,
        collector_version=COMMAND_COLLECTOR_VERSION,
    )
    try:
        completed = subprocess.run(
            list(command),
            cwd=Path(cwd),
            capture_output=True,
            timeout=float(timeout_seconds),
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        # A command that cannot run, or that outlives its budget, is an absent
        # observation -- never a failing one, which would read as real evidence.
        return _evidence(
            tool_event_ref=tool_event_ref,
            phase=phase,
            resource_type="command",
            resource_id=resource_id,
            status="collection_failed",
            raw_fingerprint=None,
            canonical_fingerprint=None,
            delta=None,
            collector_version=COMMAND_COLLECTOR_VERSION,
        )
    outcome = _SUCCESS if completed.returncode == 0 else _FAILURE
    delta: dict[str, Any] | None = None
    if prior is not None and prior.status == "observed":
        previous = command_outcome_of(prior)
        delta = {"changed": previous != outcome, "before": previous, "after": outcome}
    return _evidence(
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type="command",
        resource_id=resource_id,
        status="observed",
        raw_fingerprint={"outcome": outcome},
        canonical_fingerprint=outcome,
        delta=delta,
        collector_version=COMMAND_COLLECTOR_VERSION,
    )


def command_outcome_of(evidence: StateEvidence) -> str:
    """The outcome a command record carries; raises ``ValueError`` on any mismatch."""
    raw = evidence.raw_fingerprint
    if not isinstance(raw, Mapping) or raw.keys() != {"outcome"}:
        raise ValueError("command evidence has an invalid fingerprint")
    outcome = raw["outcome"]
    if outcome not in (_SUCCESS, _FAILURE) or evidence.canonical_fingerprint != outcome:
        raise ValueError("command evidence has an invalid fingerprint")
    assert isinstance(outcome, str)
    return outcome


def validate_command(command: Sequence[str]) -> tuple[str, ...]:
    """Validate a configured argument vector; raises ``ValueError`` when unusable."""
    if isinstance(command, (str, bytes)):
        raise ValueError("command must be an argument list, not a string")
    arguments = tuple(command)
    if not arguments or len(arguments) > _MAX_COMMAND_LENGTH:
        raise ValueError(f"command must contain 1 to {_MAX_COMMAND_LENGTH} arguments")
    for argument in arguments:
        if (
            not isinstance(argument, str)
            or not argument
            or len(argument) > _MAX_COMMAND_ARGUMENT_LENGTH
            or "\x00" in argument
        ):
            raise ValueError("each command argument must be a non-empty bounded string")
    return arguments


def _validate_boundary(
    before: StateEvidence | None,
    *,
    tool_event_ref: str,
    phase: Phase,
    resource_type: str,
    resource_id: str,
    collector_version: str,
) -> StateEvidence | None:
    """Check the prior observation belongs to *this* boundary and resource.

    Mirrors ``pathlayer.state_collectors._validate_boundary``: these collectors
    are public, so a caller pairing an ``after`` capture with a ``before`` from
    another tool call or another resource must be refused here rather than
    silently producing a delta between two unrelated observations.
    """
    if phase not in ("before", "after"):
        raise ValueError("phase must be 'before' or 'after'")
    if phase == "before":
        if before is not None:
            raise ValueError("before evidence cannot be supplied for a before capture")
        return None
    if before is None:
        raise ValueError("after capture requires explicit before evidence")
    if (
        before.phase != "before"
        or before.tool_event_ref != tool_event_ref
        or before.resource_type != resource_type
        or before.resource_id != resource_id
        or before.collector_version != collector_version
    ):
        raise ValueError("before evidence does not match this capture boundary")
    return before


def _evidence(
    *,
    tool_event_ref: str,
    phase: Phase,
    resource_type: str,
    resource_id: str,
    status: str,
    raw_fingerprint: Any,
    canonical_fingerprint: Any,
    delta: Any,
    collector_version: str,
) -> StateEvidence:
    evidence = validate_state_evidence(
        {
            "schema_version": 1,
            "tool_event_ref": tool_event_ref,
            "phase": phase,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "status": status,
            "raw_fingerprint": raw_fingerprint,
            "canonical_fingerprint": canonical_fingerprint,
            "delta": delta,
            "collector_version": collector_version,
        }
    )
    if evidence is None:
        raise ValueError("invalid StateEvidence collector arguments")
    return evidence


__all__ = [
    "COMMAND_COLLECTOR_VERSION",
    "GIT_COLLECTOR_VERSION",
    "CommandStateSensor",
    "GitWorkspaceError",
    "GitWorkspaceProbe",
    "GitWorkspaceStateSensor",
    "collect_command_evidence",
    "collect_git_evidence",
    "command_outcome_of",
    "git_directory",
    "git_tree_of",
    "validate_command",
]
