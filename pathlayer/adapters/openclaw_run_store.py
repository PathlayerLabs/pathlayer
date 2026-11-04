"""Filesystem persistence for one OpenClaw run directory."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pathlayer.adapters.openclaw_lifecycle import sanitize_run_id_for_path

SCHEMA_VERSION = 1
LISTENER_VERSION = "1"

# Where recorded runs live, and the environment variable that overrides it. Kept
# here (not in one CLI) so every entry point -- the recorder, the export CLI, the
# report CLI -- resolves the same root from the same single definition.
RUNS_DIR_ENV = "PATHLAYER_RUNS_DIR"
DEFAULT_RUNS_ROOT = Path("/var/lib/pathlayer/runs")

# Manifest statuses a run never leaves: it has an ``ended_at`` and its files are
# final. Reopening such a run would restart the sequence counter at 1 and
# overwrite ``trace.json`` / ``final.json`` with the state of an empty run, so
# both the recorder (any further hook) and the startup sweep skip them. The one
# exception is ``late.jsonl``: a cost hook that lost the race with
# ``agent_end`` is appended there, and the derived ``report.json`` rebuilt.
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "interrupted"})


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def runs_root_from_env() -> Path:
    """Recorded-runs root: ``$PATHLAYER_RUNS_DIR``, else :data:`DEFAULT_RUNS_ROOT`."""
    return Path(os.environ.get(RUNS_DIR_ENV, DEFAULT_RUNS_ROOT))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically via a temp file in the same directory.

    The temp file is created owner-only and moved into place with
    :func:`os.replace`, so the destination is never observed half-written and
    never inherits looser permissions from the umask (recorded runs carry
    prompts and tool results -- see the privacy note in
    ``examples/openclaw_plugin/README.md``). If anything fails before the
    replace, the previous version of ``path`` is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        restrict_file(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def restrict_run_dir(path: Path) -> None:
    """Best-effort restrictive permissions (owner-only) on Unix."""
    if os.name != "nt":
        path.chmod(0o700)


def restrict_file(path: Path) -> None:
    """Best-effort owner-only permissions (``0600``) on Unix.

    Windows has no POSIX mode bits to set (``chmod`` there only toggles the
    read-only flag), so this is a no-op on ``nt`` -- same contract as
    :func:`restrict_run_dir`.
    """
    if os.name != "nt":
        path.chmod(0o600)


@dataclass(slots=True)
class RunDirectory:
    """On-disk layout for one recorded OpenClaw run."""

    root: Path
    original_run_id: str
    dir_name: str

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def request_path(self) -> Path:
        return self.root / "request.json"

    @property
    def raw_events_path(self) -> Path:
        return self.root / "events.raw.jsonl"

    @property
    def trace_path(self) -> Path:
        return self.root / "trace.json"

    @property
    def decisions_path(self) -> Path:
        return self.root / "decisions.jsonl"

    @property
    def final_path(self) -> Path:
        return self.root / "final.json"

    @property
    def report_path(self) -> Path:
        """End-of-session report, written once the run is finalized."""
        return self.root / "report.json"

    @property
    def late_events_path(self) -> Path:
        """Append-only annex of cost hooks that arrived after the run ended."""
        return self.root / "late.jsonl"

    def read_manifest(self) -> dict[str, Any]:
        raw: Any = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("manifest must be a JSON object")
        return raw

    def update_manifest(self, **fields: Any) -> dict[str, Any]:
        manifest = self.read_manifest()
        manifest.update(fields)
        atomic_write_json(self.manifest_path, manifest)
        return manifest


class RunStore:
    """Manage run directories under a shared root."""

    def __init__(self, runs_root: Path) -> None:
        self._root = runs_root
        self._lock = threading.Lock()
        self._root.mkdir(parents=True, exist_ok=True)
        restrict_run_dir(self._root)

    @property
    def root(self) -> Path:
        return self._root

    def resolve_run(self, run_id: str) -> RunDirectory | None:
        """Find a run by original (unsanitized) ID."""
        safe = sanitize_run_id_for_path(run_id)
        direct = self._root / safe
        if direct.is_dir() and (direct / "manifest.json").is_file():
            manifest = json.loads((direct / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("run_id") == run_id:
                return RunDirectory(root=direct, original_run_id=run_id, dir_name=safe)
        for candidate in self._root.iterdir():
            if not candidate.is_dir():
                continue
            manifest_path = candidate / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_id") == run_id:
                return RunDirectory(
                    root=candidate,
                    original_run_id=run_id,
                    dir_name=candidate.name,
                )
        return None

    def list_runs(self) -> list[RunDirectory]:
        runs: list[RunDirectory] = []
        if not self._root.is_dir():
            return runs
        for candidate in sorted(self._root.iterdir()):
            manifest_path = candidate / "manifest.json"
            if not candidate.is_dir() or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_id = manifest.get("run_id")
            if isinstance(run_id, str):
                runs.append(
                    RunDirectory(root=candidate, original_run_id=run_id, dir_name=candidate.name)
                )
        return runs

    def open_or_create(
        self,
        run_id: str,
        *,
        manifest: Mapping[str, Any],
        allow_existing: bool = True,
    ) -> RunDirectory:
        with self._lock:
            existing = self.resolve_run(run_id)
            if existing is not None:
                if not allow_existing:
                    raise FileExistsError(f"run already exists: {run_id!r}")
                return existing
            dir_name = sanitize_run_id_for_path(run_id)
            run_dir = self._root / dir_name
            if run_dir.exists():
                # Same sanitized name, different original ID -- refuse to overwrite.
                if (run_dir / "manifest.json").is_file():
                    other = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                    if other.get("run_id") != run_id:
                        raise FileExistsError(
                            f"sanitized run directory {dir_name!r} already used by "
                            f"{other.get('run_id')!r}"
                        )
                else:
                    raise FileExistsError(f"run directory exists without manifest: {dir_name!r}")
            run_dir.mkdir(parents=True, exist_ok=False)
            restrict_run_dir(run_dir)
            run = RunDirectory(root=run_dir, original_run_id=run_id, dir_name=dir_name)
            atomic_write_json(run.manifest_path, manifest)
            return run

    def mark_unfinished_as_interrupted(self) -> list[RunDirectory]:
        """On listener startup, preserve unfinished runs and mark them interrupted.

        Returns the runs it just marked, so a caller can finish them off -- the
        recorder writes their ``report.json`` there, since a run whose process is
        gone will never reach ``agent_end`` to write it itself.
        """
        interrupted: list[RunDirectory] = []
        for run in self.list_runs():
            manifest = run.read_manifest()
            status = manifest.get("status")
            if status in TERMINAL_STATUSES:
                continue
            run.update_manifest(status="interrupted", ended_at=utc_now_iso())
            interrupted.append(run)
        return interrupted

    def iter_matching(self, run_id: str) -> Iterator[RunDirectory]:
        matches = [run for run in self.list_runs() if run.original_run_id == run_id]
        yield from matches
