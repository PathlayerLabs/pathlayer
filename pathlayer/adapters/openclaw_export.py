"""Export one recorded OpenClaw run directory as a tar.gz archive.

An exported archive carries the whole run verbatim -- prompts, tool arguments,
tool results -- so it is at least as sensitive as the run directory it comes
from. Both files this module writes (the archive and its checksum manifest) are
therefore created owner-only (``0600``) regardless of the caller's umask, and
both are moved into place with :func:`os.replace` so the final path never exists
in a half-written or more permissive state.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import tempfile
from pathlib import Path

from pathlayer.adapters.openclaw_run_store import RunStore, atomic_write_json, restrict_file

DEFAULT_RUNS_ROOT = Path("/var/lib/pathlayer/runs")


def export_run(
    run_id: str,
    output: Path,
    *,
    runs_root: Path | None = None,
) -> Path:
    """Package exactly one run directory plus a SHA-256 checksum manifest."""
    store = RunStore(runs_root or _runs_root_from_env())
    matches = list(store.iter_matching(run_id))
    if not matches:
        raise LookupError(f"no recorded run found for run_id={run_id!r}")
    if len(matches) > 1:
        raise LookupError(
            f"ambiguous run_id={run_id!r}: {len(matches)} directories match; refine the id"
        )
    run = matches[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = _write_restricted_archive(run.root, run.dir_name, output)
    manifest = {
        "run_id": run.original_run_id,
        "dir_name": run.dir_name,
        "files": checksums,
    }
    # atomic_write_json creates its temp file owner-only and replaces the
    # destination with it, so the manifest lands at 0600 like the archive.
    manifest_path = Path(f"{output}.sha256.json")
    atomic_write_json(manifest_path, manifest)
    return output


def _write_restricted_archive(root: Path, arc_root: str, output: Path) -> dict[str, str]:
    """Write ``root`` as an owner-only tar.gz at ``output``; return checksums.

    The tarball is built into a temp file in the destination directory --
    created ``0600`` by :func:`tempfile.mkstemp`, re-asserted by
    :func:`restrict_file` so the mode never depends on the umask -- and only
    then moved onto ``output``. A pre-existing, more permissive ``output`` is
    replaced by the restricted file rather than written into, and a failed
    export leaves no partial archive behind.
    """
    checksums: dict[str, str] = {}
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with tarfile.open(fileobj=raw, mode="w:gz") as archive:
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        rel = path.relative_to(root).as_posix()
                        archive.add(path, arcname=f"{arc_root}/{rel}")
                        checksums[rel] = _sha256_file(path)
            raw.flush()
            os.fsync(raw.fileno())
        restrict_file(tmp_path)
        os.replace(tmp_path, output)
    finally:
        tmp_path.unlink(missing_ok=True)
    return checksums


def _runs_root_from_env() -> Path:
    return Path(os.environ.get("PATHLAYER_RUNS_DIR", DEFAULT_RUNS_ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export one recorded OpenClaw run.")
    parser.add_argument("run_id", help="Original OpenClaw runId (ctx.runId)")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output tar.gz path (checksum manifest written alongside)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Recorded runs root (default: $PATHLAYER_RUNS_DIR or /var/lib/pathlayer/runs)",
    )
    args = parser.parse_args(argv)
    try:
        export_run(args.run_id, args.output, runs_root=args.runs_root)
    except (LookupError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
