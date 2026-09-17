"""Bounded filesystem and pytest ``StateEvidence`` collectors.

This module's bytes are one of the inputs COMP-1 hashes into its seal (see
``benchmarks/comp1/run.py``), so a new collector belongs in
:mod:`pathlayer.coding_state_collectors` unless it is genuinely part of the
sealed set. These deterministic helpers are shared by the offline COMP-1
harness and the opt-in runtime producer introduced by ARCH-2. They bracket a caller-owned tool
boundary with explicit
``before``/``after`` records and deliberately keep the prior observation in
the call signature: an unavailable past observation can be represented, but
it can never be reconstructed from the present.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pathlayer._pytest_evidence_protocol import (
    MAX_NODEID_LENGTH,
    MAX_OUTCOMES,
    OUTPUT_ENV,
)
from pathlayer.detectors.state_evidence import StateEvidence, validate_state_evidence

Phase = Literal["before", "after"]

_EXPORT_SCHEMA_VERSION = 1
_FILESYSTEM_COLLECTOR_VERSION = "filesystem-collector@1"
_PYTEST_COLLECTOR_VERSION = "tests-collector@1"
_SHA256_PREFIX = "sha256:"
_MISSING_FINGERPRINT = "missing"
_MAX_EXPORT_RECORDS = 10_000
_MAX_EXPORT_BYTES = 64 * 1024 * 1024
_MAX_PYTEST_OUTPUT_BYTES = 8 * 1024 * 1024
_HASH_CHUNK_SIZE = 64 * 1024

__all__ = [
    "FILESYSTEM_COLLECTOR_VERSION",
    "PYTEST_COLLECTOR_VERSION",
    "collect_filesystem_evidence",
    "collect_pytest_evidence",
    "dump_state_evidence_export",
    "load_state_evidence_export",
    "unavailable_state_evidence",
]

FILESYSTEM_COLLECTOR_VERSION = _FILESYSTEM_COLLECTOR_VERSION
PYTEST_COLLECTOR_VERSION = _PYTEST_COLLECTOR_VERSION


def unavailable_state_evidence(
    *,
    tool_event_ref: str,
    resource_type: Literal["filesystem", "tests"],
    resource_id: str,
    phase: Phase,
) -> StateEvidence:
    """Represent a capture that the caller could not attempt at this boundary."""
    return _make_evidence(
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type=resource_type,
        resource_id=resource_id,
        status="unavailable",
        raw_fingerprint=None,
        canonical_fingerprint=None,
        delta=None,
        collector_version=_collector_version(resource_type),
    )


def collect_filesystem_evidence(
    path: str | Path,
    *,
    tool_event_ref: str,
    resource_id: str,
    phase: Phase,
    before: StateEvidence | None = None,
) -> StateEvidence:
    """Capture existence and a stable content hash for one regular file.

    Missing paths are successful observations.  Existing symlinks,
    directories and other non-regular resources are collection failures: the
    bounded COMP-1 collector intentionally does not define tree or link
    semantics.
    """
    prior = _validate_boundary(
        before,
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type="filesystem",
        resource_id=resource_id,
        collector_version=_FILESYSTEM_COLLECTOR_VERSION,
    )
    target = Path(path)
    try:
        try:
            metadata = target.lstat()
        except FileNotFoundError:
            raw_fingerprint: dict[str, Any] = {"exists": False, "sha256": None}
            canonical_fingerprint = _MISSING_FINGERPRINT
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise OSError("symbolic links are outside the bounded filesystem collector")
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("resource is not a regular file")
            digest = _sha256_file(target, metadata)
            raw_fingerprint = {"exists": True, "sha256": digest}
            canonical_fingerprint = _SHA256_PREFIX + digest
    except (OSError, ValueError):
        # lstat()/os.open() raise ValueError (not OSError) for a path containing
        # an embedded NUL byte -- another way "this isn't a valid regular-file
        # resource" surfaces, so it degrades the same as any other OSError here.
        return _make_evidence(
            tool_event_ref=tool_event_ref,
            phase=phase,
            resource_type="filesystem",
            resource_id=resource_id,
            status="collection_failed",
            raw_fingerprint=None,
            canonical_fingerprint=None,
            delta=None,
            collector_version=_FILESYSTEM_COLLECTOR_VERSION,
        )

    delta: dict[str, Any] | None = None
    if prior is not None and prior.status == "observed":
        prior_fingerprint = _filesystem_canonical_fingerprint(prior)
        delta = _filesystem_delta(prior_fingerprint, canonical_fingerprint)
    return _make_evidence(
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type="filesystem",
        resource_id=resource_id,
        status="observed",
        raw_fingerprint=raw_fingerprint,
        canonical_fingerprint=canonical_fingerprint,
        delta=delta,
        collector_version=_FILESYSTEM_COLLECTOR_VERSION,
    )


def collect_pytest_evidence(
    test_ids: Sequence[str],
    *,
    cwd: str | Path,
    tool_event_ref: str,
    resource_id: str,
    phase: Phase,
    before: StateEvidence | None = None,
    timeout_seconds: float = 60.0,
) -> StateEvidence:
    """Run exact pytest leaf nodeids and capture their pass/non-pass state.

    Pytest is invoked through the current Python interpreter without a shell.
    Its internal capture plugin emits only stable nodeids and terminal
    outcomes; console output, timing and host metadata are ignored.
    """
    selected = _validate_test_ids(test_ids)
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
        resource_type="tests",
        resource_id=resource_id,
        collector_version=_PYTEST_COLLECTOR_VERSION,
    )
    prior_outcomes: dict[str, str] | None = None
    if prior is not None and prior.status == "observed":
        prior_outcomes = _test_outcomes(prior)
        if set(prior_outcomes) != set(selected):
            raise ValueError("before evidence covers a different pytest selection")

    outcomes = _run_pytest(selected, Path(cwd), float(timeout_seconds))
    if outcomes is None:
        return _make_evidence(
            tool_event_ref=tool_event_ref,
            phase=phase,
            resource_type="tests",
            resource_id=resource_id,
            status="collection_failed",
            raw_fingerprint=None,
            canonical_fingerprint=None,
            delta=None,
            collector_version=_PYTEST_COLLECTOR_VERSION,
        )

    canonical_fingerprint = _canonical_test_fingerprint(outcomes)
    delta: dict[str, list[str]] | None = None
    if prior_outcomes is not None:
        delta = _test_delta(prior_outcomes, outcomes)
    return _make_evidence(
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type="tests",
        resource_id=resource_id,
        status="observed",
        raw_fingerprint={"outcomes": dict(sorted(outcomes.items()))},
        canonical_fingerprint=canonical_fingerprint,
        delta=delta,
        collector_version=_PYTEST_COLLECTOR_VERSION,
    )


def dump_state_evidence_export(evidence: Sequence[StateEvidence], path: str | Path) -> None:
    """Write a validated, ordered V1 evidence export as UTF-8 JSON."""
    records = tuple(evidence)
    _validate_export_records(records)
    payload = {
        "schema_version": _EXPORT_SCHEMA_VERSION,
        "evidence": [asdict(record) for record in records],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > _MAX_EXPORT_BYTES:
        raise ValueError(f"state evidence export exceeds {_MAX_EXPORT_BYTES} bytes")
    output.write_bytes(encoded)


def load_state_evidence_export(path: str | Path) -> tuple[StateEvidence, ...]:
    """Load and validate a closed-schema V1 evidence export."""
    raw = _load_bounded_json(Path(path))
    if not isinstance(raw, Mapping) or raw.keys() != {"schema_version", "evidence"}:
        raise ValueError("state evidence export must contain exactly schema_version and evidence")
    version = raw["schema_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != _EXPORT_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported state evidence export schema version: {version!r}")
    raw_records = raw["evidence"]
    if not isinstance(raw_records, list):
        raise ValueError("state evidence export evidence must be a list")
    records: list[StateEvidence] = []
    for index, raw_record in enumerate(raw_records):
        record = validate_state_evidence(raw_record)
        if record is None:
            raise ValueError(f"invalid StateEvidence record at index {index}")
        records.append(record)
    result = tuple(records)
    _validate_export_records(result)
    return result


def _collector_version(resource_type: str) -> str:
    if resource_type == "filesystem":
        return _FILESYSTEM_COLLECTOR_VERSION
    if resource_type == "tests":
        return _PYTEST_COLLECTOR_VERSION
    raise ValueError(f"unsupported resource_type: {resource_type!r}")


def _make_evidence(
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
    raw = {
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
    evidence = validate_state_evidence(raw)
    if evidence is None:
        raise ValueError("invalid StateEvidence collector arguments")
    return evidence


def _validate_boundary(
    before: StateEvidence | None,
    *,
    tool_event_ref: str,
    phase: Phase,
    resource_type: str,
    resource_id: str,
    collector_version: str,
) -> StateEvidence | None:
    if phase not in ("before", "after"):
        raise ValueError("phase must be 'before' or 'after'")
    if phase == "before":
        if before is not None:
            raise ValueError("before evidence cannot be supplied for a before capture")
        return None
    if before is None:
        raise ValueError("after capture requires explicit before evidence")
    validated = validate_state_evidence(asdict(before))
    if validated is None or (
        validated.phase != "before"
        or validated.tool_event_ref != tool_event_ref
        or validated.resource_type != resource_type
        or validated.resource_id != resource_id
        or validated.collector_version != collector_version
    ):
        raise ValueError("before evidence does not match this capture boundary")
    return validated


def _sha256_file(path: Path, expected: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (expected.st_dev, expected.st_ino):
            raise OSError("resource changed identity during collection")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
            finished = os.fstat(handle.fileno())
            if (opened.st_size, opened.st_mtime_ns) != (
                finished.st_size,
                finished.st_mtime_ns,
            ):
                raise OSError("resource changed while it was being hashed")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _filesystem_canonical_fingerprint(evidence: StateEvidence) -> str:
    raw = evidence.raw_fingerprint
    fingerprint = evidence.canonical_fingerprint
    if not isinstance(raw, Mapping) or raw.keys() != {"exists", "sha256"}:
        raise ValueError("filesystem evidence has an invalid fingerprint")
    exists = raw["exists"]
    digest = raw["sha256"]
    if not isinstance(exists, bool):
        raise ValueError("filesystem evidence has an invalid fingerprint")
    if exists:
        if not isinstance(digest, str) or not _is_sha256_digest(digest):
            raise ValueError("filesystem evidence has an invalid fingerprint")
        expected = _SHA256_PREFIX + digest
    else:
        if digest is not None:
            raise ValueError("filesystem evidence has an invalid fingerprint")
        expected = _MISSING_FINGERPRINT
    if fingerprint != expected:
        raise ValueError("filesystem evidence has an invalid fingerprint")
    return expected


def _is_sha256_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _content_hash_or_none(fingerprint: str) -> str | None:
    return fingerprint if fingerprint.startswith(_SHA256_PREFIX) else None


def _filesystem_delta(before: str, after: str) -> dict[str, Any]:
    return {
        "changed": before != after,
        "hash_before": _content_hash_or_none(before),
        "hash_after": _content_hash_or_none(after),
    }


def _validate_test_ids(test_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(test_ids, (str, bytes)):
        raise ValueError("test_ids must be a sequence of exact pytest nodeids")
    selected = tuple(test_ids)
    if not selected or len(selected) > MAX_OUTCOMES:
        raise ValueError(f"test_ids must contain between 1 and {MAX_OUTCOMES} nodeids")
    if any(
        not isinstance(test_id, str)
        or not test_id
        or len(test_id) > MAX_NODEID_LENGTH
        or test_id.startswith("-")
        or "\x00" in test_id
        for test_id in selected
    ):
        # The NUL check matters beyond well-formedness: subprocess.Popen raises
        # ValueError (uncaught by _run_pytest's except tuple) for an argument
        # containing one, so this rejects it here instead of crashing the harness.
        raise ValueError("each pytest nodeid must be a non-empty bounded string")
    if len(set(selected)) != len(selected):
        raise ValueError("pytest nodeids must be unique")
    return tuple(sorted(selected))


def _run_pytest(
    test_ids: tuple[str, ...], cwd: Path, timeout_seconds: float
) -> dict[str, str] | None:
    env = os.environ.copy()
    env[OUTPUT_ENV] = ""
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    try:
        with tempfile.TemporaryDirectory(prefix="pathlayer-pytest-evidence-") as tmp_dir:
            output = Path(tmp_dir) / "outcomes.json"
            env[OUTPUT_ENV] = str(output)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=no",
                    "--rootdir=.",
                    "-o",
                    "addopts=",
                    "-p",
                    "no:cacheprovider",
                    "-p",
                    "pathlayer.validation._pytest_evidence_plugin",
                    "--",
                    *test_ids,
                ],
                cwd=cwd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=sys.platform != "win32",
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if sys.platform == "win32"
                    else 0
                ),
            )
            return_code = _wait_for_pytest(process, timeout_seconds)
            if return_code not in (0, 1) or not output.is_file():
                return None
            if output.stat().st_size > _MAX_PYTEST_OUTPUT_BYTES:
                return None
            raw: Any = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    if not isinstance(raw, Mapping) or raw.keys() != {"exit_code", "outcomes"}:
        return None
    if raw["exit_code"] != return_code or not isinstance(raw["outcomes"], Mapping):
        return None
    outcomes = raw["outcomes"]
    if outcomes.keys() != set(test_ids):
        return None
    if any(status not in ("passed", "failed") for status in outcomes.values()):
        return None
    return dict(outcomes)


def _test_outcomes(evidence: StateEvidence) -> dict[str, str]:
    raw = evidence.raw_fingerprint
    if not isinstance(raw, Mapping) or raw.keys() != {"outcomes"}:
        raise ValueError("tests evidence has an invalid fingerprint")
    outcomes = raw["outcomes"]
    if (
        not isinstance(outcomes, Mapping)
        or not outcomes
        or len(outcomes) > MAX_OUTCOMES
        or any(
            not isinstance(test_id, str)
            or not test_id
            or len(test_id) > MAX_NODEID_LENGTH
            or status not in ("passed", "failed")
            for test_id, status in outcomes.items()
        )
    ):
        raise ValueError("tests evidence has invalid pytest outcomes")
    result = dict(outcomes)
    if evidence.canonical_fingerprint != _canonical_test_fingerprint(result):
        raise ValueError("tests evidence has invalid pytest outcomes")
    return result


def _test_delta(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, list[str]]:
    if before.keys() != after.keys():
        raise ValueError("before evidence covers a different pytest selection")
    before_passing = {test_id for test_id, status in before.items() if status == "passed"}
    before_failing = set(before) - before_passing
    after_passing = {test_id for test_id, status in after.items() if status == "passed"}
    after_failing = set(after) - after_passing
    return {
        "newly_passing": sorted(after_passing - before_passing),
        "newly_failing": sorted(after_failing - before_failing),
        "still_failing": sorted(after_failing & before_failing),
    }


def _canonical_test_fingerprint(outcomes: Mapping[str, str]) -> dict[str, list[str]]:
    passing = sorted(test_id for test_id, status in outcomes.items() if status == "passed")
    failing = sorted(test_id for test_id, status in outcomes.items() if status != "passed")
    return {"failing": failing, "passing": passing}


def _validate_export_records(records: tuple[StateEvidence, ...]) -> None:
    if not records or len(records) > _MAX_EXPORT_RECORDS:
        raise ValueError(f"state evidence export must contain 1 to {_MAX_EXPORT_RECORDS} records")
    seen: set[tuple[str, str, str, str, str]] = set()
    groups: dict[tuple[str, str, str, str], list[tuple[int, StateEvidence]]] = {}
    boundary_indices: dict[str, dict[str, list[int]]] = {}
    for index, record in enumerate(records):
        validated = validate_state_evidence(asdict(record))
        if validated is None:
            raise ValueError(f"invalid StateEvidence record at index {index}")
        _validate_collector_evidence(validated)
        unique_key = (
            validated.tool_event_ref,
            validated.phase,
            validated.resource_type,
            validated.resource_id,
            validated.collector_version,
        )
        if unique_key in seen:
            raise ValueError(f"duplicate StateEvidence record at index {index}")
        seen.add(unique_key)
        group_key = (
            validated.tool_event_ref,
            validated.resource_type,
            validated.resource_id,
            validated.collector_version,
        )
        groups.setdefault(group_key, []).append((index, validated))
        boundary_indices.setdefault(validated.tool_event_ref, {"before": [], "after": []})[
            validated.phase
        ].append(index)

    for observations in groups.values():
        phases = {record.phase for _, record in observations}
        if phases != {"before", "after"} or len(observations) != 2:
            raise ValueError("each exported resource requires exactly one before/after pair")
        by_phase = {record.phase: (index, record) for index, record in observations}
        before_index, before = by_phase["before"]
        after_index, after = by_phase["after"]
        if before_index >= after_index:
            raise ValueError("before evidence must precede matching after evidence")
        both_observed = before.status == "observed" and after.status == "observed"
        if both_observed != (after.delta is not None):
            raise ValueError("after delta must exist exactly when both observations succeeded")
        expected_delta: Any = None
        if both_observed and before.resource_type == "filesystem":
            expected_delta = _filesystem_delta(
                _filesystem_canonical_fingerprint(before),
                _filesystem_canonical_fingerprint(after),
            )
        elif both_observed and before.resource_type == "tests":
            expected_delta = _test_delta(_test_outcomes(before), _test_outcomes(after))
        if after.delta != expected_delta:
            raise ValueError("after delta does not match the before/after observations")

    for boundary_phases in boundary_indices.values():
        if max(boundary_phases["before"]) >= min(boundary_phases["after"]):
            raise ValueError("all before evidence must precede after evidence for a tool boundary")


def _validate_collector_evidence(evidence: StateEvidence) -> None:
    expected_version = _collector_version(evidence.resource_type)
    if evidence.collector_version != expected_version:
        raise ValueError("StateEvidence collector version does not match its resource type")
    if evidence.status != "observed":
        return
    if evidence.resource_type == "filesystem":
        _filesystem_canonical_fingerprint(evidence)
    else:
        _test_outcomes(evidence)


def _load_bounded_json(path: Path) -> Any:
    with path.open("rb") as handle:
        encoded = handle.read(_MAX_EXPORT_BYTES + 1)
    if len(encoded) > _MAX_EXPORT_BYTES:
        raise ValueError(f"state evidence export exceeds {_MAX_EXPORT_BYTES} bytes")
    try:
        return json.loads(encoded)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("state evidence export is not valid bounded JSON") from exc


def _wait_for_pytest(process: subprocess.Popen[bytes], timeout_seconds: float) -> int | None:
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_and_reap_pytest(process)
        return None
    except BaseException:
        _terminate_and_reap_pytest(process)
        raise


def _terminate_and_reap_pytest(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        if sys.platform == "win32":
            _terminate_windows_process_tree(process)
        else:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _terminate_windows_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        completed = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        process.kill()
        return
    if completed.returncode != 0:
        process.kill()
