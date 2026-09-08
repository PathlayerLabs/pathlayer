"""Opt-in runtime capture and transport for ``StateEvidence`` (ARCH-2).

Collection happens at an adapter's tool boundary. The resulting closed-schema
records are embedded in ``ToolEvent.metadata`` so the pure pipeline, durable
store and trace replay can consume the observation without performing I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from pathlayer.detectors.state_evidence import StateEvidence, validate_state_evidence
from pathlayer.events import ToolEvent
from pathlayer.state_collectors import (
    FILESYSTEM_COLLECTOR_VERSION,
    PYTEST_COLLECTOR_VERSION,
    collect_filesystem_evidence,
    collect_pytest_evidence,
)

STATE_EVIDENCE_METADATA_KEY = "pathlayer_state_evidence"
_ENVELOPE_VERSION = 1
_MAX_SENSORS = 64


class StateOutcome(str, Enum):
    """Deterministic result of one tool boundary's state observations."""

    CHANGED = "changed"
    UNCHANGED = "unchanged"
    UNAVAILABLE = "unavailable"
    COLLECTION_FAILED = "collection_failed"


class StateSensor(Protocol):
    """One explicitly configured resource observed around every tool call."""

    @property
    def resource_type(self) -> str: ...

    @property
    def resource_id(self) -> str: ...

    @property
    def collector_version(self) -> str: ...

    def capture(
        self, *, tool_event_ref: str, phase: str, before: StateEvidence | None = None
    ) -> StateEvidence: ...


@dataclass(frozen=True, slots=True)
class FilesystemStateSensor:
    """Observe one regular file using the ARCH-1B filesystem collector."""

    path: str | Path
    resource_id: str
    resource_type: str = "filesystem"
    collector_version: str = FILESYSTEM_COLLECTOR_VERSION

    def capture(
        self, *, tool_event_ref: str, phase: str, before: StateEvidence | None = None
    ) -> StateEvidence:
        return collect_filesystem_evidence(
            self.path,
            tool_event_ref=tool_event_ref,
            resource_id=self.resource_id,
            phase=phase,  # type: ignore[arg-type]
            before=before,
        )


@dataclass(frozen=True, slots=True)
class PytestStateSensor:
    """Observe exact pytest nodeids using the ARCH-1B test collector."""

    test_ids: tuple[str, ...]
    cwd: str | Path
    resource_id: str
    timeout_seconds: float = 60.0
    resource_type: str = "tests"
    collector_version: str = PYTEST_COLLECTOR_VERSION

    def capture(
        self, *, tool_event_ref: str, phase: str, before: StateEvidence | None = None
    ) -> StateEvidence:
        return collect_pytest_evidence(
            self.test_ids,
            cwd=self.cwd,
            tool_event_ref=tool_event_ref,
            resource_id=self.resource_id,
            phase=phase,  # type: ignore[arg-type]
            before=before,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class StateCapture:
    """Run-local before observations passed explicitly to the after boundary."""

    tool_event_ref: str
    before: tuple[StateEvidence, ...]


@dataclass(frozen=True, slots=True)
class StateEvidenceEnvelope:
    """Validated state result transported on a ``ToolEvent``."""

    outcome: StateOutcome
    records: tuple[StateEvidence, ...]


class StateEvidenceProducer:
    """Bracket tool calls with a fixed, explicit set of deterministic sensors.

    The producer owns no per-run mutable state. A ``StateCapture`` is returned
    to the adapter and stored beside that adapter's pending tool call, which
    prevents concurrent runs from sharing or confusing observations.
    """

    def __init__(self, sensors: Sequence[StateSensor]) -> None:
        self._sensors = tuple(sensors)
        if not self._sensors:
            raise ValueError("StateEvidenceProducer requires at least one sensor")
        if len(self._sensors) > _MAX_SENSORS:
            raise ValueError(f"StateEvidenceProducer supports at most {_MAX_SENSORS} sensors")
        identities = [(sensor.resource_type, sensor.resource_id) for sensor in self._sensors]
        if len(set(identities)) != len(identities):
            raise ValueError("state sensor resource identities must be unique")
        for sensor in self._sensors:
            if _unavailable_record(sensor, tool_event_ref="validation", phase="before") is None:
                raise ValueError("state sensor identity or collector version is invalid")

    def capture_before(self, tool_event_ref: str) -> StateCapture:
        records = tuple(
            _capture_safely(sensor, tool_event_ref=tool_event_ref, phase="before")
            for sensor in self._sensors
        )
        return StateCapture(tool_event_ref=tool_event_ref, before=records)

    def unavailable_capture(self, tool_event_ref: str) -> StateCapture:
        """Return an explicit abstention for an overlapping tool boundary."""
        records = tuple(
            _required_unavailable_record(sensor, tool_event_ref=tool_event_ref, phase="before")
            for sensor in self._sensors
        )
        return StateCapture(tool_event_ref=tool_event_ref, before=records)

    def attach_after(self, event: ToolEvent, capture: StateCapture) -> ToolEvent:
        if event.run_id != capture.tool_event_ref:
            raise ValueError("StateCapture belongs to a different ToolEvent")
        after = tuple(
            _required_unavailable_record(
                sensor, tool_event_ref=capture.tool_event_ref, phase="after"
            )
            if prior.status == "unavailable"
            else _capture_safely(
                sensor,
                tool_event_ref=capture.tool_event_ref,
                phase="after",
                before=prior,
            )
            for sensor, prior in zip(self._sensors, capture.before, strict=True)
        )
        pairs = zip(capture.before, after, strict=True)
        records = tuple(record for pair in pairs for record in pair)
        envelope = {
            "schema_version": _ENVELOPE_VERSION,
            "outcome": _outcome(capture.before, after).value,
            "records": [asdict(record) for record in records],
        }
        metadata = dict(event.metadata)
        metadata[STATE_EVIDENCE_METADATA_KEY] = envelope
        return replace(event, metadata=metadata)


def state_evidence_from_event(event: ToolEvent) -> StateEvidenceEnvelope | None:
    """Read and validate embedded evidence without I/O; ``None`` if absent/invalid."""
    raw = event.metadata.get(STATE_EVIDENCE_METADATA_KEY)
    if not isinstance(raw, Mapping) or raw.keys() != {"schema_version", "outcome", "records"}:
        return None
    version = raw["schema_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != _ENVELOPE_VERSION
        or not isinstance(raw["records"], list)
    ):
        return None
    try:
        outcome = StateOutcome(raw["outcome"])
    except (TypeError, ValueError):
        return None
    records: list[StateEvidence] = []
    for item in raw["records"]:
        record = validate_state_evidence(item)
        if record is None:
            return None
        records.append(record)
    if not records or len(records) % 2:
        return None
    before = tuple(records[::2])
    after = tuple(records[1::2])
    identities: set[tuple[str, str]] = set()
    for prior, current in zip(before, after, strict=True):
        identity = (prior.resource_type, prior.resource_id)
        if (
            prior.phase != "before"
            or current.phase != "after"
            or prior.tool_event_ref != current.tool_event_ref
            or prior.resource_type != current.resource_type
            or prior.resource_id != current.resource_id
            or prior.collector_version != current.collector_version
            or prior.tool_event_ref != event.run_id
            or identity in identities
        ):
            return None
        identities.add(identity)
    if _outcome(before, after) is not outcome:
        return None
    return StateEvidenceEnvelope(outcome=outcome, records=tuple(records))


def _capture_safely(
    sensor: StateSensor,
    *,
    tool_event_ref: str,
    phase: str,
    before: StateEvidence | None = None,
) -> StateEvidence:
    try:
        record = sensor.capture(tool_event_ref=tool_event_ref, phase=phase, before=before)
        if not isinstance(record, StateEvidence):
            raise TypeError("sensor did not return StateEvidence")
        validated_record = validate_state_evidence(_record_mapping(record))
        if validated_record is None or (
            validated_record.tool_event_ref != tool_event_ref
            or validated_record.phase != phase
            or validated_record.resource_type != sensor.resource_type
            or validated_record.resource_id != sensor.resource_id
            or validated_record.collector_version != sensor.collector_version
        ):
            raise ValueError("sensor returned mismatched StateEvidence")
        return validated_record
    except Exception:
        # A sensor is an integration boundary. Its failure must be explicit but
        # must not break the observed agent. Do not include exception text: it
        # can carry secrets and is irrelevant to deterministic replay.
        raw = {
            "schema_version": 1,
            "tool_event_ref": tool_event_ref,
            "phase": phase,
            "resource_type": sensor.resource_type,
            "resource_id": sensor.resource_id,
            "status": "collection_failed",
            "raw_fingerprint": None,
            "canonical_fingerprint": None,
            "delta": None,
            "collector_version": sensor.collector_version,
        }
        failed_record = validate_state_evidence(raw)
        assert failed_record is not None
        return failed_record


def _record_mapping(record: StateEvidence) -> dict[str, object]:
    """Expose fields for validation without recursively copying untrusted JSON."""
    return {
        "schema_version": record.schema_version,
        "tool_event_ref": record.tool_event_ref,
        "phase": record.phase,
        "resource_type": record.resource_type,
        "resource_id": record.resource_id,
        "status": record.status,
        "raw_fingerprint": record.raw_fingerprint,
        "canonical_fingerprint": record.canonical_fingerprint,
        "delta": record.delta,
        "collector_version": record.collector_version,
    }


def _unavailable_record(
    sensor: StateSensor, *, tool_event_ref: str, phase: str
) -> StateEvidence | None:
    return validate_state_evidence(
        {
            "schema_version": 1,
            "tool_event_ref": tool_event_ref,
            "phase": phase,
            "resource_type": sensor.resource_type,
            "resource_id": sensor.resource_id,
            "status": "unavailable",
            "raw_fingerprint": None,
            "canonical_fingerprint": None,
            "delta": None,
            "collector_version": sensor.collector_version,
        }
    )


def _required_unavailable_record(
    sensor: StateSensor, *, tool_event_ref: str, phase: str
) -> StateEvidence:
    record = _unavailable_record(sensor, tool_event_ref=tool_event_ref, phase=phase)
    assert record is not None  # identities are validated by StateEvidenceProducer.__init__
    return record


def _outcome(before: Sequence[StateEvidence], after: Sequence[StateEvidence]) -> StateOutcome:
    records = (*before, *after)
    if any(record.status == "collection_failed" for record in records):
        return StateOutcome.COLLECTION_FAILED
    if any(record.status == "unavailable" for record in records):
        return StateOutcome.UNAVAILABLE
    changed = any(
        prior.canonical_fingerprint != current.canonical_fingerprint
        for prior, current in zip(before, after, strict=True)
    )
    return StateOutcome.CHANGED if changed else StateOutcome.UNCHANGED


__all__ = [
    "STATE_EVIDENCE_METADATA_KEY",
    "FilesystemStateSensor",
    "PytestStateSensor",
    "StateCapture",
    "StateEvidenceEnvelope",
    "StateEvidenceProducer",
    "StateOutcome",
    "StateSensor",
    "state_evidence_from_event",
]
