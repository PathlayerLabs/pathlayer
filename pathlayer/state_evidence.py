"""Opt-in runtime capture and transport for ``StateEvidence`` (ARCH-2).

Collection happens at an adapter's tool boundary. The resulting closed-schema
records are embedded in ``ToolEvent.metadata`` so the pure pipeline, durable
store and trace replay can consume the observation without performing I/O.

STATE-1 adds a second way to choose *what* to observe. A producer built with a
fixed sensor list observes the same resources around every tool call; one built
with a :class:`StateSensorPlanner` asks the planner, at each boundary, which
sensors that call's own arguments make observable. A planner that selects
nothing means "this call claims no state": the event then carries no envelope
at all, exactly as it would with no producer configured. Planning is the only
new degree of freedom -- the capture, validation, transport and replay path
below is the one ARCH-2 already proved.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pathlayer.detectors.state_evidence import StateEvidence, validate_state_evidence
from pathlayer.events import ToolEvent
from pathlayer.state_collectors import (
    FILESYSTEM_COLLECTOR_VERSION,
    PYTEST_COLLECTOR_VERSION,
    collect_filesystem_evidence,
    collect_pytest_evidence,
)

logger = logging.getLogger(__name__)

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
    """One resource observed around a tool call.

    Fixed for the whole run when the producer was built with a sensor list, or
    chosen for this one call by a :class:`StateSensorPlanner`.
    """

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


class StateSensorPlanner(Protocol):
    """Choose the sensors one tool call's own arguments make observable.

    Called at the ``before`` boundary, inside the adapter's lock, with the tool
    name and the arguments the framework reported. It must be deterministic
    (the same call selects the same sensors), must never decide anything about
    the *outcome* of a call, and returns an empty sequence to abstain.
    """

    def plan(self, *, tool: str, args: Mapping[str, Any]) -> Sequence[StateSensor]: ...


@dataclass(frozen=True, slots=True)
class StateCapture:
    """Run-local before observations passed explicitly to the after boundary.

    ``sensors`` are the sensors this boundary actually planned, so the after
    capture re-observes exactly those resources even when the next tool call
    plans different ones.
    """

    tool_event_ref: str
    before: tuple[StateEvidence, ...]
    sensors: tuple[StateSensor, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class StateEvidenceEnvelope:
    """Validated state result transported on a ``ToolEvent``."""

    outcome: StateOutcome
    records: tuple[StateEvidence, ...]


class StateEvidenceProducer:
    """Bracket tool calls with deterministic sensors, fixed or planned per call.

    The producer owns no per-run mutable state. A ``StateCapture`` is returned
    to the adapter and stored beside that adapter's pending tool call, which
    prevents concurrent runs from sharing or confusing observations.

    Built with ``sensors``, every tool call observes those same resources.
    Built with a ``planner`` (STATE-1), each call observes what its own
    arguments make observable, and a call the planner abstains on carries no
    evidence at all. A planner is an integration boundary like a sensor: an
    invalid or raising plan leaves the call unobserved and is logged, never
    raised into the observed agent.
    """

    def __init__(
        self,
        sensors: Sequence[StateSensor] | None = None,
        *,
        planner: StateSensorPlanner | None = None,
    ) -> None:
        if (sensors is None) == (planner is None):
            raise ValueError("StateEvidenceProducer requires either sensors or a planner")
        self._planner = planner
        self._sensors: tuple[StateSensor, ...] = ()
        if sensors is not None:
            self._sensors = tuple(sensors)
            problem = _sensor_problem(self._sensors)
            if problem is not None:
                raise ValueError(problem)

    def capture_before(self, tool_event_ref: str) -> StateCapture:
        """Observe the fixed sensor set. Requires a producer built with sensors.

        :meth:`plan_capture` without the planning step; kept because it is the
        ARCH-2/VAL-5 entry point and says at the call site that this producer
        observes the same resources around every call.
        """
        return self._capture(tool_event_ref, self._fixed_sensors(), observable=True)

    def unavailable_capture(self, tool_event_ref: str) -> StateCapture:
        """Return an explicit abstention for an overlapping tool boundary."""
        return self._capture(tool_event_ref, self._fixed_sensors(), observable=False)

    def plan_capture(
        self,
        tool_event_ref: str,
        *,
        tool: str,
        args: Mapping[str, Any],
        observable: bool = True,
    ) -> StateCapture | None:
        """Observe what this call makes observable, or ``None`` if it claims nothing.

        The adapter's entry point, for both kinds of producer: on one built
        with a fixed sensor list ``tool`` and ``args`` are ignored, that list is
        always observed, and the result is never ``None``.

        ``observable=False`` records the planned resources as explicitly
        unavailable, for a boundary that overlaps another tool call.
        """
        sensors = self._plan(tool, args)
        if sensors is None:
            return None
        return self._capture(tool_event_ref, sensors, observable=observable)

    def unavailable_for(self, capture: StateCapture) -> StateCapture:
        """Withdraw an in-flight capture, keeping the resources it had planned."""
        return self._capture(capture.tool_event_ref, capture.sensors, observable=False)

    def attach_after(self, event: ToolEvent, capture: StateCapture) -> ToolEvent:
        if event.run_id != capture.tool_event_ref:
            raise ValueError("StateCapture belongs to a different ToolEvent")
        if len(capture.sensors) != len(capture.before):
            # Only a hand-built capture can get here: every producer method
            # records the sensors it observed alongside their observations. Say
            # so explicitly rather than failing inside the zip below.
            raise ValueError("StateCapture does not carry the sensors it observed")
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
            for sensor, prior in zip(capture.sensors, capture.before, strict=True)
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

    def _fixed_sensors(self) -> tuple[StateSensor, ...]:
        if not self._sensors:
            raise ValueError("this producer plans its sensors per tool call")
        return self._sensors

    def _plan(self, tool: str, args: Mapping[str, Any]) -> tuple[StateSensor, ...] | None:
        if self._planner is None:
            return self._sensors
        try:
            planned = tuple(self._planner.plan(tool=tool, args=args))
        except Exception:
            # A planner reads adapter-supplied tool arguments: untrusted, and
            # its failure must never reach the observed agent. Leaving the call
            # unobserved is the abstention the planner itself would have used.
            logger.warning(
                "state evidence planner raised; the tool call is left unobserved", exc_info=True
            )
            return None
        if not planned:
            return None
        problem = _sensor_problem(planned)
        if problem is not None:
            logger.warning(
                "state evidence planner returned an unusable plan (%s); "
                "the tool call is left unobserved",
                problem,
            )
            return None
        return planned

    @staticmethod
    def _capture(
        tool_event_ref: str, sensors: tuple[StateSensor, ...], *, observable: bool
    ) -> StateCapture:
        records = tuple(
            _capture_safely(sensor, tool_event_ref=tool_event_ref, phase="before")
            if observable
            else _required_unavailable_record(sensor, tool_event_ref=tool_event_ref, phase="before")
            for sensor in sensors
        )
        return StateCapture(tool_event_ref=tool_event_ref, before=records, sensors=sensors)


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


def _sensor_problem(sensors: tuple[StateSensor, ...]) -> str | None:
    """Why this sensor set cannot be observed, or ``None`` when it can.

    Shared by the fixed set (validated once, at construction, where a mistake
    is a configuration error) and every planned set (validated at each call,
    where a mistake must degrade to an abstention instead of raising).
    """
    if not sensors:
        return "StateEvidenceProducer requires at least one sensor"
    if len(sensors) > _MAX_SENSORS:
        return f"StateEvidenceProducer supports at most {_MAX_SENSORS} sensors"
    identities = [(sensor.resource_type, sensor.resource_id) for sensor in sensors]
    if len(set(identities)) != len(identities):
        return "state sensor resource identities must be unique"
    for sensor in sensors:
        if _unavailable_record(sensor, tool_event_ref="validation", phase="before") is None:
            return "state sensor identity or collector version is invalid"
    return None


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
    # Identities are validated at construction for a fixed sensor set, and per
    # call in StateEvidenceProducer._plan for a planned one -- see
    # _sensor_problem, which both paths share.
    assert record is not None
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
    "StateSensorPlanner",
    "state_evidence_from_event",
]
