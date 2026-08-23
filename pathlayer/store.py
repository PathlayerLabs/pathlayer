"""SQLite-backed, event-sourced state for resumable Pathlayer pipelines.

Storage is deliberately outside detectors and arbitration. The pure
``Pipeline`` remains the specification; ``PersistentPipeline`` journals each
successful tick and reconstructs a fresh pipeline by replaying that journal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeVar, cast

from pathlayer.detectors.base import Signal
from pathlayer.detectors.budget import BudgetState
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline, PipelineDecision
from pathlayer.policies.actions import Action
from pathlayer.policies.actuation import ActuationContext, EventSnapshot

SCHEMA_VERSION = 3
EVENT_CODEC_VERSION = 1
# Bump when unchanged configuration can change future decisions in a way full
# prefix replay cannot expose. Package/OBS-only changes deliberately do not.
STATE_SEMANTICS_VERSION = 1
# RES-2 deliberately does not bump this version: its changed semantics are
# fully visible in the newly persisted RUN_START/RUN_TIMEOUT event prefix. The
# config fingerprint rejects pre-RES-2 active-resource scopes, while disabled
# resource configs preserve their historical fingerprint exactly.
TERMINAL_STATUSES = frozenset({"completed", "interrupted", "stopped"})
_VALID_CLOSE_STATUSES = TERMINAL_STATUSES
_T = TypeVar("_T")


class StateStoreError(RuntimeError):
    """Base error for durable pipeline state failures."""


class StateCompatibilityError(StateStoreError):
    """The database or scope cannot be interpreted by this configuration."""


class StateCorruptionError(StateStoreError):
    """Durable rows are internally inconsistent or cannot be decoded."""


class StateConcurrencyError(StateStoreError):
    """Another writer advanced the same scope concurrently."""


class StateWriteError(StateStoreError):
    """SQLite became unavailable while accessing durable state."""


class StateClosedError(StateStoreError):
    """A terminal scope was opened for further processing."""


class StateNotFoundError(StateStoreError):
    """A requested durable scope does not exist."""


class StateIncompleteError(StateStoreError):
    """A known non-durable journal prefix cannot be resumed."""


class EventSerializationError(StateStoreError):
    """A ToolEvent cannot be represented without changing its semantics."""


CostCounters = BudgetState


@dataclass(slots=True, frozen=True)
class StoredSignal:
    """Immutable signal value read from the durable journal."""

    name: str
    confidence: float
    reason: str
    diagnostics: tuple[str, ...]

    def to_signal(self) -> Signal:
        return Signal(self.name, self.confidence, self.reason, self.diagnostics)


@dataclass(slots=True, frozen=True)
class StoredTick:
    """One complete, immutable persisted decision tick."""

    index: int
    ts: float
    event_id: str | None
    event_json: str
    fused: float
    evidence: Mapping[str, float]
    reason: str
    arbitrated_action: Action
    emitted_action: Action
    dry_run: bool
    signals: tuple[StoredSignal, ...]

    @property
    def event(self) -> ToolEvent:
        """Return a fresh event so callers cannot mutate the stored snapshot."""
        event = decode_event(self.event_json)
        if not _same_float(event.ts, self.ts):
            raise StateCorruptionError("stored tick timestamp disagrees with its event")
        return event


@dataclass(slots=True, frozen=True)
class JournalEntry:
    index: int
    event_id: str | None
    event: ToolEvent
    emitted_action: Action


@dataclass(slots=True, frozen=True)
class ScopeSnapshot:
    """One committed revision and whether its journal is known to be whole.

    ``durable`` is false after a successfully recorded poison marker.
    ``complete`` additionally requires a terminal status; an active journal is
    always a prefix of a run that may still continue.
    """

    state_key: str
    status: str
    revision: int
    ticks: tuple[StoredTick, ...]
    costs: CostCounters
    durable: bool = True

    @property
    def complete(self) -> bool:
        """Whether this is a terminal journal with no known missing tail."""
        return self.durable and self.status in TERMINAL_STATUSES

    @property
    def entries(self) -> tuple[JournalEntry, ...]:
        """Replay-compatible journal projection kept for existing callers."""
        return tuple(
            JournalEntry(tick.index, tick.event_id, tick.event, tick.emitted_action)
            for tick in self.ticks
        )


@dataclass(slots=True, frozen=True)
class AppendResult:
    revision: int
    index: int
    action: Action
    inserted: bool


class StateStore(Protocol):
    def load_or_create(self, state_key: str, fingerprint: str) -> ScopeSnapshot: ...

    def read_scope(self, state_key: str) -> ScopeSnapshot: ...

    def lookup_event(self, state_key: str, event_id: str) -> StoredTick | None: ...

    def append_tick(
        self,
        state_key: str,
        fingerprint: str,
        expected_revision: int,
        event_id: str | None,
        event: ToolEvent,
        decision: PipelineDecision,
        costs: BudgetState,
    ) -> AppendResult: ...

    def close_scope(
        self, state_key: str, fingerprint: str, expected_revision: int, status: str
    ) -> int: ...

    def mark_nondurable(self, state_key: str, fingerprint: str) -> None: ...

    def close(self) -> None: ...


def _encode_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"$pathlayer": "float", "value": "nan"}
        if math.isinf(value):
            return {"$pathlayer": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, list):
        return [_encode_value(item, f"{path}[]") for item in value]
    if isinstance(value, tuple):
        return {
            "$pathlayer": "tuple",
            "items": [_encode_value(item, f"{path}[]") for item in value],
        }
    if isinstance(value, Mapping):
        items: list[list[Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise EventSerializationError(f"{path} has a non-string mapping key")
            items.append([key, _encode_value(item, f"{path}.{key}")])
        return {"$pathlayer": "mapping", "items": items}
    raise EventSerializationError(
        f"{path} contains unsupported value of type {type(value).__name__}"
    )


def _decode_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list):
        return [_decode_value(item, f"{path}[]") for item in value]
    if not isinstance(value, dict):
        raise StateCorruptionError(f"invalid encoded value at {path}")
    tag = value.get("$pathlayer")
    if tag == "float":
        encoded = value.get("value")
        values = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
        if encoded not in values:
            raise StateCorruptionError(f"invalid encoded float at {path}")
        return values[cast(str, encoded)]
    if tag == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise StateCorruptionError(f"invalid encoded tuple at {path}")
        return tuple(_decode_value(item, f"{path}[]") for item in items)
    if tag == "mapping":
        items = value.get("items")
        if not isinstance(items, list):
            raise StateCorruptionError(f"invalid encoded mapping at {path}")
        decoded: dict[str, Any] = {}
        for pair in items:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or pair[0] in decoded
            ):
                raise StateCorruptionError(f"invalid encoded mapping entry at {path}")
            decoded[pair[0]] = _decode_value(pair[1], f"{path}.{pair[0]}")
        return decoded
    raise StateCorruptionError(f"unknown encoded value tag at {path}")


def encode_event(event: ToolEvent) -> str:
    """Encode a ToolEvent without silently stringifying arbitrary objects."""
    envelope = {
        "version": EVENT_CODEC_VERSION,
        "tool": event.tool,
        "args": _encode_value(event.args, "event.args"),
        "result": _encode_value(event.result, "event.result"),
        "cost_usd": _encode_value(event.cost_usd, "event.cost_usd"),
        "ts": _encode_value(event.ts, "event.ts"),
        "kind": event.kind.value,
        "run_id": event.run_id,
        "metadata": _encode_value(event.metadata, "event.metadata"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_event(payload: str) -> ToolEvent:
    """Decode and validate a versioned durable ToolEvent."""
    try:
        envelope = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StateCorruptionError("invalid event JSON") from exc
    if not isinstance(envelope, dict) or envelope.get("version") != EVENT_CODEC_VERSION:
        raise StateCompatibilityError("unsupported event codec version")
    try:
        kind = EventKind(envelope["kind"])
        tool = envelope["tool"]
        run_id = envelope["run_id"]
        args = _decode_value(envelope["args"], "event.args")
        result = _decode_value(envelope["result"], "event.result")
        cost = _decode_value(envelope["cost_usd"], "event.cost_usd")
        ts = _decode_value(envelope["ts"], "event.ts")
        metadata = _decode_value(envelope["metadata"], "event.metadata")
    except (KeyError, TypeError, ValueError) as exc:
        raise StateCorruptionError("invalid event envelope") from exc
    if (
        not isinstance(tool, str)
        or not isinstance(args, dict)
        or not isinstance(metadata, dict)
        or (run_id is not None and not isinstance(run_id, str))
        or isinstance(ts, bool)
        or not isinstance(ts, int | float)
    ):
        raise StateCorruptionError("invalid decoded ToolEvent fields")
    return ToolEvent(
        tool=tool,
        args=args,
        result=result,
        cost_usd=cast(float, cost),
        ts=float(ts),
        kind=kind,
        run_id=run_id,
        metadata=metadata,
    )


def _decode_json_value(payload: str, path: str) -> Any:
    try:
        encoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StateCorruptionError(f"invalid JSON at {path}") from exc
    return _decode_value(encoded, path)


def _encode_event_ts(value: Any, path: str = "event.ts") -> float | str:
    """Keep ordinary timestamps queryable while preserving non-finite values."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EventSerializationError(f"{path} must be a number")
    number = float(value)
    if math.isfinite(number):
        return number
    return json.dumps(_encode_value(number, path), separators=(",", ":"), allow_nan=False)


def _decode_event_ts(value: Any) -> float:
    if isinstance(value, bool):
        raise StateCorruptionError("invalid stored event timestamp")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        decoded = _decode_json_value(value, "event.ts")
        if isinstance(decoded, float) and not math.isfinite(decoded):
            return decoded
    raise StateCorruptionError("invalid stored event timestamp")


def _stored_integer(value: Any, field: str, *, minimum: int = 0) -> int:
    """Validate an integer read from SQLite without coercing corrupt values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StateCorruptionError(f"invalid stored {field}")
    return value


def _stored_action(value: Any, field: str) -> Action:
    """Validate an action name read from SQLite."""
    if not isinstance(value, str):
        raise StateCorruptionError(f"invalid stored {field}")
    try:
        return Action[value]
    except KeyError as exc:
        raise StateCorruptionError(f"invalid stored {field}") from exc


def _same_float(left: float, right: float) -> bool:
    return left == right or (math.isnan(left) and math.isnan(right))


def _fingerprint_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _fingerprint_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set | frozenset):
        return sorted((_fingerprint_value(item) for item in value), key=repr)
    if isinstance(value, list | tuple):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def configuration_fingerprint(config: Any) -> str:
    """Hash every configuration field that can influence a decision."""
    decision_config = {
        name: _fingerprint_value(getattr(config, name))
        for name in ("detectors", "canonicalization", "policy", "pipeline")
    }
    # RES-2 adds one decision-relevant field. When the resource detector is
    # disabled it cannot affect replay, so omit only that inactive field and
    # preserve the exact historical fingerprint of v3/default databases. When
    # enabled, retain it: a pre-RES-2 state is then rejected conservatively.
    detectors = decision_config.get("detectors")
    if isinstance(detectors, dict):
        resource = detectors.get("resource")
        if isinstance(resource, dict) and resource.get("enabled") is False:
            resource.pop("max_duration_seconds", None)
    canonical = json.dumps(
        {
            # Keep the same canonical shape as the first INF-1 draft while
            # explicitly excluding future observability/pricing-only sections.
            "config": decision_config,
            "event_codec": EVENT_CODEC_VERSION,
            "state_semantics": STATE_SEMANTICS_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SQLiteStateStore:
    """Transactional SQLite journal implementing :class:`StateStore`."""

    def __init__(self, path: str | Path, *, timeout: float = 5.0) -> None:
        raw_path = os.fspath(path)
        if raw_path != ":memory:":
            resolved = Path(raw_path)
            parent_existed = resolved.parent.exists()
            try:
                resolved.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            except OSError as exc:
                raise StateWriteError(
                    f"cannot create SQLite state directory {resolved.parent!s}: {exc}"
                ) from exc
            if not parent_existed:
                with suppress(OSError):
                    resolved.parent.chmod(0o700)
        self.path = raw_path
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(raw_path, timeout=timeout, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(f"PRAGMA busy_timeout = {max(0, int(timeout * 1000))}")
            if raw_path != ":memory:":
                self._enable_wal(timeout)
            self._migrate()
        except StateStoreError:
            with suppress(Exception):
                self._connection.close()
            raise
        except sqlite3.OperationalError as exc:
            with suppress(Exception):
                self._connection.close()
            raise StateWriteError(f"cannot open SQLite state at {raw_path!r}: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            with suppress(Exception):
                self._connection.close()
            raise StateCorruptionError(f"cannot open SQLite state at {raw_path!r}: {exc}") from exc
        self._protect_files()

    def _enable_wal(self, timeout: float) -> None:
        """Enable WAL despite SQLite's transient first-open schema race."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                row = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()
                mode = None if row is None else str(row[0]).lower()
                if mode != "wal":
                    raise StateWriteError(f"SQLite state requires WAL journal mode, got {mode!r}")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _migrate(self) -> None:
        with self._lock:
            try:
                version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
                if version > SCHEMA_VERSION:
                    raise StateCompatibilityError(
                        f"SQLite schema {version} is newer than supported schema {SCHEMA_VERSION}"
                    )
                if version == SCHEMA_VERSION:
                    return

                # Only an actual migration takes the write lock. Re-read the
                # version inside it because another process may have migrated
                # while this connection waited for BEGIN IMMEDIATE.
                self._connection.execute("BEGIN IMMEDIATE")
                version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
                if version > SCHEMA_VERSION:
                    raise StateCompatibilityError(
                        f"SQLite schema {version} is newer than supported schema {SCHEMA_VERSION}"
                    )
                if version == 0:
                    statements = (
                        """CREATE TABLE scopes (
                            state_key TEXT PRIMARY KEY,
                            fingerprint TEXT NOT NULL,
                            status TEXT NOT NULL CHECK(status IN
                                ('active','completed','interrupted','stopped')),
                            revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
                            last_event_index INTEGER NOT NULL DEFAULT -1
                        )""",
                        """CREATE TABLE events (
                            state_key TEXT NOT NULL,
                            event_index INTEGER NOT NULL CHECK(event_index >= 0),
                            event_id TEXT,
                            payload_json TEXT NOT NULL,
                            PRIMARY KEY(state_key, event_index),
                            FOREIGN KEY(state_key) REFERENCES scopes(state_key) ON DELETE CASCADE
                        )""",
                        """CREATE UNIQUE INDEX events_state_event_id
                            ON events(state_key, event_id) WHERE event_id IS NOT NULL""",
                        """CREATE TABLE ticks (
                            state_key TEXT NOT NULL,
                            event_index INTEGER NOT NULL,
                            fused REAL NOT NULL,
                            evidence_json TEXT NOT NULL,
                            arbitrated_action TEXT NOT NULL,
                            emitted_action TEXT NOT NULL,
                            dry_run INTEGER NOT NULL CHECK(dry_run IN (0,1)),
                            PRIMARY KEY(state_key, event_index),
                            FOREIGN KEY(state_key, event_index)
                                REFERENCES events(state_key, event_index) ON DELETE CASCADE
                        )""",
                        """CREATE TABLE signals (
                            state_key TEXT NOT NULL,
                            event_index INTEGER NOT NULL,
                            signal_index INTEGER NOT NULL,
                            name TEXT NOT NULL,
                            confidence_json TEXT NOT NULL,
                            reason TEXT NOT NULL,
                            diagnostics_json TEXT NOT NULL,
                            PRIMARY KEY(state_key, event_index, signal_index),
                            FOREIGN KEY(state_key, event_index)
                                REFERENCES events(state_key, event_index) ON DELETE CASCADE
                        )""",
                        """CREATE TABLE cost_counters (
                            state_key TEXT PRIMARY KEY,
                            captured_usd REAL NOT NULL DEFAULT 0,
                            estimated_usd REAL NOT NULL DEFAULT 0,
                            invalid_costs INTEGER NOT NULL DEFAULT 0,
                            effective_usd REAL NOT NULL DEFAULT 0,
                            FOREIGN KEY(state_key) REFERENCES scopes(state_key) ON DELETE CASCADE
                        )""",
                    )
                    for statement in statements:
                        self._connection.execute(statement)
                    self._connection.execute("PRAGMA user_version = 1")
                    version = 1
                if version == 1:
                    self._connection.execute(
                        "ALTER TABLE ticks ADD COLUMN action_reason TEXT NOT NULL DEFAULT ''"
                    )
                    self._connection.execute("PRAGMA user_version = 2")
                    version = 2
                if version == 2:
                    self._connection.execute(
                        "ALTER TABLE scopes ADD COLUMN durable INTEGER NOT NULL DEFAULT 1 "
                        "CHECK(durable IN (0,1))"
                    )
                    self._connection.execute("ALTER TABLE ticks ADD COLUMN event_ts REAL")
                    self._connection.execute(
                        "ALTER TABLE cost_counters ADD COLUMN priced_events "
                        "INTEGER NOT NULL DEFAULT 0 CHECK(priced_events >= 0)"
                    )

                    # A pre-v3 active scope may already have continued in
                    # memory after losing persistence. With no historical
                    # marker, treating it as resumable would be false certainty.
                    self._connection.execute(
                        "UPDATE scopes SET durable = 0 WHERE status = 'active'"
                    )
                    self._connection.execute(
                        "UPDATE ticks SET action_reason = '' WHERE arbitrated_action = 'NONE'"
                    )
                    event_rows = self._connection.execute(
                        "SELECT state_key, event_index, payload_json FROM events "
                        "ORDER BY state_key, event_index"
                    ).fetchall()
                    rows_by_scope: dict[str, list[sqlite3.Row]] = {}
                    for row in event_rows:
                        rows_by_scope.setdefault(row["state_key"], []).append(row)
                    for state_key, scope_rows in rows_by_scope.items():
                        backfill: list[tuple[float | str, int]] = []
                        priced_events = 0
                        try:
                            for row in scope_rows:
                                event = decode_event(row["payload_json"])
                                backfill.append((_encode_event_ts(event.ts), row["event_index"]))
                                raw_cost = event.cost_usd
                                if (
                                    isinstance(raw_cost, int | float)
                                    and not isinstance(raw_cost, bool)
                                    and math.isfinite(raw_cost)
                                    and raw_cost > 0.0
                                ):
                                    priced_events += 1
                        except StateStoreError:
                            self._connection.execute(
                                "UPDATE scopes SET durable = 0 WHERE state_key = ?",
                                (state_key,),
                            )
                            continue
                        for event_ts, event_index in backfill:
                            self._connection.execute(
                                "UPDATE ticks SET event_ts = ? "
                                "WHERE state_key = ? AND event_index = ?",
                                (event_ts, state_key, event_index),
                            )
                        self._connection.execute(
                            "UPDATE cost_counters SET priced_events = ? WHERE state_key = ?",
                            (priced_events, state_key),
                        )
                    self._connection.execute("PRAGMA user_version = 3")
                self._connection.commit()
            except StateStoreError:
                with suppress(sqlite3.Error):
                    self._connection.rollback()
                raise
            except sqlite3.DatabaseError as exc:
                with suppress(sqlite3.Error):
                    self._connection.rollback()
                raise StateWriteError(f"SQLite schema migration failed: {exc}") from exc

    def _protect_files(self) -> None:
        """Best-effort owner-only permissions for every SQLite payload file."""
        if self.path == ":memory:" or os.name == "nt":
            return
        for candidate in (self.path, f"{self.path}-wal", f"{self.path}-shm"):
            with suppress(OSError):
                Path(candidate).chmod(0o600)

    def _transaction(self, operation: Callable[[], _T]) -> _T:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                result = operation()
                self._connection.commit()
                self._protect_files()
                return result
            except StateStoreError:
                with suppress(sqlite3.Error):
                    self._connection.rollback()
                raise
            except sqlite3.DatabaseError as exc:
                with suppress(sqlite3.Error):
                    self._connection.rollback()
                raise StateWriteError(f"SQLite state transaction failed: {exc}") from exc
            except Exception:
                with suppress(sqlite3.Error):
                    self._connection.rollback()
                raise

    def load_or_create(self, state_key: str, fingerprint: str) -> ScopeSnapshot:
        if not state_key:
            raise ValueError("state_key must not be empty")

        def open_scope() -> None:
            row = self._connection.execute(
                "SELECT fingerprint, status, durable FROM scopes WHERE state_key = ?",
                (state_key,),
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO scopes(state_key, fingerprint, status) VALUES (?, ?, 'active')",
                    (state_key, fingerprint),
                )
                self._connection.execute(
                    "INSERT INTO cost_counters(state_key) VALUES (?)", (state_key,)
                )
                return
            if row["fingerprint"] != fingerprint:
                raise StateCompatibilityError(
                    f"state_key {state_key!r} was created with a different configuration"
                )
            if not bool(row["durable"]):
                raise StateIncompleteError(
                    f"state_key {state_key!r} is a known non-durable journal prefix"
                )
            # A stopped scope reconstructs as an inert STOP pipeline so a
            # duplicate delivery can return the recorded STOP. Explicitly
            # closed completed/interrupted runs are never reopened.
            if row["status"] in {"completed", "interrupted"}:
                raise StateClosedError(f"state_key {state_key!r} is terminal ({row['status']})")

        self._transaction(open_scope)
        return self._load_snapshot(state_key)

    def read_scope(self, state_key: str) -> ScopeSnapshot:
        """Read active or terminal history without opening it for mutation."""
        return self._load_snapshot(state_key)

    def _load_snapshot(self, state_key: str) -> ScopeSnapshot:
        with self._lock:
            try:
                self._connection.execute("BEGIN")
                scope = self._connection.execute(
                    "SELECT status, revision, last_event_index, durable "
                    "FROM scopes WHERE state_key = ?",
                    (state_key,),
                ).fetchone()
                if scope is None:
                    raise StateNotFoundError(f"missing scope {state_key!r}")
                rows = self._connection.execute(
                    """
                    SELECT e.event_index, e.event_id, e.payload_json,
                           t.event_ts, t.fused, t.evidence_json, t.action_reason,
                           t.arbitrated_action, t.emitted_action, t.dry_run
                    FROM events e
                    JOIN ticks t USING(state_key, event_index)
                    WHERE e.state_key = ?
                    ORDER BY e.event_index
                    """,
                    (state_key,),
                ).fetchall()
                signal_rows = self._connection.execute(
                    """SELECT event_index, signal_index, name, confidence_json,
                              reason, diagnostics_json
                       FROM signals WHERE state_key = ?
                       ORDER BY event_index, signal_index""",
                    (state_key,),
                ).fetchall()
                cost_row = self._connection.execute(
                    """SELECT captured_usd, priced_events, estimated_usd,
                              invalid_costs, effective_usd
                       FROM cost_counters WHERE state_key = ?""",
                    (state_key,),
                ).fetchone()
                self._connection.commit()
                self._protect_files()
            except StateStoreError:
                with suppress(sqlite3.Error):
                    self._connection.rollback()
                raise
            except sqlite3.DatabaseError as exc:
                with suppress(sqlite3.Error):
                    self._connection.rollback()
                raise StateCorruptionError(f"cannot read state_key {state_key!r}: {exc}") from exc
        signals_by_tick: dict[int, list[sqlite3.Row]] = {}
        for row in signal_rows:
            event_index = _stored_integer(row["event_index"], "signal event index")
            signals_by_tick.setdefault(event_index, []).append(row)
        ticks: list[StoredTick] = []
        for expected, row in enumerate(rows):
            if row["event_index"] != expected:
                raise StateCorruptionError(f"non-contiguous event journal for {state_key!r}")
            ticks.append(self._decode_tick(row, signals_by_tick.get(expected, [])))
        revision = _stored_integer(scope["revision"], "scope revision")
        last_event_index = _stored_integer(
            scope["last_event_index"], "scope last event index", minimum=-1
        )
        if scope["status"] not in {"active", *TERMINAL_STATUSES}:
            raise StateCorruptionError(f"invalid status for state_key {state_key!r}")
        if scope["durable"] not in (0, 1):
            raise StateCorruptionError(f"invalid durability for state_key {state_key!r}")
        if revision != len(ticks):
            raise StateCorruptionError(f"revision mismatch for state_key {state_key!r}")
        if last_event_index != len(ticks) - 1:
            raise StateCorruptionError(f"event index mismatch for state_key {state_key!r}")
        if cost_row is None:
            raise StateCorruptionError(f"cost counter mismatch for state_key {state_key!r}")
        try:
            costs = CostCounters(
                captured_usd=float(cost_row["captured_usd"]),
                priced_events=_stored_integer(cost_row["priced_events"], "priced event count"),
                estimated_usd=float(cost_row["estimated_usd"]),
                invalid_costs=_stored_integer(cost_row["invalid_costs"], "invalid cost count"),
            )
            effective_usd = float(cost_row["effective_usd"])
        except StateCorruptionError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise StateCorruptionError(f"invalid cost counters for {state_key!r}") from exc
        if (
            not math.isfinite(costs.captured_usd)
            or costs.captured_usd < 0.0
            or costs.priced_events < 0
            or not math.isfinite(costs.estimated_usd)
            or costs.estimated_usd < 0.0
            or costs.invalid_costs < 0
            or not math.isfinite(effective_usd)
            or not math.isclose(costs.effective_usd, effective_usd)
        ):
            raise StateCorruptionError(f"effective cost mismatch for state_key {state_key!r}")
        return ScopeSnapshot(
            state_key,
            scope["status"],
            revision,
            tuple(ticks),
            costs,
            bool(scope["durable"]),
        )

    @staticmethod
    def _decode_tick(row: sqlite3.Row, signal_rows: list[sqlite3.Row]) -> StoredTick:
        try:
            arbitrated = _stored_action(row["arbitrated_action"], "arbitrated action")
            emitted = _stored_action(row["emitted_action"], "emitted action")
            evidence_raw = json.loads(row["evidence_json"])
            index = _stored_integer(row["event_index"], "tick event index")
            ts = _decode_event_ts(row["event_ts"])
            fused = float(row["fused"])
        except StateCorruptionError:
            raise
        except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
            raise StateCorruptionError("invalid stored tick") from exc
        if not isinstance(evidence_raw, dict) or not all(
            isinstance(key, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for key, value in evidence_raw.items()
        ):
            raise StateCorruptionError("invalid stored evidence")
        signals: list[StoredSignal] = []
        for expected, signal_row in enumerate(signal_rows):
            signal_index = _stored_integer(signal_row["signal_index"], "signal index")
            if signal_index != expected:
                raise StateCorruptionError("non-contiguous stored signals")
            confidence = _decode_json_value(signal_row["confidence_json"], "signal.confidence")
            try:
                diagnostics = json.loads(signal_row["diagnostics_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise StateCorruptionError("invalid stored signal diagnostics") from exc
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not isinstance(signal_row["name"], str)
                or not isinstance(signal_row["reason"], str)
                or not isinstance(diagnostics, list)
                or not all(isinstance(item, str) for item in diagnostics)
            ):
                raise StateCorruptionError("invalid stored signal")
            signals.append(
                StoredSignal(
                    signal_row["name"],
                    float(confidence),
                    signal_row["reason"],
                    tuple(diagnostics),
                )
            )
        reason = row["action_reason"]
        event_id = row["event_id"]
        payload = row["payload_json"]
        dry_run = row["dry_run"]
        if (
            not isinstance(reason, str)
            or (event_id is not None and not isinstance(event_id, str))
            or not isinstance(payload, str)
            or index < 0
            or not math.isfinite(fused)
            or not 0.0 <= fused <= 1.0
            or dry_run not in (0, 1)
        ):
            raise StateCorruptionError("invalid stored action reason")
        return StoredTick(
            index=index,
            ts=ts,
            event_id=event_id,
            event_json=payload,
            fused=fused,
            evidence=MappingProxyType({key: float(value) for key, value in evidence_raw.items()}),
            reason=reason,
            arbitrated_action=arbitrated,
            emitted_action=emitted,
            dry_run=bool(dry_run),
            signals=tuple(signals),
        )

    def lookup_event(self, state_key: str, event_id: str) -> StoredTick | None:
        try:
            with self._lock:
                row = self._connection.execute(
                    """
                    SELECT e.event_index, e.event_id, e.payload_json,
                           t.event_ts, t.fused, t.evidence_json, t.action_reason,
                           t.arbitrated_action, t.emitted_action, t.dry_run
                    FROM events e JOIN ticks t USING(state_key, event_index)
                    WHERE e.state_key = ? AND e.event_id = ?
                    """,
                    (state_key, event_id),
                ).fetchone()
                signal_rows = (
                    []
                    if row is None
                    else self._connection.execute(
                        """SELECT event_index, signal_index, name, confidence_json,
                                  reason, diagnostics_json
                           FROM signals WHERE state_key=? AND event_index=?
                           ORDER BY signal_index""",
                        (state_key, row["event_index"]),
                    ).fetchall()
                )
        except (sqlite3.OperationalError, sqlite3.ProgrammingError) as exc:
            raise StateWriteError(f"SQLite event lookup failed: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            raise StateCorruptionError(f"SQLite event lookup is corrupt: {exc}") from exc
        if row is None:
            return None
        return self._decode_tick(row, signal_rows)

    def append_tick(
        self,
        state_key: str,
        fingerprint: str,
        expected_revision: int,
        event_id: str | None,
        event: ToolEvent,
        decision: PipelineDecision,
        costs: BudgetState,
    ) -> AppendResult:
        payload = encode_event(event)
        try:
            _encode_event_ts(event.ts)
            event_ts = _encode_event_ts(decision.ts, "decision.ts")
            if not _same_float(float(event.ts), decision.ts):
                raise ValueError("decision timestamp must match its event")
            if (
                isinstance(decision.fused, bool)
                or not isinstance(decision.fused, int | float)
                or not math.isfinite(decision.fused)
                or not 0.0 <= decision.fused <= 1.0
            ):
                raise ValueError("decision fused score must be finite and in [0, 1]")
            evidence = json.dumps(
                dict(decision.evidence), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            if not isinstance(decision.reason, str):
                raise TypeError("decision reason must be a string")
            if decision.arbitrated_action is Action.NONE and decision.reason:
                raise ValueError("a NONE decision must have an empty action reason")
            if (
                not isinstance(costs, BudgetState)
                or isinstance(costs.captured_usd, bool)
                or not isinstance(costs.captured_usd, int | float)
                or not math.isfinite(costs.captured_usd)
                or costs.captured_usd < 0.0
                or isinstance(costs.priced_events, bool)
                or not isinstance(costs.priced_events, int)
                or costs.priced_events < 0
                or isinstance(costs.estimated_usd, bool)
                or not isinstance(costs.estimated_usd, int | float)
                or not math.isfinite(costs.estimated_usd)
                or costs.estimated_usd < 0.0
                or isinstance(costs.invalid_costs, bool)
                or not isinstance(costs.invalid_costs, int)
                or costs.invalid_costs < 0
            ):
                raise ValueError("cost counters must be finite and non-negative")
            signal_payloads: list[tuple[str, str, str, str]] = []
            for signal in decision.signals:
                if (
                    not isinstance(signal.name, str)
                    or not isinstance(signal.reason, str)
                    or not isinstance(signal.diagnostics, tuple)
                    or not all(isinstance(item, str) for item in signal.diagnostics)
                ):
                    raise TypeError("signal text fields must be strings")
                signal_payloads.append(
                    (
                        signal.name,
                        json.dumps(
                            _encode_value(signal.confidence, "signal.confidence"),
                            allow_nan=False,
                        ),
                        signal.reason,
                        json.dumps(list(signal.diagnostics), allow_nan=False),
                    )
                )
        except (TypeError, ValueError) as exc:
            raise EventSerializationError(f"decision cannot be serialized: {exc}") from exc

        def append() -> AppendResult:
            scope = self._connection.execute(
                "SELECT fingerprint, status, revision, last_event_index, durable "
                "FROM scopes WHERE state_key=?",
                (state_key,),
            ).fetchone()
            if scope is None:
                raise StateCorruptionError(f"missing scope {state_key!r}")
            if scope["fingerprint"] != fingerprint:
                raise StateCompatibilityError("configuration fingerprint changed")
            if not bool(scope["durable"]):
                raise StateIncompleteError(
                    f"state_key {state_key!r} is a known non-durable journal prefix"
                )
            if scope["status"] != "active":
                raise StateClosedError(f"state_key {state_key!r} is terminal ({scope['status']})")
            revision = _stored_integer(scope["revision"], "scope revision")
            last_event_index = _stored_integer(
                scope["last_event_index"], "scope last event index", minimum=-1
            )
            if event_id is not None:
                duplicate = self._connection.execute(
                    """
                    SELECT e.event_index, e.payload_json, t.emitted_action
                    FROM events e JOIN ticks t USING(state_key, event_index)
                    WHERE e.state_key=? AND e.event_id=?
                    """,
                    (state_key, event_id),
                ).fetchone()
                if duplicate is not None:
                    if duplicate["payload_json"] != payload:
                        raise StateConcurrencyError(
                            f"event_id {event_id!r} was reused with a different event"
                        )
                    return AppendResult(
                        revision,
                        _stored_integer(duplicate["event_index"], "duplicate event index"),
                        _stored_action(duplicate["emitted_action"], "duplicate emitted action"),
                        False,
                    )
            if revision != expected_revision:
                raise StateConcurrencyError(
                    f"state_key {state_key!r} advanced from revision "
                    f"{expected_revision} to {scope['revision']}"
                )
            index = last_event_index + 1
            if decision.index != index:
                raise StateCorruptionError(
                    f"pipeline index {decision.index} does not match durable index {index}"
                )
            self._connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?)",
                (state_key, index, event_id, payload),
            )
            self._connection.execute(
                """INSERT INTO ticks(
                       state_key, event_index, event_ts, fused, evidence_json,
                       arbitrated_action, emitted_action, dry_run, action_reason
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    state_key,
                    index,
                    event_ts,
                    decision.fused,
                    evidence,
                    decision.arbitrated_action.name,
                    decision.emitted_action.name,
                    int(decision.dry_run),
                    decision.reason,
                ),
            )
            for signal_index, (name, confidence, reason, diagnostics) in enumerate(signal_payloads):
                self._connection.execute(
                    "INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        state_key,
                        index,
                        signal_index,
                        name,
                        confidence,
                        reason,
                        diagnostics,
                    ),
                )
            updated = self._connection.execute(
                """
                UPDATE cost_counters SET
                    captured_usd = captured_usd + ?,
                    priced_events = priced_events + ?,
                    estimated_usd = estimated_usd + ?,
                    invalid_costs = invalid_costs + ?,
                    effective_usd = effective_usd + ?
                WHERE state_key = ?
                """,
                (
                    costs.captured_usd,
                    costs.priced_events,
                    costs.estimated_usd,
                    costs.invalid_costs,
                    costs.effective_usd,
                    state_key,
                ),
            )
            if updated.rowcount != 1:
                raise StateCorruptionError(f"missing cost counters for state_key {state_key!r}")
            status = "stopped" if decision.emitted_action is Action.STOP else "active"
            new_revision = expected_revision + 1
            self._connection.execute(
                "UPDATE scopes SET revision=?, last_event_index=?, status=? WHERE state_key=?",
                (new_revision, index, status, state_key),
            )
            return AppendResult(new_revision, index, decision.emitted_action, True)

        return self._transaction(append)

    def mark_nondurable(self, state_key: str, fingerprint: str) -> None:
        """Poison a journal prefix so no later process can resume it."""

        def mark() -> None:
            row = self._connection.execute(
                "SELECT fingerprint FROM scopes WHERE state_key = ?", (state_key,)
            ).fetchone()
            if row is None:
                raise StateNotFoundError(f"missing scope {state_key!r}")
            if row["fingerprint"] != fingerprint:
                raise StateCompatibilityError("configuration fingerprint changed")
            self._connection.execute(
                "UPDATE scopes SET durable = 0 WHERE state_key = ?", (state_key,)
            )

        self._transaction(mark)

    def close_scope(
        self, state_key: str, fingerprint: str, expected_revision: int, status: str
    ) -> int:
        if status not in _VALID_CLOSE_STATUSES:
            raise ValueError(f"invalid terminal status {status!r}")

        def close() -> int:
            row = self._connection.execute(
                "SELECT fingerprint, status, revision, durable FROM scopes WHERE state_key=?",
                (state_key,),
            ).fetchone()
            if row is None:
                raise StateCorruptionError(f"missing scope {state_key!r}")
            if row["fingerprint"] != fingerprint:
                raise StateCompatibilityError("configuration fingerprint changed")
            if not bool(row["durable"]):
                raise StateIncompleteError(
                    f"state_key {state_key!r} is a known non-durable journal prefix"
                )
            revision = _stored_integer(row["revision"], "scope revision")
            if revision != expected_revision:
                raise StateConcurrencyError(f"state_key {state_key!r} advanced concurrently")
            if row["status"] in TERMINAL_STATUSES:
                if row["status"] == status:
                    return expected_revision
                raise StateClosedError(f"state_key {state_key!r} is already {row['status']}")
            self._connection.execute(
                "UPDATE scopes SET status=? WHERE state_key=?", (status, state_key)
            )
            return expected_revision

        return self._transaction(close)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class PersistentPipeline:
    """I/O boundary that makes a pure Pipeline durable and resumable."""

    def __init__(
        self,
        pipeline_factory: Callable[[], Pipeline],
        store: StateStore,
        state_key: str,
        fingerprint: str,
        *,
        cost_resolver: Callable[[ToolEvent], BudgetState] | None = None,
        concurrency_retries: int = 2,
        owns_store: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self._factory = pipeline_factory
        self._store = store
        self._state_key = state_key
        self._fingerprint = fingerprint
        self._cost_resolver = cost_resolver or (lambda _event: BudgetState())
        self._concurrency_retries = concurrency_retries
        self._owns_store = owns_store
        self._logger = logger or logging.getLogger(__name__)
        self._process_lock = threading.RLock()
        self._closed = False
        self._durable = True
        self._persistence_error: StateStoreError | None = None
        self._duplicate_tick: StoredTick | None = None
        self._pipeline: Pipeline
        self._revision = 0
        self._restore()

    def _restore(self) -> None:
        snapshot = self._store.load_or_create(self._state_key, self._fingerprint)
        pipeline = self._factory()
        decisions = pipeline.restore_decisions(tick.event for tick in snapshot.ticks)
        if len(decisions) != len(snapshot.ticks):
            raise StateCorruptionError("durable journal contains events after STOP")
        for tick, decision in zip(snapshot.ticks, decisions, strict=True):
            divergence = self._decision_divergence(tick, decision)
            if divergence is not None:
                raise StateCompatibilityError(
                    f"replay diverged at event {tick.index}: {divergence}"
                )
        self._pipeline = pipeline
        self._revision = snapshot.revision
        self._duplicate_tick = None

    def _degrade(self, error: StateStoreError) -> None:
        """Permanently make this instance memory-only after a runtime outage."""
        if not self._durable:
            return
        # The original failure remains the useful diagnosis. A store that
        # cannot record its own outage necessarily leaves only an active
        # prefix; integration metadata (OpenClaw's manifest) is the
        # complementary poison marker in that case.
        with suppress(StateStoreError):
            self._store.mark_nondurable(self._state_key, self._fingerprint)
        self._durable = False
        self._persistence_error = error
        self._logger.error(
            "Pathlayer persistence failed; this run is now memory-only",
            exc_info=(type(error), error, error.__traceback__),
        )

    @staticmethod
    def _decision_from_tick(tick: StoredTick) -> PipelineDecision:
        return PipelineDecision(
            index=tick.index,
            ts=tick.ts,
            signals=tuple(signal.to_signal() for signal in tick.signals),
            evidence=tick.evidence,
            fused=tick.fused,
            reason=tick.reason,
            arbitrated_action=tick.arbitrated_action,
            emitted_action=tick.emitted_action,
            dry_run=tick.dry_run,
        )

    @staticmethod
    def _decision_divergence(tick: StoredTick, decision: PipelineDecision) -> str | None:
        if tick.index != decision.index:
            return f"index stored {tick.index}, replayed {decision.index}"
        if not _same_float(tick.ts, decision.ts):
            return f"timestamp stored {tick.ts!r}, replayed {decision.ts!r}"
        if dict(tick.evidence) != dict(decision.evidence):
            return "progress evidence changed"
        if not _same_float(tick.fused, decision.fused):
            return f"fused score stored {tick.fused!r}, replayed {decision.fused!r}"
        if tick.reason != decision.reason:
            return f"action reason stored {tick.reason!r}, replayed {decision.reason!r}"
        if tick.arbitrated_action is not decision.arbitrated_action:
            return (
                f"arbitrated action stored {tick.arbitrated_action.name}, "
                f"replayed {decision.arbitrated_action.name}"
            )
        if tick.emitted_action is not decision.emitted_action:
            return (
                f"emitted action stored {tick.emitted_action.name}, "
                f"replayed {decision.emitted_action.name}"
            )
        if tick.dry_run is not decision.dry_run:
            return f"dry-run stored {tick.dry_run}, replayed {decision.dry_run}"
        if len(tick.signals) != len(decision.signals):
            return f"signal count stored {len(tick.signals)}, replayed {len(decision.signals)}"
        for index, (stored, replayed) in enumerate(
            zip(tick.signals, decision.signals, strict=True)
        ):
            if stored.name != replayed.name:
                return f"signal {index} name stored {stored.name!r}, replayed {replayed.name!r}"
            replayed_confidence = replayed.confidence
            if isinstance(replayed_confidence, bool) or not isinstance(
                replayed_confidence, int | float
            ):
                return f"signal {index} confidence type changed"
            if not _same_float(stored.confidence, float(replayed_confidence)):
                return f"signal {index} confidence changed"
            if stored.reason != replayed.reason:
                return f"signal {index} reason changed"
            if stored.diagnostics != replayed.diagnostics:
                return f"signal {index} diagnostics changed"
        return None

    def _return_duplicate(self, tick: StoredTick, event: ToolEvent) -> Action:
        if encode_event(tick.event) != encode_event(event):
            raise StateConcurrencyError(
                f"event_id {tick.event_id!r} was reused with a different event"
            )
        self._duplicate_tick = tick
        return tick.emitted_action

    def process(self, event: ToolEvent, *, event_id: str | None = None) -> Action:
        with self._process_lock:
            if self._closed:
                raise StateClosedError(f"state_key {self._state_key!r} is closed")
            self._duplicate_tick = None
            if not self._durable:
                return self._pipeline.process(event, event_id=event_id)

            try:
                encode_event(event)
            except EventSerializationError as error:
                self._degrade(error)
                return self._pipeline.process(event, event_id=event_id)

            if event_id is not None:
                try:
                    duplicate = self._store.lookup_event(self._state_key, event_id)
                    if duplicate is not None:
                        return self._return_duplicate(duplicate, event)
                except StateConcurrencyError:
                    raise
                except StateStoreError as error:
                    self._degrade(error)
                    return self._pipeline.process(event, event_id=event_id)
            if self._pipeline.stopped:
                return Action.STOP

            for attempt in range(self._concurrency_retries + 1):
                action = self._pipeline.process(event, event_id=event_id)
                decision = self._pipeline.last_decision
                if decision is None:
                    return action
                try:
                    costs = self._cost_resolver(event)
                    if not isinstance(costs, BudgetState):
                        raise TypeError("cost_resolver must return BudgetState")
                except StateStoreError as error:
                    self._degrade(error)
                    return action
                except Exception as exc:
                    wrapped_error = StateWriteError("persistent cost accounting failed")
                    wrapped_error.__cause__ = exc
                    self._degrade(wrapped_error.with_traceback(exc.__traceback__))
                    return action
                try:
                    result = self._store.append_tick(
                        self._state_key,
                        self._fingerprint,
                        self._revision,
                        event_id,
                        event,
                        decision,
                        costs,
                    )
                except StateConcurrencyError as error:
                    if attempt >= self._concurrency_retries:
                        self._degrade(error)
                        return action
                    try:
                        self._restore()
                        if event_id is not None:
                            duplicate = self._store.lookup_event(self._state_key, event_id)
                            if duplicate is not None:
                                return self._return_duplicate(duplicate, event)
                    except StateConcurrencyError:
                        raise
                    except StateStoreError as restore_error:
                        self._degrade(restore_error)
                        return action
                    if self._pipeline.stopped:
                        return Action.STOP
                    if event.kind is EventKind.RUN_START and any(
                        stored.kind is EventKind.RUN_START for stored in self._pipeline.history
                    ):
                        # RES-2 lifecycle start is idempotent per scope. A
                        # concurrent writer already established the durable
                        # anchor; do not append this racer's timestamp as a
                        # second start event. The runtime sink re-reads history
                        # after this return and schedules from the winner.
                        restored = self._pipeline.last_decision
                        return restored.emitted_action if restored is not None else Action.NONE
                    continue
                except StateClosedError as error:
                    # Another writer may have committed the terminal STOP for
                    # this same active scope. That is normal convergence, not
                    # a persistence outage: restore the canonical STOP and
                    # keep its journal durable. Completed/interrupted scopes
                    # still fail restoration and follow the fail-open poison
                    # path below.
                    try:
                        self._restore()
                    except StateStoreError:
                        self._degrade(error)
                        return action
                    if self._pipeline.stopped:
                        return Action.STOP
                    self._degrade(error)
                    return action
                except StateStoreError as error:
                    self._degrade(error)
                    return action
                if not result.inserted:
                    try:
                        self._restore()
                        if event_id is not None:
                            duplicate = self._store.lookup_event(self._state_key, event_id)
                            if duplicate is not None:
                                return self._return_duplicate(duplicate, event)
                    except StateConcurrencyError:
                        raise
                    except StateStoreError as error:
                        self._degrade(error)
                    return result.action
                self._revision = result.revision
                return action
            raise AssertionError("unreachable concurrency retry state")

    def close(self, status: str = "completed") -> None:
        if status not in _VALID_CLOSE_STATUSES:
            raise ValueError(f"invalid terminal status {status!r}")
        with self._process_lock:
            if self._closed:
                return
            resolved_status = "stopped" if self.stopped else status
            if self._durable:
                try:
                    self._store.close_scope(
                        self._state_key,
                        self._fingerprint,
                        self._revision,
                        resolved_status,
                    )
                except StateStoreError as error:
                    self._degrade(error)
            self._closed = True
            if self._owns_store:
                with suppress(Exception):
                    self._store.close()

    @property
    def state_key(self) -> str:
        return self._state_key

    @property
    def durable(self) -> bool:
        return self._durable

    @property
    def persistence_error(self) -> StateStoreError | None:
        return self._persistence_error

    def abandon_persistence(self, error: StateStoreError) -> None:
        """Poison this scope after an integration detects journal divergence."""
        with self._process_lock:
            self._degrade(error)

    @property
    def last_decision(self) -> PipelineDecision | None:
        if self._duplicate_tick is not None:
            return self._decision_from_tick(self._duplicate_tick)
        return self._pipeline.last_decision

    @property
    def actuation_context(self) -> ActuationContext | None:
        tick = self._duplicate_tick
        if tick is None:
            return self._pipeline.actuation_context
        if tick.emitted_action is Action.NONE or tick.dry_run:
            return None
        event = tick.event
        return ActuationContext(
            action=tick.emitted_action,
            fused=tick.fused,
            reason=tick.reason,
            signals=tuple(signal.to_signal() for signal in tick.signals),
            event=EventSnapshot(
                tool=event.tool,
                kind=event.kind,
                ts=event.ts,
                index=tick.index,
                cost_usd=event.cost_usd,
            ),
            post_mortem=self._pipeline.post_mortem if tick.emitted_action is Action.STOP else None,
        )

    def run(self, events: Iterable[ToolEvent]) -> list[Action]:
        actions: list[Action] = []
        for event in events:
            actions.append(self.process(event))
            if self.stopped:
                break
        return actions

    def sink(self, event: ToolEvent) -> None:
        self.process(event)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)
