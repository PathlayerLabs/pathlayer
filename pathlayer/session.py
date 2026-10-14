"""Append-only session observations outside the pure detector/policy layer.

Each input identity has one immutable enriched event. SQLite serializes writers;
retrying a captured event returns the same context even after other runs arrive.
Only the bounded trailing comparable window is read on the live path.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Iterable
from contextlib import closing, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from pathlayer.detectors.canonical import ToolMasks, canonicalize
from pathlayer.detectors.session import (
    MAX_CONTEXT_BYTES,
    MAX_SESSION_EVENTS,
    SESSION_METADATA_KEY,
    SESSION_VERSION,
    continuity_key,
    session_history,
    without_context,
)
from pathlayer.events import EventKind, ToolEvent
from pathlayer.store import _canonical_sha256, decode_event, encode_event

logger = logging.getLogger(__name__)


def _identity(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("session identities must be non-empty strings of at most 512 characters")
    return value


class SessionJournal:
    """Durable observation context; never holds run policy, budgets, or STOP.

    Use a fresh caller-issued session key after explicit closure. An evidence
    gap or different action starts a new incident automatically, not a new run.
    A missed observation is journaled as a gap row: no window crosses it, and
    continuity resumes with the observations captured after it.
    """

    def __init__(
        self, path: str | Path, *, fingerprint: str, tool_masks: ToolMasks | None = None
    ) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.tool_masks = tool_masks
        self._warned: set[tuple[str, str]] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_scopes (
                    source TEXT NOT NULL, session_key TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, closed INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(source, session_key)
                );
                CREATE TABLE IF NOT EXISTS session_observations (
                    idx INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL, session_key TEXT NOT NULL,
                    run_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    event TEXT NOT NULL, context TEXT NOT NULL,
                    -- NULL for an observation; the reason for a gap row.
                    gap TEXT,
                    UNIQUE(source, session_key, run_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS session_order
                    ON session_observations(source, session_key, idx);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def _gap_prefix(self, source: str, session_key: str) -> str:
        return f"{self.path.name}.{_canonical_sha256([source, session_key])}"

    def _gap_markers(self, source: str, session_key: str) -> list[Path]:
        # Plain prefix matching, not glob: a store path containing glob
        # metacharacters must never hide a marker (that would bridge a gap).
        prefix = f"{self._gap_prefix(source, session_key)}."
        return sorted(
            entry
            for entry in self.path.parent.iterdir()
            if entry.name.startswith(prefix) and entry.name.endswith(".gap")
        )

    @staticmethod
    def _insert_gap(db: sqlite3.Connection, source: str, session_key: str, reason: str) -> None:
        # Deterministic, never-colliding identity: real run ids are non-empty.
        (count,) = db.execute(
            "SELECT COUNT(*) FROM session_observations "
            "WHERE source=? AND session_key=? AND gap IS NOT NULL",
            (source, session_key),
        ).fetchone()
        db.execute(
            "INSERT INTO session_observations"
            "(source, session_key, run_id, event_id, event, context, gap) "
            "VALUES (?, ?, '', ?, '', ?, ?)",
            (source, session_key, f"gap:{count + 1}", json.dumps({"observations": []}), reason),
        )

    def mark_gap(self, *, source: str, session_key: str, reason: str) -> None:
        """Break the window here: nothing observed before joins anything after.

        Journaled at once when the database is usable. Otherwise an fsynced
        marker survives the outage and process restarts, and the next capture
        for the scope journals the gap first. If not even the marker can be
        persisted, propagate: claiming restart-safe continuity would otherwise
        be dishonest.
        """
        try:
            with closing(self._connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                self._insert_gap(db, source, session_key, reason)
            return
        except Exception:
            pass
        # One file per missed observation: a capture journaling earlier markers
        # removes only the files it listed, never one written concurrently.
        marker = self.path.with_name(
            f"{self._gap_prefix(source, session_key)}.{uuid.uuid4().hex}.gap"
        )
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(reason)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            fd = os.open(marker.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def capture(self, event: ToolEvent, **identity: Any) -> ToolEvent:
        """Contain a failed append without ever bridging the missing observation.

        The failure becomes a gap (:meth:`mark_gap`): the continuity window
        restarts after it instead of the session being disabled for good.
        """
        source, session_key = identity["source"], identity["session_key"]
        try:
            enriched = self.enrich(event, **identity)
        except Exception:
            # Includes codec/validation failures, not just SQLite outages: any
            # omitted observation makes inference across it unsafe.
            self.mark_gap(source=source, session_key=session_key, reason="capture_failed")
            if (source, session_key) not in self._warned:
                # Once per failure streak, not once per tool call.
                self._warned.add((source, session_key))
                logger.warning("session capture failed; continuity window broken", exc_info=True)
            return self._attach(
                without_context(event),
                {
                    "version": SESSION_VERSION,
                    "source": source,
                    "session_key": session_key,
                    "run_id": identity["run_id"],
                    "observations": [],
                    "unavailable": True,
                },
            )
        self._warned.discard((source, session_key))
        return enriched

    def _journal_pending_gaps(self, source: str, session_key: str) -> None:
        """Journal marker gaps before the next observation can be appended.

        Markers are listed inside the write lock and only those are removed
        after the commit, so a marker written concurrently is never lost. A
        crash before the removal only journals one more gap, never a bridge.
        """
        if not self._gap_markers(source, session_key):
            return
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            markers = self._gap_markers(source, session_key)
            if markers:
                self._insert_gap(db, source, session_key, "missed_observation")
        for marker in markers:
            with suppress(OSError):
                marker.unlink()

    def enrich(
        self,
        event: ToolEvent,
        *,
        source: str,
        session_key: str,
        run_id: str,
        event_id: str,
        linked: bool = True,
    ) -> ToolEvent:
        """Append once and return a copy carrying replayable detector context.

        Errors propagate to the integration: silently skipping an observation
        and later stitching across that unknown gap would be unsafe.
        """
        source, session_key, run_id, event_id = map(
            _identity, (source, session_key, run_id, event_id)
        )
        event = without_context(event)
        if event.kind is not EventKind.TOOL_CALL:
            return event
        payload = encode_event(event)
        self._journal_pending_gaps(source, session_key)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            scope = db.execute(
                "SELECT fingerprint, closed FROM session_scopes WHERE source=? AND session_key=?",
                (source, session_key),
            ).fetchone()
            if scope is None:
                db.execute(
                    "INSERT INTO session_scopes(source, session_key, fingerprint) VALUES (?, ?, ?)",
                    (source, session_key, self.fingerprint),
                )
            elif scope["closed"]:
                raise ValueError("session is closed; use a new session key")
            elif scope["fingerprint"] != self.fingerprint:
                # Observations under another configuration are not comparable:
                # restart the window instead of disabling the session.
                self._insert_gap(db, source, session_key, "configuration_changed")
                db.execute(
                    "UPDATE session_scopes SET fingerprint=? WHERE source=? AND session_key=?",
                    (self.fingerprint, source, session_key),
                )
            duplicate = db.execute(
                "SELECT event, context FROM session_observations "
                "WHERE source=? AND session_key=? AND run_id=? AND event_id=?",
                (source, session_key, run_id, event_id),
            ).fetchone()
            if duplicate is not None:
                if duplicate["event"] != payload:
                    raise ValueError("session event identity reused with a different payload")
                return self._attach(event, json.loads(duplicate["context"]))
            rows = db.execute(
                "SELECT idx, run_id, event_id, event, context, gap FROM session_observations "
                "WHERE source=? AND session_key=? ORDER BY idx DESC LIMIT ?",
                (source, session_key, MAX_SESSION_EVENTS),
            ).fetchall()
            prior: list[dict[str, Any]] = []
            key = continuity_key(event) if linked else None
            proof = event.metadata.get("progress_proof")
            if key is not None and isinstance(proof, dict):
                for row in rows:
                    if row["gap"] is not None:
                        # Keep looking: a stale proof replayed right after a
                        # gap must still not start a new window.
                        continue
                    previous_proof = decode_event(row["event"]).metadata.get("progress_proof")
                    if (
                        row["run_id"] == run_id
                        and isinstance(previous_proof, dict)
                        and previous_proof.get("spec_hash") == proof["spec_hash"]
                        and type(previous_proof.get("sequence")) is int
                        and previous_proof.get("sequence", -1) >= proof["sequence"]
                    ):
                        key = None
                        break
            token = canonicalize(event, self.tool_masks)
            size = len(payload.encode("utf-8"))
            if key is not None:
                for row in rows:
                    if row["gap"] is not None or not json.loads(row["context"])["observations"]:
                        break
                    previous = decode_event(row["event"])
                    if (
                        continuity_key(previous) != key
                        or canonicalize(previous, self.tool_masks) != token
                    ):
                        break
                    size += len(row["event"].encode("utf-8"))
                    prior.append(
                        {
                            "index": row["idx"],
                            "run_id": row["run_id"],
                            "event_id": row["event_id"],
                            "event": row["event"],
                        }
                    )
            cursor = db.execute(
                "INSERT INTO session_observations"
                "(source, session_key, run_id, event_id, event, context) VALUES (?, ?, ?, ?, ?, ?)",
                (source, session_key, run_id, event_id, payload, "{}"),
            )
            index = cursor.lastrowid
            assert index is not None
            observations = [
                *list(reversed(prior)),
                {"index": index, "run_id": run_id, "event_id": event_id, "event": payload},
            ]
            context: dict[str, Any] = {
                "version": SESSION_VERSION,
                "source": source,
                "session_key": session_key,
                "run_id": run_id,
                "observations": [],
            }
            if (
                key is not None
                and len(observations) <= MAX_SESSION_EVENTS
                and size <= MAX_CONTEXT_BYTES
            ):
                context["observations"] = observations
            db.execute(
                "UPDATE session_observations SET context=? WHERE idx=?",
                (json.dumps(context), index),
            )
            return self._attach(event, context)

    @staticmethod
    def _attach(event: ToolEvent, context: dict[str, Any]) -> ToolEvent:
        return replace(event, metadata={**event.metadata, SESSION_METADATA_KEY: context})

    def lookup(
        self, *, source: str, session_key: str, run_id: str, event_id: str
    ) -> ToolEvent | None:
        """Recover a committed capture without reordering or re-probing it."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT event, context FROM session_observations "
                "WHERE source=? AND session_key=? AND run_id=? AND event_id=?",
                (source, session_key, run_id, event_id),
            ).fetchone()
        return (
            None
            if row is None
            else self._attach(decode_event(row["event"]), json.loads(row["context"]))
        )

    def close_session(self, *, source: str, session_key: str) -> None:
        """Explicit source/caller closure, independent of individual run ends."""
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO session_scopes(source, session_key, fingerprint, closed) "
                "VALUES (?, ?, ?, 1) ON CONFLICT(source, session_key) DO UPDATE SET closed=1",
                (_identity(source), _identity(session_key), self.fingerprint),
            )

    def report(self, *, source: str, session_key: str) -> dict[str, Any]:
        """One row per comparable repeated-action episode, in capture order.

        These are observed incidents, not new alerts or enforcement claims.
        Raw events and canonical per-run decisions are never rewritten.
        """
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT context FROM session_observations WHERE source=? AND session_key=? "
                "ORDER BY idx",
                (_identity(source), _identity(session_key)),
            ).fetchall()
        events = []
        for row in rows:
            context = json.loads(row["context"])
            if context["observations"]:
                events.append(
                    self._attach(decode_event(context["observations"][-1]["event"]), context)
                )
        return project_session(
            events, source=source, session_key=session_key, tool_masks=self.tool_masks
        )


def project_session(
    events: Iterable[ToolEvent],
    *,
    source: str,
    session_key: str,
    tool_masks: ToolMasks | None = None,
) -> dict[str, Any]:
    """Rebuild grouping from frozen traces alone, independent of input file order.

    Indexes describe the session capture order, not per-run decision positions.
    Run/event references link back to canonical decisions without rewriting or
    inventing them. A group is an observation, never an additional alert.
    """
    incidents: dict[int, dict[str, Any]] = {}
    for event in events:
        context = event.metadata.get(SESSION_METADATA_KEY)
        if (
            not isinstance(context, dict)
            or context.get("source") != source
            or context.get("session_key") != session_key
            or not session_history(event, tool_masks)
        ):
            continue
        observations = context["observations"]
        first, last = observations[0], observations[-1]
        existing = incidents.get(first["index"])
        if existing is not None and existing["last_observation_index"] >= last["index"]:
            continue
        incidents[first["index"]] = {
            "kind": "comparable_repeated_action",
            "action_identity": _canonical_sha256(canonicalize(event, tool_masks)),
            "tool": event.tool[:256],
            "occurrences": len(observations),
            "run_ids": list(dict.fromkeys(item["run_id"] for item in observations)),
            "first_observation_index": first["index"],
            "last_observation_index": last["index"],
            "event_refs": [
                {k: item[k] for k in ("run_id", "event_id", "index")} for item in observations
            ],
        }
    return {
        "source": source,
        "session_key": session_key,
        "incidents": [incidents[index] for index in sorted(incidents)],
    }


def main(argv: list[str] | None = None) -> int:
    """Read-only cross-turn report; no database or live probes are required."""
    import argparse

    from pathlayer.config import load_config
    from pathlayer.validation.traces import load_trace

    parser = argparse.ArgumentParser(
        description="Group captured session repetitions without new alerts"
    )
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--session-key", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else None
    report = project_session(
        (event for path in args.traces for event in load_trace(path).events),
        source=args.source,
        session_key=args.session_key,
        tool_masks=cfg.canonicalization.tool_masks if cfg else None,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Session {args.session_key!r} ({args.source})")
        print("Comparable repetitions; actions remain in the per-run decision reports.")
        for incident in report["incidents"]:
            print(
                f"- {incident['tool']!r}: {incident['occurrences']} observations across "
                f"{len(incident['run_ids'])} runs; indexes "
                f"{incident['first_observation_index']}-{incident['last_observation_index']}"
            )
            print("  Runs: " + ", ".join(repr(run) for run in incident["run_ids"]))
        if not report["incidents"]:
            print("No comparable cross-run repetition captured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
