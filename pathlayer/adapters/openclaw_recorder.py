"""Per-run OpenClaw recording: durable run directories + one pipeline per run.

Three properties this module owes its callers, all of them load-bearing:

- **A finished run is immutable.** Once a run reaches a terminal status
  (:data:`~pathlayer.adapters.openclaw_run_store.TERMINAL_STATUSES`) any further
  hook for that ``runId`` -- a duplicated ``agent_end``, a late tool call -- is
  ignored instead of reopening it. Reopening would restart the sequence counter
  and rewrite ``trace.json`` / ``final.json`` from an empty in-memory run.
- **A passive STOP ends analysis, not observation.** Pathlayer is a passive
  observer of OpenClaw (V1 never blocks a tool call), so events keep arriving
  after a STOP. They stay normalized, ordered and written to ``trace.json``;
  only the detectors and the arbitration stop running.
- **A live recording has no inferred ground truth.** Reaching ``agent_end`` or
  receiving a Pathlayer decision describes the observed outcome, not whether
  the run was truly healthy or pathological. The trace therefore stays at the
  neutral :data:`~pathlayer.validation.traces.UNKNOWN_LABEL` until curation.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pathlayer import __version__
from pathlayer.adapters.http_ingest import HttpIngestAdapter
from pathlayer.adapters.openclaw import SOURCE, translate_openclaw_event
from pathlayer.adapters.openclaw_lifecycle import (
    AFTER_TOOL_CALL,
    OpenClawHookKind,
    classify_openclaw_hook,
    extract_openclaw_context,
    extract_openclaw_run_id,
)
from pathlayer.adapters.openclaw_run_store import (
    LISTENER_VERSION,
    SCHEMA_VERSION,
    TERMINAL_STATUSES,
    RunDirectory,
    RunStore,
    append_jsonl,
    atomic_write_json,
    utc_now_iso,
)
from pathlayer.config import PathlayerConfig, StoreSection, build_pipeline
from pathlayer.events import ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies.actions import Action
from pathlayer.store import (
    PersistentPipeline,
    SQLiteStateStore,
    StateClosedError,
    StateCompatibilityError,
    StateConcurrencyError,
    StateCorruptionError,
    StateIncompleteError,
    StateStoreError,
)
from pathlayer.validation.traces import UNKNOWN_LABEL, Trace, trace_to_dict

logger = logging.getLogger(__name__)

PipelineFactory = Callable[[], Pipeline]


@dataclass(slots=True)
class _ActiveRun:
    directory: RunDirectory
    pipeline: Pipeline
    sequence: int = 0
    started: bool = False
    completed: bool = False
    # Every normalized event of the run, in arrival order -- including those
    # recorded after a passive STOP, which the pipeline's own ``history``
    # deliberately excludes (it stops accumulating once halted).
    events: list[ToolEvent] = field(default_factory=list)
    persistence_reported: bool = False


def _hook_part(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else {}


class OpenClawRunRecorder:
    """Record one self-contained run directory per OpenClaw ``runId``."""

    def __init__(
        self,
        runs_root: Path,
        *,
        config: PathlayerConfig | None = None,
        pipeline_factory: PipelineFactory | None = None,
        pathlayer_commit: str | None = None,
    ) -> None:
        """Record runs under ``runs_root``, one pipeline per ``ctx.runId``.

        ``config`` is a loaded :class:`~pathlayer.config.PathlayerConfig` (see
        :func:`~pathlayer.config.load_config`); omitting it keeps the built-in
        defaults. ``pipeline_factory`` stays available for tests and custom
        wiring and is mutually exclusive with ``config``.

        An invalid configuration raises :class:`~pathlayer.config.ConfigError`
        here, at construction: per-run pipelines are built lazily, and the HTTP
        ingest path deliberately swallows handler errors (SEC-3), so a config
        rejected on the first hook instead of at startup would be silent.
        """
        if config is not None and pipeline_factory is not None:
            raise ValueError("pass either config or pipeline_factory, not both")
        self._store = RunStore(runs_root)
        self._pipeline_factory = pipeline_factory
        self._config: PathlayerConfig | None = None
        self._state_store: SQLiteStateStore | None = None
        self._state_init_error: StateStoreError | None = None
        if pipeline_factory is None:
            # A recorder without an explicit config keeps all of its durable
            # artifacts together under runs_root. Explicit store paths retain
            # their normal cwd-relative configuration semantics.
            effective_config = config or PathlayerConfig()
            if effective_config.store.path == StoreSection().path:
                effective_config = replace(
                    effective_config,
                    store=StoreSection(path=str(runs_root / "state.db")),
                )
            build_pipeline(effective_config)  # fail fast before hooks are swallowed
            self._config = effective_config
            try:
                self._state_store = SQLiteStateStore(effective_config.store.path)
            except StateStoreError as error:
                self._state_init_error = error
                logger.exception(
                    "OpenClaw SQLite state is unavailable; raw recording and "
                    "memory-only analysis remain active"
                )
        self._pathlayer_commit = pathlayer_commit or _detect_commit_sha()
        self._lock = threading.Lock()
        self._runs: dict[str, _ActiveRun] = {}
        self._orphan_counters: dict[str, int] = {}

    @property
    def runs_root(self) -> Path:
        return self._store.root

    def startup(self) -> None:
        """Mark unfinished runs as interrupted; never overwrite completed runs."""
        try:
            if self._state_init_error is not None:
                for run in self._store.list_runs():
                    if run.read_manifest().get("status") not in TERMINAL_STATUSES:
                        self._record_persistence_error(run, self._state_init_error)
            elif self._state_store is not None:
                for run in self._store.list_runs():
                    manifest = run.read_manifest()
                    if (
                        manifest.get("status") in TERMINAL_STATUSES
                        or manifest.get("state_durable") is False
                    ):
                        continue
                    try:
                        try:
                            pipeline = self._new_pipeline(run.original_run_id)
                            self._restore_recorded_events(run, pipeline)
                            assert isinstance(pipeline, PersistentPipeline)
                            pipeline.close("interrupted")
                            self._record_pipeline_durability(run, pipeline)
                        except StateClosedError:
                            snapshot = self._state_store.read_scope(run.original_run_id)
                            if snapshot.status in {"completed", "interrupted"}:
                                run.update_manifest(
                                    status=snapshot.status,
                                    ended_at=run.read_manifest().get("ended_at") or utc_now_iso(),
                                )
                    except StateStoreError as error:
                        logger.exception(
                            "could not restore durable state for OpenClaw run %r",
                            run.original_run_id,
                        )
                        self._record_persistence_error(run, error)
        finally:
            self._store.mark_unfinished_as_interrupted()

    def _new_pipeline(self, run_id: str) -> Pipeline:
        if self._pipeline_factory is not None:
            return self._pipeline_factory()
        assert self._config is not None
        if self._state_store is None:
            return build_pipeline(self._config)
        return build_pipeline(self._config, state_key=run_id, store=self._state_store)

    def _memory_pipeline(self) -> Pipeline:
        if self._pipeline_factory is not None:
            return self._pipeline_factory()
        assert self._config is not None
        return build_pipeline(self._config)

    @staticmethod
    def _error_category(error: StateStoreError) -> str:
        return type(error).__name__

    def _record_persistence_error(self, directory: RunDirectory, error: StateStoreError) -> None:
        directory.update_manifest(
            state_durable=False,
            persistence_error=self._error_category(error),
        )

    def _record_pipeline_durability(self, directory: RunDirectory, pipeline: Pipeline) -> None:
        if not isinstance(pipeline, PersistentPipeline) or pipeline.durable:
            return
        error = pipeline.persistence_error
        directory.update_manifest(
            state_durable=False,
            persistence_error=self._error_category(error)
            if error is not None
            else "StateStoreError",
        )

    def _interrupt_active_run(self, active: _ActiveRun, error: StateStoreError) -> None:
        """Contain an incompatible live history without losing its raw journal."""
        if isinstance(active.pipeline, PersistentPipeline):
            active.pipeline.abandon_persistence(error)
        self._record_persistence_error(active.directory, error)
        active.directory.update_manifest(
            status="interrupted",
            ended_at=utc_now_iso(),
            event_count=active.sequence,
            stopped=active.pipeline.stopped,
        )
        active.persistence_reported = True
        active.completed = True
        with self._lock:
            self._runs.pop(active.directory.original_run_id, None)

    def process_payload(self, payload: Any) -> None:
        """Record one forwarded hook payload.

        A payload addressed to an already finished run is dropped before
        anything is written, so a duplicated ``agent_end`` leaves every file --
        and the sequence counter -- exactly as the first one left them.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"OpenClaw payload must be an object, got {type(payload).__name__}")
        hook_kind = classify_openclaw_hook(payload)
        run_id = extract_openclaw_run_id(payload)
        if run_id is None:
            run_id = self._orphan_run_id(payload)
        active = self._get_or_create_run(run_id, payload, hook_kind)
        if active is None:
            return
        record = self._append_raw(active, payload)
        if hook_kind is OpenClawHookKind.RUN_START:
            self._handle_run_start(active, payload, record)
        elif hook_kind is OpenClawHookKind.TOOL:
            self._handle_tool(active, payload, record)
        elif hook_kind is OpenClawHookKind.RUN_END:
            self._handle_run_end(active, payload, record)
        else:
            active.directory.update_manifest(event_count=active.sequence)

    def attach_http(self, *, host: str = "127.0.0.1", port: int = 0) -> HttpIngestAdapter:
        """Start loopback ingest that routes payloads to this recorder."""

        def _ingest_translator(payload: Any) -> list[ToolEvent]:
            self.process_payload(payload)
            return []

        adapter = HttpIngestAdapter(
            lambda _event: None,
            translators={SOURCE: _ingest_translator},
            host=host,
            port=port,
        )
        adapter.attach()
        return adapter

    def _orphan_run_id(self, payload: dict[str, Any]) -> str:
        context = extract_openclaw_context(payload)
        key = context.session_key or "unknown"
        counter = self._orphan_counters.get(key, 0) + 1
        self._orphan_counters[key] = counter
        return f"orphan-{key}-{counter}-{uuid.uuid4().hex[:8]}"

    def _get_or_create_run(
        self, run_id: str, payload: dict[str, Any], hook_kind: OpenClawHookKind
    ) -> _ActiveRun | None:
        """Return the active run for ``run_id``, or ``None`` if it is finished.

        Three cases: a run this process is already recording (returned as is), a
        run directory left behind by an earlier listener (re-attached, resuming
        its sequence rather than restarting at 1), and a terminal run (never
        reopened -- ``None``).
        """
        with self._lock:
            active = self._runs.get(run_id)
            if active is not None:
                status = active.directory.read_manifest().get("status")
                if status in TERMINAL_STATUSES:
                    self._runs.pop(run_id, None)
                    return None
                return active
            existing = self._store.resolve_run(run_id)
            if existing is not None:
                return self._reattach(existing, run_id, payload)
            context = extract_openclaw_context(payload)
            status = "started" if hook_kind is OpenClawHookKind.RUN_START else "orphan"
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "dir_name": None,  # filled by store
                "session_key": context.session_key,
                "agent_id": context.agent_id,
                "status": status,
                "started_at": utc_now_iso(),
                "ended_at": None,
                "event_count": 0,
                # Execution status and health verdict are separate fields on
                # purpose: reaching ``completed`` says the run ended, never that
                # it was sound. The verdict stays neutral until ``agent_end``
                # lets :func:`evaluate_health` describe the observed outcome;
                # live capture itself never promotes the curated label.
                "health": UNKNOWN_LABEL,
                "health_reason": None,
                "stopped": False,
                "pathlayer_version": __version__,
                "pathlayer_commit": self._pathlayer_commit,
                "listener_version": LISTENER_VERSION,
                "openclaw_model": _extract_model(payload),
                "openclaw_provider": _extract_provider(payload),
                "workspace": _extract_workspace(payload),
            }
            if self._pipeline_factory is None:
                manifest["state_durable"] = self._state_store is not None
                manifest["persistence_error"] = (
                    self._error_category(self._state_init_error)
                    if self._state_init_error is not None
                    else None
                )
            directory = self._store.open_or_create(run_id, manifest=manifest)
            manifest["dir_name"] = directory.dir_name
            atomic_write_json(directory.manifest_path, manifest)
            try:
                pipeline = self._new_pipeline(run_id)
            except (
                StateCompatibilityError,
                StateCorruptionError,
                StateIncompleteError,
                StateClosedError,
            ) as error:
                logger.exception("incompatible durable state for OpenClaw run %r", run_id)
                self._record_persistence_error(directory, error)
                directory.update_manifest(status="interrupted", ended_at=utc_now_iso())
                return None
            except StateStoreError as error:
                logger.exception("durable state unavailable for OpenClaw run %r", run_id)
                self._record_persistence_error(directory, error)
                pipeline = self._memory_pipeline()
            active = _ActiveRun(directory=directory, pipeline=pipeline)
            self._runs[run_id] = active
            return active

    def _reattach(
        self, existing: RunDirectory, run_id: str, payload: dict[str, Any]
    ) -> _ActiveRun | None:
        """Adopt an on-disk run directory this process is not yet recording.

        A terminal run is never adopted: it keeps its files and its sequence,
        and the payload is dropped (the duplicated ``agent_end`` case). A
        non-terminal directory -- a listener restarted without the startup
        sweep -- is adopted with its sequence resumed from ``event_count``, so
        ``events.raw.jsonl`` keeps strictly increasing sequence numbers instead
        of restarting at 1. Its manifest is kept as is (never overwritten with a
        fresh, empty one).
        """
        manifest = existing.read_manifest()
        status = manifest.get("status")
        if status in TERMINAL_STATUSES:
            logger.warning(
                "ignoring %r hook for finished run %r (status=%s): "
                "a terminal run is never reopened",
                payload.get("hook"),
                run_id,
                status,
            )
            return None
        logger.debug("re-attaching to unfinished run %r (status=%s)", run_id, status)
        persistence_reported = manifest.get("state_durable") is False
        if persistence_reported:
            logger.warning(
                "interrupting OpenClaw run %r: its SQLite journal is only a prefix",
                run_id,
            )
            existing.update_manifest(status="interrupted", ended_at=utc_now_iso())
            return None
        try:
            pipeline = self._new_pipeline(run_id)
        except (
            StateCompatibilityError,
            StateCorruptionError,
            StateIncompleteError,
            StateClosedError,
        ) as error:
            logger.exception("incompatible durable state for OpenClaw run %r", run_id)
            self._record_persistence_error(existing, error)
            existing.update_manifest(status="interrupted", ended_at=utc_now_iso())
            return None
        except StateStoreError as error:
            logger.exception("durable state unavailable for OpenClaw run %r", run_id)
            self._record_persistence_error(existing, error)
            pipeline = self._memory_pipeline()
            persistence_reported = True
        if self._state_init_error is not None and not persistence_reported:
            self._record_persistence_error(existing, self._state_init_error)
            persistence_reported = True
        try:
            sequence, events = self._restore_recorded_events(existing, pipeline)
        except (StateConcurrencyError, StateCompatibilityError, StateCorruptionError) as error:
            logger.exception("durable/raw journal divergence for OpenClaw run %r", run_id)
            if isinstance(pipeline, PersistentPipeline):
                pipeline.abandon_persistence(error)
            self._record_persistence_error(existing, error)
            existing.update_manifest(status="interrupted", ended_at=utc_now_iso())
            return None
        except StateStoreError as error:
            logger.exception("durable state failed while reattaching OpenClaw run %r", run_id)
            self._record_persistence_error(existing, error)
            persistence_reported = True
            pipeline = self._memory_pipeline()
            sequence, events = self._restore_recorded_events(existing, pipeline)
        if isinstance(pipeline, PersistentPipeline) and not pipeline.durable:
            self._record_pipeline_durability(existing, pipeline)
            persistence_reported = True
            pipeline = self._memory_pipeline()
            sequence, events = self._restore_recorded_events(existing, pipeline)
        sequence = max(sequence, _as_int(manifest.get("event_count")))
        active = _ActiveRun(
            directory=existing,
            pipeline=pipeline,
            sequence=sequence,
            started=existing.request_path.is_file(),
            events=events,
            persistence_reported=persistence_reported,
        )
        self._runs[run_id] = active
        return active

    def _restore_recorded_events(
        self, directory: RunDirectory, pipeline: Pipeline
    ) -> tuple[int, list[ToolEvent]]:
        """Rebuild an unfinished run from its durable raw journal.

        Re-attaching only the sequence counter would make a later ``trace.json``
        omit every pre-restart event and would restart decision indexes at zero.
        The raw JSONL journal is the durable source of truth, so replay its tool
        payloads into a fresh pipeline without writing duplicate decisions.
        """
        if not directory.raw_events_path.is_file():
            return 0, []
        sequence = 0
        events: list[ToolEvent] = []
        for line_number, line in enumerate(
            directory.raw_events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {directory.raw_events_path} at line {line_number}"
                ) from exc
            if not isinstance(record, dict) or not isinstance(record.get("payload"), dict):
                raise ValueError(
                    f"invalid record in {directory.raw_events_path} at line {line_number}"
                )
            sequence = max(sequence, _as_int(record.get("sequence")))
            for offset, event in enumerate(translate_openclaw_event(record["payload"])):
                event.ts = float(len(events))
                events.append(event)
                if not pipeline.stopped:
                    _process_pipeline(
                        pipeline,
                        event,
                        _openclaw_event_id(_as_int(record.get("sequence")), offset),
                    )
        return sequence, events

    def _append_raw(self, active: _ActiveRun, payload: dict[str, Any]) -> dict[str, Any]:
        active.sequence += 1
        record = {
            "sequence": active.sequence,
            "received_at": utc_now_iso(),
            "hook": payload.get("hook"),
            "payload": payload,
        }
        append_jsonl(active.directory.raw_events_path, record)
        return record

    def _handle_run_start(
        self, active: _ActiveRun, payload: dict[str, Any], record: dict[str, Any]
    ) -> None:
        if active.started and active.directory.request_path.is_file():
            # Duplicate start for the same run_id: preserve first request snapshot.
            active.directory.update_manifest(status="started", event_count=active.sequence)
            return
        active.started = True
        hook_event: dict[str, Any] = _hook_part(payload, "event")
        hook_ctx: dict[str, Any] = _hook_part(payload, "ctx")
        request_doc = {
            "sequence": record["sequence"],
            "received_at": record["received_at"],
            "prompt": hook_event.get("prompt"),
            "messages": hook_event.get("messages"),
            "system_prompt": hook_event.get("systemPrompt") or hook_event.get("system_prompt"),
            "session_key": hook_ctx.get("sessionKey"),
            "agent_id": hook_ctx.get("agentId"),
            "run_id": extract_openclaw_run_id(payload),
            "raw": payload,
        }
        atomic_write_json(active.directory.request_path, request_doc)
        active.directory.update_manifest(
            status="started",
            event_count=active.sequence,
            session_key=hook_ctx.get("sessionKey"),
            agent_id=hook_ctx.get("agentId"),
            openclaw_model=_extract_model(payload),
            openclaw_provider=_extract_provider(payload),
            workspace=_extract_workspace(payload),
        )

    def _handle_tool(
        self, active: _ActiveRun, payload: dict[str, Any], record: dict[str, Any]
    ) -> None:
        """Normalize one ``after_tool_call`` and, unless halted, analyze it.

        A STOP in V1 is passive: OpenClaw keeps running and keeps forwarding
        hooks. Every event is therefore appended to :attr:`_ActiveRun.events`
        (and so to ``trace.json``) with its position preserved, whether or not
        the pipeline is still analyzing. After a STOP the detectors and the
        arbitration no longer run -- that is what "stopped" means -- and the
        event is recorded as not analyzed.
        """
        if payload.get("hook") != AFTER_TOOL_CALL:
            return
        for offset, event in enumerate(translate_openclaw_event(payload)):
            event.ts = float(len(active.events))
            active.events.append(event)
            index = len(active.events) - 1
            if active.pipeline.stopped:
                self._record_decision(
                    active, record, event, Action.STOP, index=index, analyzed=False
                )
                continue
            try:
                action = _process_pipeline(
                    active.pipeline,
                    event,
                    _openclaw_event_id(_as_int(record.get("sequence")), offset),
                )
            except StateStoreError as error:
                logger.exception(
                    "interrupting OpenClaw run %r after a durable identity conflict",
                    active.directory.original_run_id,
                )
                self._interrupt_active_run(active, error)
                return
            if (
                isinstance(active.pipeline, PersistentPipeline)
                and not active.pipeline.durable
                and not active.persistence_reported
            ):
                self._record_pipeline_durability(active.directory, active.pipeline)
                active.persistence_reported = True
            self._record_decision(active, record, event, action, index=index)
        active.directory.update_manifest(event_count=active.sequence)

    def _handle_run_end(
        self, active: _ActiveRun, payload: dict[str, Any], record: dict[str, Any]
    ) -> None:
        if active.completed:
            return
        hook_event: dict[str, Any] = _hook_part(payload, "event")
        hook_ctx: dict[str, Any] = _hook_part(payload, "ctx")
        duration_ms = hook_event.get("durationMs")
        if duration_ms is None:
            duration_ms = hook_event.get("duration_ms")
        final_doc = {
            "sequence": record["sequence"],
            "received_at": record["received_at"],
            "success": hook_event.get("success"),
            "duration_ms": duration_ms,
            "messages": hook_event.get("messages"),
            "result": hook_event.get("result"),
            "session_key": hook_ctx.get("sessionKey"),
            "agent_id": hook_ctx.get("agentId"),
            "run_id": extract_openclaw_run_id(payload),
            "raw": payload,
        }
        atomic_write_json(active.directory.final_path, final_doc)
        health, health_reason = evaluate_health(active.pipeline, success=hook_event.get("success"))
        metadata: dict[str, Any] = {
            "run_id": active.directory.original_run_id,
            "session_key": hook_ctx.get("sessionKey"),
            "agent_id": hook_ctx.get("agentId"),
            "recorded_at": utc_now_iso(),
        }
        if health_reason is not None:
            metadata["health_reason"] = health_reason
        trace = Trace(
            label=health,
            description=f"Recorded OpenClaw run {active.directory.original_run_id}",
            events=list(active.events),
            metadata=metadata,
        )
        # Same atomic writer as every other run file: a crash mid-write must
        # leave the previous trace.json intact rather than a truncated one
        # (``dump_trace`` writes in place -- fine for offline corpus tooling,
        # not for a live listener).
        atomic_write_json(active.directory.trace_path, trace_to_dict(trace))
        if isinstance(active.pipeline, PersistentPipeline):
            try:
                active.pipeline.close("completed")
            except StateStoreError as error:
                logger.exception(
                    "OpenClaw run %r completed after durable close failed",
                    active.directory.original_run_id,
                )
                active.pipeline.abandon_persistence(error)
                self._record_persistence_error(active.directory, error)
                active.persistence_reported = True
            self._record_pipeline_durability(active.directory, active.pipeline)
        active.directory.update_manifest(
            status="completed",
            ended_at=utc_now_iso(),
            event_count=active.sequence,
            health=health,
            health_reason=health_reason,
            stopped=active.pipeline.stopped,
        )
        active.completed = True
        with self._lock:
            self._runs.pop(active.directory.original_run_id, None)

    def _record_decision(
        self,
        active: _ActiveRun,
        record: dict[str, Any],
        event: ToolEvent,
        action: Action,
        *,
        index: int,
        analyzed: bool = True,
    ) -> None:
        """Append one line to ``decisions.jsonl``.

        ``index`` is the event's position in ``trace.json``, so the two files
        line up event for event -- including past a passive STOP, where
        ``analyzed`` is ``False`` and no signals are attached because no
        detector ran for that event.
        """
        decision: dict[str, Any] = {
            "sequence": record["sequence"],
            "received_at": record["received_at"],
            "event_index": index,
            "event_ts": event.ts,
            "tool": event.tool,
            "action": action.name,
            "stopped": active.pipeline.stopped,
            "analyzed": analyzed,
        }
        post_mortem = active.pipeline.post_mortem
        if analyzed and post_mortem is not None:
            decision["signals"] = [
                {"name": signal.name, "confidence": signal.confidence, "reason": signal.reason}
                for signal in post_mortem.signals
            ]
            decision["reason"] = post_mortem.reason
        append_jsonl(active.directory.decisions_path, decision)


def evaluate_health(pipeline: Pipeline, *, success: Any) -> tuple[str, str]:
    """Describe the observed outcome without inventing a ground-truth label.

    Live capture can establish what OpenClaw reported and what Pathlayer
    decided, but neither detector silence nor a STOP proves the curated VAL-1
    class. Every recorded trace therefore remains ``unknown`` until a human or
    another explicit curation workflow labels it. ``reason`` preserves the
    useful observed outcome separately from that ground truth.
    """
    post_mortem = pipeline.post_mortem
    if post_mortem is not None:
        return UNKNOWN_LABEL, f"pathlayer stopped: {post_mortem.reason}"
    emitted = sorted({action.name for action in pipeline.actions if action is not Action.NONE})
    if emitted:
        return UNKNOWN_LABEL, f"pathlayer intervened below STOP: {', '.join(emitted)}"
    if pipeline.would_be_actions:
        tiers = sorted({entry.action.name for entry in pipeline.would_be_actions})
        return UNKNOWN_LABEL, f"dry-run would have acted: {', '.join(tiers)}"
    if not pipeline.history:
        return UNKNOWN_LABEL, "no tool call was analyzed"
    if success is not True:
        return UNKNOWN_LABEL, f"agent_end did not report success (success={success!r})"
    return UNKNOWN_LABEL, "live recording requires human curation"


def _as_int(value: Any) -> int:
    """Coerce a manifest counter to a non-negative int (0 when unusable)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _openclaw_event_id(sequence: int, offset: int) -> str:
    """Stable id for one normalized event produced by a raw hook record."""
    return f"openclaw:{sequence}:{offset}"


def _process_pipeline(pipeline: Pipeline, event: ToolEvent, event_id: str) -> Action:
    """Persist built-in recorder pipelines without widening custom factories."""
    if isinstance(pipeline, PersistentPipeline):
        return pipeline.process(event, event_id=event_id)
    return pipeline.process(event)


def _detect_commit_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _extract_model(payload: dict[str, Any]) -> Any:
    hook_event = _hook_part(payload, "event")
    hook_ctx = _hook_part(payload, "ctx")
    return hook_event.get("model") or hook_ctx.get("model") or hook_ctx.get("resolvedModel")


def _extract_provider(payload: dict[str, Any]) -> Any:
    hook_event = _hook_part(payload, "event")
    hook_ctx = _hook_part(payload, "ctx")
    return (
        hook_event.get("provider") or hook_ctx.get("provider") or hook_ctx.get("resolvedProvider")
    )


def _extract_workspace(payload: dict[str, Any]) -> Any:
    hook_ctx = _hook_part(payload, "ctx")
    return hook_ctx.get("workspaceDir") or hook_ctx.get("workspace")
