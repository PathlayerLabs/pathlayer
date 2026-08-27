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

OBS-2 adds a fourth: **every finalized run ends with a report**, written from the
run directory itself (``report.json``) -- at ``agent_end`` for a run that ends
normally, and during the startup sweep for one that was cut short, since nothing
else will ever speak for it. Every analyzed line of ``decisions.jsonl`` now
carries the tick that produced it, so the detector chronology a report needs is
durable rather than reconstructed from a live pipeline.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
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
from pathlayer.adapters.openclaw_report import (
    COST_BUDGET_DISABLED,
    COST_BUDGET_ENABLED,
    COST_BUDGET_STATUS_KEY,
    COST_BUDGET_UNKNOWN,
    cost_manifest_fields,
    tick_columns,
    write_run_report,
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
from pathlayer.detectors.budget import BudgetDetector, BudgetState
from pathlayer.detectors.proof import _PROGRESS_PROOF_KEY, _PROOF_SPEC_KEY
from pathlayer.detectors.proof_producer import ProgressProofProducer, build_producer
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies.actions import Action
from pathlayer.postmortem import TickRecord
from pathlayer.resource_sampling import sample_rss_bytes
from pathlayer.runtime import Scheduler, WatchdogEventSink, stamp_rss, wrap_with_rss_sampling
from pathlayer.store import (
    PersistentPipeline,
    SQLiteStateStore,
    StateClosedError,
    StateCompatibilityError,
    StateConcurrencyError,
    StateCorruptionError,
    StateIncompleteError,
    StateStoreError,
    StateWriteError,
)
from pathlayer.validation.traces import UNKNOWN_LABEL, Trace, trace_to_dict

logger = logging.getLogger(__name__)

PipelineFactory = Callable[[], Pipeline]
_LIFECYCLE_HOOK = "pathlayer_lifecycle"
# ANC-2: reserved hook name for a durable proof sidecar record (see
# _handle_tool/_restore_recorded_events). A probe result cannot be recomputed
# from the raw wire payload alone -- replaying it must reconstruct the exact
# enriched event a live PersistentPipeline durably committed, or a resumed
# run's idempotency check (store.py's duplicate-tick comparison) sees a
# mismatched event and raises StateConcurrencyError.
_PROOF_HOOK = "pathlayer_progress_proof"


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
    # OBS-2 cost attribution only, and only when a budget detector is configured.
    # Advanced event by event so the estimate is already durable in the manifest
    # if the process dies before ``agent_end``; equal by construction to folding
    # ``events`` in one batch (``BudgetDetector.accumulate``).
    cost_state: BudgetState | None = None
    # HTTP hooks and the timer are two producers for one run. They share this
    # re-entrant lock so lifecycle injection, raw journaling and decisions stay
    # in one total order without deadlocking when start_run emits synchronously.
    lock: Any = field(default_factory=threading.RLock)
    runtime_sink: WatchdogEventSink | None = None
    # A lifecycle sink failure can occur after raw/event journaling but before
    # the pipeline commits. Reuse this logical record on the bounded watchdog
    # retry so RUN_START/RUN_TIMEOUT remain unique in raw history and replay.
    runtime_records: dict[EventKind, tuple[dict[str, Any], ToolEvent, int]] = field(
        default_factory=dict
    )


def _hook_part(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _translate_recorded_payload(
    payload: dict[str, Any], *, internal_lifecycle: bool = False
) -> list[ToolEvent]:
    """Translate external OpenClaw hooks and Pathlayer lifecycle records."""
    if payload.get("hook") != _LIFECYCLE_HOOK:
        return translate_openclaw_event(payload)
    if not internal_lifecycle:
        # The hook name is reserved for records synthesized by this recorder.
        # Never promote an externally supplied payload into a trusted timeout
        # merely because it later passes through journal replay.
        return []
    raw = _hook_part(payload, "event")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        kind = EventKind(raw.get("kind"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid recorded Pathlayer lifecycle kind") from exc
    if kind not in {EventKind.RUN_START, EventKind.RUN_TIMEOUT}:
        raise ValueError("recorded Pathlayer lifecycle kind is not a run boundary")
    return [
        ToolEvent(
            tool=str(raw.get("tool") or f"pathlayer_{kind.value}"),
            kind=kind,
            run_id=extract_openclaw_run_id(payload),
            metadata=dict(metadata),
        )
    ]


def _write_report(directory: RunDirectory) -> bool:
    """Write a run's OBS-2 report, never letting a failure escape the hook.

    The report is a summary of data that is already durable, so failing to write
    it must not take down the listener -- one unreadable journal would otherwise
    abort the whole startup sweep, or turn a completed run's ``agent_end`` into a
    raised exception. The failure is logged rather than swallowed silently.
    """
    try:
        write_run_report(directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "could not write the session report for run %r: %s",
            directory.original_run_id,
            exc,
        )
        return False
    return True


def _budget_detector_for(config: PathlayerConfig | None) -> BudgetDetector | None:
    """A detector mirroring the run's budget config, for cost attribution only.

    ``None`` when there is no config or the budget detector is disabled: no
    per-tool estimate was ever computed for such a run, and the report states
    that instead of printing ``$0.00`` as though it had measured something.

    The detector is stateless -- each run carries its own accumulator, which is
    passed in and returned (see
    :meth:`~pathlayer.detectors.budget.BudgetDetector.step`) -- so one instance
    shared across concurrent runs cannot leak one run's spend into another's
    report. Values are already validated: the constructor builds a pipeline from
    the same config first, which rejects an out-of-range ceiling.
    """
    if config is None or not config.detectors.budget.enabled:
        return None
    spec = config.detectors.budget
    return BudgetDetector(
        max_cost_usd=spec.max_cost_usd,
        baseline_usd=spec.baseline_usd,
        default_cost_usd=spec.default_cost_usd,
        tool_costs_usd=spec.tool_costs_usd,
    )


class OpenClawRunRecorder:
    """Record one self-contained run directory per OpenClaw ``runId``."""

    def __init__(
        self,
        runs_root: Path,
        *,
        config: PathlayerConfig | None = None,
        pipeline_factory: PipelineFactory | None = None,
        pathlayer_commit: str | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        scheduler: Scheduler | None = None,
        sample_rss: Callable[[], int | None] = sample_rss_bytes,
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
        self._proof_producer: ProgressProofProducer | None = None
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
            if effective_config.detectors.progress_proof.enabled:
                # Same fail-fast boundary as build_pipeline above: a malformed
                # manifest must reject the recorder at construction, not the
                # first swallowed HTTP hook (SEC-3).
                self._proof_producer = build_producer(
                    effective_config.detectors.progress_proof.manifest_path,
                    effective_config.detectors.progress_proof.probe_command,
                    effective_config.detectors.progress_proof.timeout_seconds,
                )
            try:
                self._state_store = SQLiteStateStore(effective_config.store.path)
            except StateStoreError as error:
                self._state_init_error = error
                logger.exception(
                    "OpenClaw SQLite state is unavailable; raw recording and "
                    "memory-only analysis remain active"
                )
        # Kept for the OBS-2 cost section only: the BUD-3 estimate and the
        # cross-run baseline depend on this config and cannot be recomputed from
        # the run directory later. A recorder wired with a bare
        # ``pipeline_factory`` (tests, custom wiring) has none, and the report
        # says the estimate is unavailable rather than inventing a zero.
        self._budget_detector = _budget_detector_for(config)
        self._pathlayer_commit = pathlayer_commit or _detect_commit_sha()
        self._clock = clock
        self._monotonic = monotonic
        self._scheduler = scheduler
        self._sample_rss = sample_rss
        self._lock = threading.Lock()
        self._runs: dict[str, _ActiveRun] = {}
        self._orphan_counters: dict[str, int] = {}

    @property
    def runs_root(self) -> Path:
        return self._store.root

    def startup(self) -> None:
        """Mark unfinished runs as interrupted; never overwrite completed runs.

        An interrupted run gets its OBS-2 ``report.json`` written here, at the
        only moment anything can: its process is gone, so it will never reach
        ``agent_end``. The report is rebuilt from the run directory, so it
        describes what was actually recorded before the cut -- and its outcome is
        ``interrupted``, never ``completed``.

        This sweep adds no cost figure of its own, and must not: the recording
        process is the only thing that knew the run's budget configuration. It
        persisted the estimate as it went (see :meth:`_cost_fields`), so the
        report reads the last durable value. When even that is missing, the cost
        section reports the estimate as indeterminate rather than blaming a
        detector that may well have been enabled.
        """
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
                                # Terminal here means the sweep below will not
                                # see this run, so its report is written now.
                                _write_report(run)
                    except StateStoreError as error:
                        logger.exception(
                            "could not restore durable state for OpenClaw run %r",
                            run.original_run_id,
                        )
                        self._record_persistence_error(run, error)
        finally:
            for run in self._store.mark_unfinished_as_interrupted():
                _write_report(run)

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
        self._finalize_interrupted_run(active.directory, error=error, active=active)

    def _finalize_interrupted_run(
        self,
        directory: RunDirectory,
        *,
        error: StateStoreError | None = None,
        active: _ActiveRun | None = None,
    ) -> None:
        """Make one interrupted run terminal and leave its report behind.

        Every caller that discovers an unrecoverable persistence state goes
        through this seam. Otherwise setting the terminal manifest status and
        returning would make the startup sweep skip the run forever, leaving no
        ``report.json``. For a live run, persist the cost accumulator *after* the
        failing event was folded so the controlled interruption does not lose
        its final known spend.
        """
        if active is not None:
            # Stop timer production before any filesystem write below: this is
            # already an error path, so even the manifest update may fail.
            active.completed = True
            if active.runtime_sink is not None:
                active.runtime_sink.end_run()
        fields: dict[str, Any] = {
            "status": "interrupted",
            "ended_at": utc_now_iso(),
        }
        if active is not None:
            fields.update(
                event_count=active.sequence,
                stopped=active.pipeline.stopped,
                **self._cost_fields(active),
            )
            active.persistence_reported = True
        if error is not None:
            try:
                self._record_persistence_error(directory, error)
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception(
                    "could not persist the interruption error for OpenClaw run %r",
                    directory.original_run_id,
                )
        try:
            directory.update_manifest(**fields)
        except (OSError, ValueError, KeyError, TypeError):
            logger.exception(
                "could not persist terminal interruption metadata for OpenClaw run %r",
                directory.original_run_id,
            )
        _write_report(directory)
        if active is not None:
            with self._lock:
                self._runs.pop(directory.original_run_id, None)

    def process_payload(self, payload: Any) -> None:
        """Record one forwarded hook payload.

        A payload addressed to an already finished run is dropped before
        anything is written, so a duplicated ``agent_end`` leaves every file --
        and the sequence counter -- exactly as the first one left them.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"OpenClaw payload must be an object, got {type(payload).__name__}")
        if payload.get("hook") == _LIFECYCLE_HOOK:
            raise ValueError(f"OpenClaw hook {_LIFECYCLE_HOOK!r} is reserved for Pathlayer")
        if payload.get("hook") == _PROOF_HOOK:
            raise ValueError(f"OpenClaw hook {_PROOF_HOOK!r} is reserved for Pathlayer")
        hook_kind = classify_openclaw_hook(payload)
        run_id = extract_openclaw_run_id(payload)
        if run_id is None:
            run_id = self._orphan_run_id(payload)
        active = self._get_or_create_run(run_id, payload, hook_kind)
        if active is None:
            return
        with active.lock:
            if active.completed:
                return
            assert active.runtime_sink is not None
            if any(event.kind is EventKind.RUN_START for event in active.pipeline.history):
                # Reattachment happens under the recorder's global map lock;
                # re-arm only after it is released so an immediately overdue
                # callback can safely finalize/remove this run.
                active.runtime_sink.start_run()
            elif hook_kind is OpenClawHookKind.TOOL:
                # A missing before_agent_run must not disable RES-2. Start at
                # the first observable tool boundary and make the unmeasured
                # prefix explicit in logs, matching the generic sink fallback.
                active.runtime_sink.start_run(fallback=True)
            if active.completed:
                return
            # OpenClaw processes tool hooks itself rather than forwarding them
            # through WatchdogEventSink.__call__. Preserve the same delayed-
            # scheduler guarantee explicitly at every observed HTTP boundary,
            # Check before reserving the external sequence so the raw source of
            # truth and live processing share one total order. A successful
            # passive STOP still records the payload below as not analyzed; a
            # controlled interruption does not append past its terminal state.
            active.runtime_sink.check_deadline()
            if active.completed:
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
                COST_BUDGET_STATUS_KEY: (
                    COST_BUDGET_UNKNOWN
                    if self._pipeline_factory is not None
                    else (
                        COST_BUDGET_DISABLED
                        if self._budget_detector is None
                        else COST_BUDGET_ENABLED
                    )
                ),
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
                self._finalize_interrupted_run(directory, error=error)
                return None
            except StateStoreError as error:
                logger.exception("durable state unavailable for OpenClaw run %r", run_id)
                self._record_persistence_error(directory, error)
                pipeline = self._memory_pipeline()
            active = _ActiveRun(
                directory=directory,
                pipeline=pipeline,
                cost_state=self._initial_cost_state(),
            )
            self._arm_runtime(active)
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
            self._finalize_interrupted_run(existing)
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
            self._finalize_interrupted_run(existing, error=error)
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
            self._finalize_interrupted_run(existing, error=error)
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
            # The replayed events are part of this run's spend: fold them in one
            # batch so the resumed accumulator matches an uninterrupted one.
            cost_state=(
                None if self._budget_detector is None else self._budget_detector.accumulate(events)
            ),
        )
        self._arm_runtime(active)
        self._runs[run_id] = active
        return active

    def _arm_runtime(self, active: _ActiveRun) -> None:
        # RES-3: tool-call events never reach this sink -- OpenClaw processes
        # tool hooks itself, see _handle_tool -- so they are stamped directly
        # there via self._sample_rss instead. RSS sampling is stateless (a
        # live current reading, not a growth delta), so no per-run object
        # needs to be shared between the two call sites.
        inner_sink = wrap_with_rss_sampling(
            partial(self._handle_runtime_event, active),
            max_duration_seconds=active.pipeline.max_duration_seconds,
            sample=self._sample_rss,
        )
        active.runtime_sink = WatchdogEventSink(
            inner_sink,
            max_duration_seconds=active.pipeline.max_duration_seconds,
            history=lambda: active.pipeline.history,
            clock=self._clock,
            monotonic=self._monotonic,
            scheduler=self._scheduler,
            terminal=lambda: active.completed or active.pipeline.stopped,
            logger=logger,
            lock=active.lock,
        )

    def _initial_cost_state(self) -> BudgetState | None:
        """A fresh cost accumulator, or ``None`` when no budget detector is wired."""
        return None if self._budget_detector is None else self._budget_detector.create_state()

    def _restore_recorded_events(
        self, directory: RunDirectory, pipeline: Pipeline
    ) -> tuple[int, list[ToolEvent]]:
        """Rebuild an unfinished run from its durable raw journal.

        Re-attaching only the sequence counter would make a later ``trace.json``
        omit every pre-restart event and would restart decision indexes at zero.
        The raw JSONL journal is the durable source of truth, so replay its tool
        payloads into a fresh pipeline without writing duplicate decisions.

        ANC-2's proof sidecar records (``_PROOF_HOOK``, durably journaled by
        ``_handle_tool`` alongside the tool event they enriched) are collected
        first and re-applied by ``event_id`` below -- never replayed as events
        of their own, and never re-derived by re-probing: a live probe result
        cannot be recomputed from the raw wire payload alone, and the
        reconstructed event must match byte-for-byte what a live
        ``PersistentPipeline`` already durably committed, or its duplicate-tick
        idempotency check (see ``store.py``) raises ``StateConcurrencyError``.
        """
        if not directory.raw_events_path.is_file():
            return 0, []
        lines = directory.raw_events_path.read_text(encoding="utf-8").splitlines()
        proofs_by_event_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # malformed JSON is re-raised properly by the pass below
            if not isinstance(record, dict) or record.get("hook") != _PROOF_HOOK:
                continue
            sidecar = record.get("payload")
            if not isinstance(sidecar, dict):
                continue
            event_id = sidecar.get("for_event_id")
            proof_spec_payload = sidecar.get(_PROOF_SPEC_KEY)
            progress_proof_payload = sidecar.get(_PROGRESS_PROOF_KEY)
            if (
                isinstance(event_id, str)
                and isinstance(proof_spec_payload, dict)
                and isinstance(progress_proof_payload, dict)
            ):
                proofs_by_event_id[event_id] = (proof_spec_payload, progress_proof_payload)

        sequence = 0
        events: list[ToolEvent] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {directory.raw_events_path} at line {line_number}"
                ) from exc
            if isinstance(record, dict):
                # Sidecars occupy a real slot in the append-only raw journal.
                # Include them when recovering the next sequence even though
                # they are folded into their tool event rather than replayed as
                # events themselves. Otherwise a crash after the sidecar fsync
                # but before the manifest update can make reattachment reuse
                # the sidecar's sequence number.
                sequence = max(sequence, _as_int(record.get("sequence")))
            if isinstance(record, dict) and record.get("hook") == _PROOF_HOOK:
                continue  # already folded into proofs_by_event_id above
            if not isinstance(record, dict) or not isinstance(record.get("payload"), dict):
                raise ValueError(
                    f"invalid record in {directory.raw_events_path} at line {line_number}"
                )
            internal_lifecycle = record.get("internal_lifecycle") is True
            for offset, event in enumerate(
                _translate_recorded_payload(
                    record["payload"], internal_lifecycle=internal_lifecycle
                )
            ):
                event.ts = float(len(events))
                event_id = _openclaw_event_id(_as_int(record.get("sequence")), offset)
                proof = proofs_by_event_id.get(event_id)
                if proof is not None:
                    proof_spec_payload, progress_proof_payload = proof
                    event.metadata[_PROOF_SPEC_KEY] = proof_spec_payload
                    event.metadata[_PROGRESS_PROOF_KEY] = progress_proof_payload
                events.append(event)
                if not pipeline.stopped:
                    _process_pipeline(pipeline, event, event_id)
        return sequence, events

    def _append_raw(
        self,
        active: _ActiveRun,
        payload: dict[str, Any],
        *,
        internal_lifecycle: bool = False,
    ) -> dict[str, Any]:
        active.sequence += 1
        record = {
            "sequence": active.sequence,
            "received_at": utc_now_iso(),
            "hook": payload.get("hook"),
            "payload": payload,
        }
        if internal_lifecycle:
            record["internal_lifecycle"] = True
        append_jsonl(active.directory.raw_events_path, record)
        return record

    def _handle_run_start(
        self, active: _ActiveRun, payload: dict[str, Any], record: dict[str, Any]
    ) -> None:
        assert active.runtime_sink is not None
        active.runtime_sink.start_run()
        if active.completed:
            return
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

    def _handle_runtime_event(self, active: _ActiveRun, event: ToolEvent) -> None:
        """Journal, analyze and report one synthetic lifecycle event."""
        pending = active.runtime_records.get(event.kind)
        if pending is None:
            payload = {
                "hook": _LIFECYCLE_HOOK,
                "event": {
                    "kind": event.kind.value,
                    "tool": event.tool,
                    "metadata": dict(event.metadata),
                },
                "ctx": {"runId": active.directory.original_run_id},
            }
            raw_path = active.directory.raw_events_path
            try:
                size_before = raw_path.stat().st_size if raw_path.exists() else 0
            except OSError:
                size_before = None
            try:
                record = self._append_raw(active, payload, internal_lifecycle=True)
            except OSError as error:
                # append_jsonl may have written and flushed the complete line
                # before fsync failed. Adopt that exact last record if visible;
                # otherwise propagate so a later retry may append safely.
                recovered_record = self._recover_appended_lifecycle(active, payload)
                if recovered_record is None:
                    try:
                        size_after = raw_path.stat().st_size if raw_path.exists() else 0
                    except OSError:
                        size_after = None
                    if size_before is not None and size_after == size_before:
                        # No bytes own the reserved sequence; return it before
                        # the bounded retry. A size change with no complete
                        # record means a torn JSONL line and is non-durable.
                        active.sequence -= 1
                        raise
                wrapped = StateWriteError(
                    "OpenClaw lifecycle journal could not confirm append durability"
                )
                wrapped.__cause__ = error
                logger.error(
                    "interrupting OpenClaw run %r after lifecycle journal durability failed",
                    active.directory.original_run_id,
                    exc_info=(type(error), error, error.__traceback__),
                )
                self._interrupt_active_run(active, wrapped)
                return
            normalized = _translate_recorded_payload(payload, internal_lifecycle=True)[0]
            normalized.ts = float(len(active.events))
            active.events.append(normalized)
            index = len(active.events) - 1
            active.runtime_records[event.kind] = (record, normalized, index)
        else:
            record, normalized, index = pending
        if active.pipeline.stopped:
            self._record_decision(
                active, record, normalized, Action.STOP, index=index, analyzed=False
            )
            return
        try:
            action = _process_pipeline(
                active.pipeline,
                normalized,
                _openclaw_event_id(_as_int(record.get("sequence")), 0),
            )
        except StateStoreError as error:
            logger.exception(
                "interrupting OpenClaw run %r after lifecycle persistence failed",
                active.directory.original_run_id,
            )
            self._interrupt_active_run(active, error)
            return
        durability_must_be_reported = (
            isinstance(active.pipeline, PersistentPipeline)
            and not active.pipeline.durable
            and not active.persistence_reported
        )
        if normalized.kind is EventKind.RUN_TIMEOUT:
            self._finalize_timeout_artifacts(active, record, normalized, action, index=index)
            return
        if durability_must_be_reported:
            assert isinstance(active.pipeline, PersistentPipeline)
            self._record_pipeline_durability(active.directory, active.pipeline)
            active.persistence_reported = True
        self._record_decision(active, record, normalized, action, index=index)
        active.directory.update_manifest(
            event_count=active.sequence,
            stopped=active.pipeline.stopped,
            **self._cost_fields(active),
        )

    def _finalize_timeout_artifacts(
        self,
        active: _ActiveRun,
        record: dict[str, Any],
        event: ToolEvent,
        action: Action,
        *,
        index: int,
    ) -> None:
        """Persist the timeout snapshot after its pipeline tick committed.

        Lifecycle identity is already durable at this point, so artifact
        failures must never escape to the watchdog and invite another timeout.
        Instead the run becomes an honest controlled interruption while the
        canonical RUN_TIMEOUT stays committed exactly once.
        """
        trace_document = trace_to_dict(
            Trace(
                label=UNKNOWN_LABEL,
                description=(
                    f"Recorded OpenClaw run {active.directory.original_run_id} "
                    "at its runtime timeout"
                ),
                events=list(active.events),
                metadata={
                    "run_id": active.directory.original_run_id,
                    "recorded_at": utc_now_iso(),
                    "snapshot": "runtime_timeout",
                },
            )
        )
        try:
            if (
                isinstance(active.pipeline, PersistentPipeline)
                and not active.pipeline.durable
                and not active.persistence_reported
            ):
                # The timeout tick may have committed only in memory. Publishing
                # that poison marker is part of the same protected artifact
                # phase as the snapshot and decision; a manifest failure here
                # must produce the same controlled interruption.
                self._record_pipeline_durability(active.directory, active.pipeline)
                active.persistence_reported = True
            # Write the event-sourced snapshot first: if a later sidecar fails,
            # the controlled-interruption report can still describe the timeout.
            atomic_write_json(active.directory.trace_path, trace_document)
            self._record_decision(active, record, event, action, index=index)
            active.directory.update_manifest(
                event_count=active.sequence,
                stopped=active.pipeline.stopped,
                **self._cost_fields(active),
            )
            if not _write_report(active.directory):
                raise OSError("OpenClaw timeout report could not be written")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._interrupt_after_timeout_artifact_failure(active, error, trace_document)

    def _interrupt_after_timeout_artifact_failure(
        self,
        active: _ActiveRun,
        error: Exception,
        trace_document: dict[str, Any],
    ) -> None:
        """Contain a sidecar failure without invalidating the durable tick."""
        logger.error(
            "interrupting OpenClaw run %r after timeout artifact finalization failed",
            active.directory.original_run_id,
            exc_info=(type(error), error, error.__traceback__),
        )
        active.completed = True
        if active.runtime_sink is not None:
            active.runtime_sink.end_run()

        # Atomic trace writes are safe to retry and maximize the evidence an
        # interrupted report can consume. Append-only decisions are not retried:
        # an fsync failure could have left a complete line visible.
        if not active.directory.trace_path.is_file():
            try:
                atomic_write_json(active.directory.trace_path, trace_document)
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception(
                    "could not preserve the timeout trace for interrupted OpenClaw run %r",
                    active.directory.original_run_id,
                )

        persistence_error: StateStoreError | None = None
        if isinstance(active.pipeline, PersistentPipeline):
            active.pipeline.close("interrupted")
            if not active.pipeline.durable:
                persistence_error = active.pipeline.persistence_error or StateWriteError(
                    "OpenClaw durable scope could not close after artifact failure"
                )
        self._finalize_interrupted_run(
            active.directory,
            error=persistence_error,
            active=active,
        )

    @staticmethod
    def _recover_appended_lifecycle(
        active: _ActiveRun, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        path = active.directory.raw_events_path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return None
            record = json.loads(lines[-1])
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(record, dict)
            or record.get("sequence") != active.sequence
            or record.get("internal_lifecycle") is not True
            or record.get("payload") != payload
        ):
            return None
        return record

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
            # RES-3: tool-call events never reach runtime_sink ("OpenClaw
            # processes tool hooks itself", see process_payload above), so
            # stamp them here directly instead. Same "is RSS sampling
            # wanted" gate as wrap_with_rss_sampling (see its docstring).
            if active.pipeline.max_duration_seconds is not None:
                event = stamp_rss(event, sample=self._sample_rss)
            event_id = _openclaw_event_id(_as_int(record.get("sequence")), offset)
            # ANC-2: probe live, once per captured tool-call event, same as RSS
            # above. Unlike RSS (a real resource cost that keeps accruing after
            # a passive STOP), a proof is pure detector input: once the
            # pipeline has stopped, detectors never run again, so probing
            # further would only spend the operator's probe cost for a result
            # nothing reads.
            if self._proof_producer is not None and not active.pipeline.stopped:
                attached = self._proof_producer.attach(event)
                if attached is not None:
                    proof_spec_payload, progress_proof_payload = attached
                    # A raw-journal replay can never re-derive a live probe
                    # result (nor should it re-invoke the probe -- see
                    # _restore_recorded_events). Durably journal exactly what
                    # was attached as its own sidecar record, correlated by
                    # event_id, so a resumed run reconstructs the identical
                    # enriched event a live PersistentPipeline already
                    # committed -- otherwise the idempotency check in
                    # store.py's duplicate-tick comparison sees a mismatch and
                    # raises StateConcurrencyError.
                    self._append_raw(
                        active,
                        {
                            "hook": _PROOF_HOOK,
                            "for_event_id": event_id,
                            _PROOF_SPEC_KEY: proof_spec_payload,
                            _PROGRESS_PROOF_KEY: progress_proof_payload,
                        },
                        internal_lifecycle=True,
                    )
            event.ts = float(len(active.events))
            active.events.append(event)
            self._advance_cost(active, event)
            index = len(active.events) - 1
            if active.pipeline.stopped:
                self._record_decision(
                    active, record, event, Action.STOP, index=index, analyzed=False
                )
                continue
            try:
                action = _process_pipeline(active.pipeline, event, event_id)
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
        active.directory.update_manifest(event_count=active.sequence, **self._cost_fields(active))

    def _handle_run_end(
        self, active: _ActiveRun, payload: dict[str, Any], record: dict[str, Any]
    ) -> None:
        if active.completed:
            return
        assert active.runtime_sink is not None
        active.runtime_sink.end_run()
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
            **self._cost_fields(active),
        )
        active.completed = True
        # OBS-2: written last, from the finalized directory, so the report is
        # exactly what a later ``pathlayer-report`` rebuilds from the same files.
        _write_report(active.directory)
        with self._lock:
            self._runs.pop(active.directory.original_run_id, None)

    def _advance_cost(self, active: _ActiveRun, event: ToolEvent) -> None:
        """Fold one event into the run's cost accumulator (OBS-2), if any.

        Every recorded event counts, analyzed or not: a call made after a passive
        STOP still spent money. The arbitration side of
        :meth:`~pathlayer.detectors.budget.BudgetDetector.step` is discarded --
        this accumulator only attributes cost in the report, it never influences
        detection, which the pipeline's own budget detector owns.
        """
        detector = self._budget_detector
        if detector is None or active.cost_state is None:
            return
        active.cost_state, _ = detector.step(active.cost_state, event)

    def _cost_fields(self, active: _ActiveRun) -> dict[str, Any]:
        """Manifest cost fields for this run, empty when nothing computes them.

        Written on every manifest update rather than only at ``agent_end``, so a
        run whose process disappears keeps the estimate for what it had already
        spent. Without that, its report could not tell "the budget detector was
        off" from "the run was cut before anything was persisted", and it used to
        assert the first.
        """
        if self._budget_detector is None or active.cost_state is None:
            return {}
        return cost_manifest_fields(self._budget_detector, active.cost_state)

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
        ``analyzed`` is ``False``.

        OBS-2: every *analyzed* line also carries the tick Pathlayer just
        computed (every detector's confidence, the fused score, the arbitrated
        action and its reason -- see
        :func:`~pathlayer.adapters.openclaw_report.tick_columns`). This journal is
        therefore the run's durable detector chronology, and an end-of-session
        report needs nothing else. Before OBS-2 those columns only appeared once
        a post-mortem existed, i.e. from the STOP onward, so a run that never
        stopped left no detector evidence at all.

        A line past a passive STOP still carries none: no detector ran for that
        event, and repeating the STOP tick's confidences next to it would assert
        a reading that was never taken.
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
        last_decision = active.pipeline.last_decision
        if analyzed and last_decision is not None:
            decision.update(tick_columns(TickRecord.from_decision(last_decision)))
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
