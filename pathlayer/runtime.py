"""Runtime lifecycle and hard-duration watchdog (RES-2).

The watchdog is deliberately outside detectors, policy and Pipeline. It owns
the clocks and scheduler, while the deterministic core sees only ordinary,
persistable RUN_START/RUN_TIMEOUT events.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from pathlayer.adapters.base import EventSink, RunAwareEventSink
from pathlayer.events import (
    RUN_START_TOOL,
    RUN_TIMEOUT_TOOL,
    EventKind,
    ToolEvent,
)


class CancelHandle(Protocol):
    def cancel(self) -> None: ...


class Scheduler(Protocol):
    def call_later(self, delay_seconds: float, callback: Callable[[], None]) -> CancelHandle: ...


class _ThreadingScheduler:
    def call_later(self, delay_seconds: float, callback: Callable[[], None]) -> CancelHandle:
        timer = threading.Timer(delay_seconds, callback)
        timer.daemon = True
        timer.start()
        return timer


class WatchdogEventSink(RunAwareEventSink):
    """Serialize external events and one synthetic timeout for a logical run."""

    def __init__(
        self,
        sink: EventSink,
        *,
        max_duration_seconds: int | None,
        history: Iterable[ToolEvent] | Callable[[], Iterable[ToolEvent]] = (),
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        scheduler: Scheduler | None = None,
        terminal: Callable[[], bool] | None = None,
        logger: logging.Logger | None = None,
        lock: Any | None = None,
    ) -> None:
        if max_duration_seconds is not None and (
            isinstance(max_duration_seconds, bool)
            or not isinstance(max_duration_seconds, int)
            or max_duration_seconds < 1
        ):
            raise ValueError("max_duration_seconds must be an int >= 1")
        self._sink = sink
        self._max_duration_seconds = max_duration_seconds
        self._history = history
        self._clock = clock
        self._monotonic = monotonic
        self._scheduler = scheduler or _ThreadingScheduler()
        self._terminal = terminal or (lambda: False)
        self._logger = logger or logging.getLogger(__name__)
        self._lock = lock or threading.RLock()
        self._timer: CancelHandle | None = None
        self._started_at: float | None = None
        self._deadline_monotonic: float | None = None
        self._started = False
        self._start_attempts = 0
        self._timed_out = False
        self._timeout_attempts = 0
        self._ended = False
        self._fallback_warned = False
        # Transient: the anchor _commit_start() resolved, if any, kept around
        # solely so a failure *after* resolution can retry with that same
        # anchor (see _recover_from_start_failure) instead of drifting to a
        # later now() on retry.
        self._pending_anchor: float | None = None

    def __call__(self, event: ToolEvent) -> None:
        with self._lock:
            self._expire_if_due()
            if self._ended:
                return
            if self._max_duration_seconds is not None and not self._started:
                self.start_run(fallback=True)
            self._sink(event)

    def start_run(self, *, started_at: float | None = None, fallback: bool = False) -> None:
        with self._lock:
            if self._ended or self._started or self._max_duration_seconds is None:
                return
            if fallback and not self._fallback_warned:
                self._logger.warning(
                    "run lifecycle was not started explicitly; starting the RES-2 "
                    "watchdog at the first event, so earlier duration was not measured"
                )
                self._fallback_warned = True
            self._pending_anchor = None
            try:
                self._commit_start(started_at)
            except Exception:
                self._recover_from_start_failure(started_at, fallback)

    def _resolve_anchor(
        self, durable_start: float | None, started_at: float | None, now: float
    ) -> float:
        """Priority: an already-durable start, else the caller's, else now."""
        if durable_start is not None:
            return durable_start
        if started_at is not None:
            return self._validated_clock(started_at, "started_at")
        return now

    def _arm(self, anchor: float, now: float, monotonic_now: float) -> float:
        """Set started_at/deadline_monotonic from one (anchor, now, monotonic_now)
        reading and return the remaining seconds computed from it (pre-clamped
        to >= 0), so a caller that must not re-read the clock afterward (see
        ``_recover_committed_start``) can still fire-or-schedule exactly.
        """
        assert self._max_duration_seconds is not None
        remaining = max(0.0, self._max_duration_seconds - max(0.0, now - anchor))
        self._started_at = anchor
        self._deadline_monotonic = monotonic_now + remaining
        return remaining

    def _fire_or_schedule(self, remaining: float) -> None:
        if remaining <= 0.0:
            self._on_timeout()
        else:
            self._timer = self._scheduler.call_later(remaining, self._on_timeout)

    def _commit_start(self, started_at: float | None) -> None:
        """Happy path: resolve the anchor, arm the deadline, emit RUN_START once.

        May raise partway through (e.g. the sink call below persisting
        RUN_START); ``self._pending_anchor`` is set as soon as the anchor is
        known so a failure here can still be recovered against it (see
        ``_recover_from_start_failure``).
        """
        durable_start, durable_timeout = self._durable_lifecycle()
        if durable_timeout:
            self._started = True
            self._timed_out = True
            return

        now = self._validated_clock(self._clock(), "clock")
        monotonic_now = self._validated_clock(self._monotonic(), "monotonic")
        anchor = self._resolve_anchor(durable_start, started_at, now)
        self._pending_anchor = anchor

        self._started = True
        self._arm(anchor, now, monotonic_now)
        if durable_start is None:
            self._sink(
                ToolEvent(
                    tool=RUN_START_TOOL,
                    kind=EventKind.RUN_START,
                    metadata={
                        "started_at_unix": anchor,
                        "max_duration_seconds": self._max_duration_seconds,
                    },
                )
            )
            # A concurrent persistent writer may have won the first-start
            # race. PersistentPipeline then restores that canonical prefix
            # instead of appending a second RUN_START; schedule from the
            # winner's durable anchor, not this process's stale reading.
            refreshed_start, refreshed_timeout = self._durable_lifecycle()
            if refreshed_timeout:
                self._timed_out = True
                return
            if refreshed_start is not None:
                self._pending_anchor = refreshed_start
                self._arm(refreshed_start, now, monotonic_now)
        if self._terminal():
            return

        assert self._deadline_monotonic is not None
        remaining = self._deadline_monotonic - self._validated_clock(self._monotonic(), "monotonic")
        self._fire_or_schedule(remaining)

    def _recover_committed_start(self) -> bool:
        """After a failed _commit_start, adopt a start/timeout it may have
        committed before raising. Returns True if fully handled (the caller
        must stop), False if nothing durable was found (fall through to
        reset + retry).
        """
        try:
            recovered_start, recovered_timeout = self._durable_lifecycle()
            if recovered_timeout:
                self._started = True
                self._timed_out = True
                return True
            if recovered_start is not None:
                recovered_now = self._validated_clock(self._clock(), "clock")
                recovered_monotonic = self._validated_clock(self._monotonic(), "monotonic")
                self._started = True
                recovered_remaining = self._arm(recovered_start, recovered_now, recovered_monotonic)
                if self._terminal():
                    return True
                self._fire_or_schedule(recovered_remaining)
                return True
        except Exception:
            self._logger.exception("RES-2 watchdog could not recover its committed lifecycle start")
        return False

    def _recover_from_start_failure(self, started_at: float | None, fallback: bool) -> None:
        self._start_attempts += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._logger.exception(
            "RES-2 watchdog start failed; checking for a committed lifecycle start"
        )
        # The downstream may have committed RUN_START before a later side
        # effect raised. Recover from that canonical fact instead of leaving
        # a durable start with no timer and no future event capable of
        # re-arming it.
        if self._recover_committed_start():
            return
        anchor = self._pending_anchor
        self._started = False
        self._started_at = None
        self._deadline_monotonic = None
        if callable(self._history) and self._start_attempts < 2:
            self._logger.error("RES-2 RUN_START rolled back; retrying the lifecycle sink once")
            self.start_run(started_at=anchor, fallback=fallback)
            return
        self._logger.error("RES-2 watchdog will retry at the next lifecycle event")

    def end_run(self) -> None:
        with self._lock:
            # With RES-2 disabled this wrapper must be observationally identical
            # to the historical callable sink, including across adapters that
            # reuse one attached sink for several root operations.
            if self._max_duration_seconds is None:
                return
            if self._ended:
                return
            self._expire_if_due()
            self._ended = True
            timer = self._timer
            self._timer = None
            if timer is not None:
                timer.cancel()

    def check_deadline(self) -> None:
        """Synchronously enforce a due deadline at an external runtime boundary."""
        with self._lock:
            self._expire_if_due()

    def _on_timeout(self) -> None:
        with self._lock:
            if self._ended or self._timed_out or self._terminal():
                return
            assert self._deadline_monotonic is not None
            remaining = self._deadline_monotonic - self._validated_clock(
                self._monotonic(), "monotonic"
            )
            if remaining > 0.0:
                # Defend the exact boundary even against a scheduler callback
                # delivered early: re-arm for the monotonic remainder instead
                # of injecting RUN_TIMEOUT before the configured ceiling.
                self._timer = self._scheduler.call_later(remaining, self._on_timeout)
                return
            self._timer = None
            assert self._max_duration_seconds is not None
            assert self._started_at is not None
            event = ToolEvent(
                tool=RUN_TIMEOUT_TOOL,
                kind=EventKind.RUN_TIMEOUT,
                ts=float(self._max_duration_seconds),
                metadata={
                    "started_at_unix": self._started_at,
                    "max_duration_seconds": self._max_duration_seconds,
                },
            )
            try:
                self._sink(event)
            except Exception:
                # Timer callbacks must never surface into the scheduler thread.
                # Retry once only when the event is demonstrably absent and the
                # run is still live. If the sink raised after committing, the
                # lifecycle history/terminal state proves that retrying would
                # duplicate the event.
                self._timeout_attempts += 1
                observed = False
                try:
                    _, observed = self._durable_lifecycle()
                except Exception:
                    self._logger.exception(
                        "RES-2 could not verify timeout persistence after sink failure"
                    )
                terminal = False
                try:
                    terminal = self._terminal()
                except Exception:
                    self._logger.exception(
                        "RES-2 could not verify terminal state after sink failure"
                    )
                if observed or terminal or self._timeout_attempts >= 2:
                    self._timed_out = True
                    self._logger.exception(
                        "RES-2 timeout sink failed; timeout retry is unsafe or exhausted"
                    )
                    return
                self._logger.exception("RES-2 timeout sink failed; retrying once")
                self._on_timeout()
            else:
                self._timed_out = True

    def _expire_if_due(self) -> None:
        """Inject timeout synchronously when a delayed scheduler is already due."""
        if (
            not self._started
            or self._ended
            or self._timed_out
            or self._deadline_monotonic is None
            or self._terminal()
        ):
            return
        now = self._validated_clock(self._monotonic(), "monotonic")
        if now >= self._deadline_monotonic:
            self._on_timeout()

    def _durable_lifecycle(self) -> tuple[float | None, bool]:
        events = self._history() if callable(self._history) else self._history
        start: float | None = None
        timed_out = False
        for event in events:
            if event.kind is EventKind.RUN_START and start is None:
                value = event.metadata.get("started_at_unix")
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError("persisted RUN_START has invalid started_at_unix")
                start = self._validated_clock(float(value), "persisted started_at_unix")
            elif event.kind is EventKind.RUN_TIMEOUT:
                timed_out = True
        return start, timed_out

    @staticmethod
    def _validated_clock(value: float, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must return a finite number")
        return float(value)
