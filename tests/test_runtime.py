"""RES-2 runtime watchdog tests with a deterministic scheduler."""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from pathlayer.adapters.base import RunAwareEventSink
from pathlayer.config import (
    DetectorsConfig,
    PathlayerConfig,
    PolicySection,
    ResourceConfig,
    StoreSection,
    build_pipeline,
)
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.resource import ResourceDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.policies.actuators import Actuator
from pathlayer.policies.wiring import wire
from pathlayer.postmortem import PostMortem
from pathlayer.runtime import WatchdogEventSink


@dataclass
class _Handle:
    due: float
    callback: Callable[[], None]
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class _ClockScheduler:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.monotonic_now = 0.0
        self.handles: list[_Handle] = []

    def clock(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.monotonic_now

    def call_later(self, delay_seconds: float, callback: Callable[[], None]) -> _Handle:
        handle = _Handle(self.monotonic_now + delay_seconds, callback)
        self.handles.append(handle)
        return handle

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.monotonic_now += seconds
        due = [
            handle
            for handle in self.handles
            if not handle.cancelled and handle.due <= self.monotonic_now
        ]
        for handle in due:
            handle.cancelled = True
            handle.callback()


class _Actuator(Actuator):
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        self.actions.append(action)


class _CoincidentalDurationDetector(Detector):
    max_duration_seconds = 3

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        _ = history
        return Signal("coincidental", 0.0)


def _wired(limit: int = 10) -> tuple[Pipeline, _Actuator, _ClockScheduler, RunAwareEventSink]:
    pipeline = Pipeline([ResourceDetector(max_duration_seconds=limit)], Policy())
    actuator = _Actuator()
    scheduler = _ClockScheduler()
    sink = wire(
        pipeline,
        actuator,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    return pipeline, actuator, scheduler, sink


def test_watchdog_has_no_signal_before_limit_and_stops_exactly_at_limit() -> None:
    pipeline, actuator, scheduler, sink = _wired()
    sink.start_run()

    # Even a misbehaving scheduler callback cannot cross the hard boundary
    # early; the sink re-arms from its monotonic deadline.
    scheduler.handles[0].cancelled = True
    scheduler.handles[0].callback()
    assert [event.kind for event in pipeline.history] == [EventKind.RUN_START]

    scheduler.advance(9.999)
    assert [event.kind for event in pipeline.history] == [EventKind.RUN_START]
    assert actuator.actions == []

    scheduler.advance(0.001)
    assert [event.kind for event in pipeline.history] == [
        EventKind.RUN_START,
        EventKind.RUN_TIMEOUT,
    ]
    assert pipeline.last_decision is not None
    assert pipeline.last_decision.signals[-1].confidence == 1.0
    assert actuator.actions == [Action.STOP]


def test_watchdog_starts_once_and_expires_without_intermediate_event() -> None:
    pipeline, _, scheduler, sink = _wired()
    sink.start_run()
    sink.start_run()
    scheduler.advance(10)
    scheduler.advance(100)
    assert [event.kind for event in pipeline.history].count(EventKind.RUN_START) == 1
    assert [event.kind for event in pipeline.history].count(EventKind.RUN_TIMEOUT) == 1


def test_wire_ignores_third_party_attribute_collision() -> None:
    scheduler = _ClockScheduler()
    pipeline = Pipeline([_CoincidentalDurationDetector()], Policy())
    sink = wire(
        pipeline,
        _Actuator(),
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )

    sink.start_run()
    assert scheduler.handles == []
    assert pipeline.history == ()


def test_pipeline_rejects_multiple_runtime_limit_owners_at_construction() -> None:
    # Fail fast at construction (see Pipeline.__init__), not lazily the first
    # time something reads max_duration_seconds -- a misconfigured
    # pipeline_factory (no build_pipeline validation upstream) must not crash
    # mid-run instead of at startup.
    with pytest.raises(ValueError, match="only one runtime-limit provider"):
        Pipeline([ResourceDetector(), ResourceDetector()], Policy())


def test_build_pipeline_timeout_stops_on_one_tick_despite_policy_hysteresis() -> None:
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10)),
        policy=PolicySection(escalate_ticks=3, weights={"resource": 0.01}),
    )
    pipeline = build_pipeline(config)
    actuator = _Actuator()
    scheduler = _ClockScheduler()
    sink = wire(
        pipeline,
        actuator,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )

    sink.start_run()
    scheduler.advance(10)

    assert pipeline.stopped
    assert actuator.actions == [Action.STOP]
    assert [event.kind for event in pipeline.history].count(EventKind.RUN_TIMEOUT) == 1


def test_end_run_cancels_timer_and_closed_run_cannot_reopen() -> None:
    pipeline, _, scheduler, sink = _wired()
    sink.start_run()
    sink.end_run()
    sink.start_run()
    sink(ToolEvent("late"))
    scheduler.advance(100)
    assert [event.kind for event in pipeline.history] == [EventKind.RUN_START]


@pytest.mark.parametrize("finish", ["end", "event"])
def test_due_deadline_expires_even_when_scheduler_callback_is_delayed(finish: str) -> None:
    pipeline, actuator, scheduler, sink = _wired()
    sink.start_run()
    scheduler.now = 111.0
    scheduler.monotonic_now = 11.0

    if finish == "end":
        sink.end_run()
    else:
        sink(ToolEvent("late"))

    assert [event.kind for event in pipeline.history] == [
        EventKind.RUN_START,
        EventKind.RUN_TIMEOUT,
    ]
    assert actuator.actions == [Action.STOP]


def test_first_event_fallback_warns_and_starts_watchdog(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline, _, _, sink = _wired()
    with caplog.at_level(logging.WARNING):
        sink(ToolEvent("read"))
    assert [event.kind for event in pipeline.history] == [EventKind.RUN_START, EventKind.TOOL_CALL]
    assert "earlier duration was not measured" in caplog.text


def test_start_accounts_for_time_spent_persisting_run_start() -> None:
    scheduler = _ClockScheduler()
    seen: list[EventKind] = []

    def slow_sink(event: ToolEvent) -> None:
        seen.append(event.kind)
        if event.kind is EventKind.RUN_START:
            scheduler.now += 11.0
            scheduler.monotonic_now += 11.0

    sink = WatchdogEventSink(
        slow_sink,
        max_duration_seconds=10,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )

    sink.start_run()

    assert seen == [EventKind.RUN_START, EventKind.RUN_TIMEOUT]
    assert scheduler.handles == []


def test_failed_start_is_contained_and_next_start_retries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler = _ClockScheduler()
    seen: list[EventKind] = []
    fail = True

    def flaky_sink(event: ToolEvent) -> None:
        nonlocal fail
        if event.kind is EventKind.RUN_START and fail:
            fail = False
            raise RuntimeError("start write failed")
        seen.append(event.kind)

    sink = WatchdogEventSink(
        flaky_sink,
        max_duration_seconds=10,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )

    with caplog.at_level(logging.ERROR):
        sink.start_run()
        sink.start_run()

    assert seen == [EventKind.RUN_START]
    assert len(scheduler.handles) == 1
    assert "retry at the next lifecycle event" in caplog.text


def test_start_failure_after_commit_recovers_timer_without_another_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler = _ClockScheduler()
    history: list[ToolEvent] = []
    fail_after_commit = True

    def partial_sink(event: ToolEvent) -> None:
        nonlocal fail_after_commit
        history.append(event)
        if event.kind is EventKind.RUN_START and fail_after_commit:
            fail_after_commit = False
            raise RuntimeError("decision journal failed after lifecycle commit")

    sink = WatchdogEventSink(
        partial_sink,
        max_duration_seconds=10,
        history=history,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )

    with caplog.at_level(logging.ERROR):
        sink.start_run()
    assert len(scheduler.handles) == 1

    scheduler.advance(10)
    assert [event.kind for event in history] == [
        EventKind.RUN_START,
        EventKind.RUN_TIMEOUT,
    ]


def test_timeout_callback_failure_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    scheduler = _ClockScheduler()
    seen: list[EventKind] = []
    history: list[ToolEvent] = []

    def failing_sink(event: ToolEvent) -> None:
        seen.append(event.kind)
        history.append(event)
        if event.kind is EventKind.RUN_TIMEOUT:
            raise RuntimeError("boom")

    sink = WatchdogEventSink(
        failing_sink,
        max_duration_seconds=10,
        history=history,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    sink.start_run()
    with caplog.at_level(logging.ERROR):
        scheduler.advance(10)
        scheduler.advance(10)
    assert seen == [EventKind.RUN_START, EventKind.RUN_TIMEOUT]
    assert "retry is unsafe or exhausted" in caplog.text


def test_timeout_retries_once_when_failed_event_was_rolled_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler = _ClockScheduler()
    history: list[ToolEvent] = []
    attempts = 0

    def flaky_sink(event: ToolEvent) -> None:
        nonlocal attempts
        if event.kind is EventKind.RUN_TIMEOUT:
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient detector failure")
        history.append(event)

    sink = WatchdogEventSink(
        flaky_sink,
        max_duration_seconds=10,
        history=history,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    sink.start_run()

    with caplog.at_level(logging.ERROR):
        scheduler.advance(10)

    assert attempts == 2
    assert [event.kind for event in history].count(EventKind.RUN_TIMEOUT) == 1
    assert "retrying once" in caplog.text


def test_timer_and_external_event_are_serialized_and_timeout_stays_unique() -> None:
    scheduler = _ClockScheduler()
    gate = threading.Barrier(2)
    probe_lock = threading.Lock()
    active_callbacks = 0
    overlap = False
    seen: list[EventKind] = []

    def downstream(event: ToolEvent) -> None:
        nonlocal active_callbacks, overlap
        with probe_lock:
            active_callbacks += 1
            overlap = overlap or active_callbacks > 1
        # Enough local work to make an unlocked overlap observable without a
        # wall-clock sleep in the test.
        sum(range(20_000))
        seen.append(event.kind)
        with probe_lock:
            active_callbacks -= 1

    sink = WatchdogEventSink(
        downstream,
        max_duration_seconds=10,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    sink.start_run()
    assert scheduler.handles
    scheduler.monotonic_now = 10.0

    def fire_timeout() -> None:
        gate.wait()
        scheduler.handles[0].callback()

    def send_event() -> None:
        gate.wait()
        sink(ToolEvent("read"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda fn: fn(), (fire_timeout, send_event)))
    scheduler.handles[0].callback()

    assert not overlap
    assert seen.count(EventKind.RUN_TIMEOUT) == 1


@pytest.mark.parametrize(
    ("now", "expected_delay", "timed_out"),
    [(105.0, 5.0, False), (111.0, None, True)],
)
def test_resume_uses_first_durable_start_and_expires_if_overdue(
    now: float, expected_delay: float | None, timed_out: bool
) -> None:
    scheduler = _ClockScheduler(now=now)
    events = [
        ToolEvent(
            "pathlayer_run_start",
            kind=EventKind.RUN_START,
            metadata={"started_at_unix": 100.0, "max_duration_seconds": 10},
        )
    ]
    sink = WatchdogEventSink(
        events.append,
        max_duration_seconds=10,
        history=events,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    sink.start_run()
    assert [event.kind for event in events].count(EventKind.RUN_START) == 1
    assert (EventKind.RUN_TIMEOUT in [event.kind for event in events]) is timed_out
    if expected_delay is not None:
        assert scheduler.handles[-1].due == expected_delay


@pytest.mark.parametrize(("resume_at", "delay"), [(105.0, 5.0), (111.0, 0.0)])
def test_persistent_watchdog_resumes_original_deadline(
    tmp_path: Path, resume_at: float, delay: float
) -> None:
    database = str(tmp_path / "state.db")
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10)),
        store=StoreSection(path=database),
    )
    first_clock = _ClockScheduler(now=100.0)
    first = build_pipeline(config, state_key="run")
    first_sink = wire(
        first,
        _Actuator(),
        clock=first_clock.clock,
        monotonic=first_clock.monotonic,
        scheduler=first_clock,
    )
    first_sink.start_run()

    resumed_clock = _ClockScheduler(now=resume_at)
    resumed = build_pipeline(config, state_key="run")
    actuator = _Actuator()
    resumed_sink = wire(
        resumed,
        actuator,
        clock=resumed_clock.clock,
        monotonic=resumed_clock.monotonic,
        scheduler=resumed_clock,
    )
    resumed_sink.start_run()
    if delay:
        assert resumed_clock.handles[-1].due == delay
        resumed_clock.advance(delay)
    assert resumed.stopped
    assert actuator.actions == [Action.STOP]
    assert [event.kind for event in resumed.history].count(EventKind.RUN_START) == 1
    assert [event.kind for event in resumed.history].count(EventKind.RUN_TIMEOUT) == 1


def test_concurrent_persistent_starts_keep_one_canonical_anchor(tmp_path: Path) -> None:
    database = str(tmp_path / "state.db")
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10)),
        store=StoreSection(path=database),
    )
    winner = build_pipeline(config, state_key="run")
    stale = build_pipeline(config, state_key="run")
    winner_clock = _ClockScheduler(now=100.0)
    stale_clock = _ClockScheduler(now=105.0)
    winner_sink = wire(
        winner,
        _Actuator(),
        clock=winner_clock.clock,
        monotonic=winner_clock.monotonic,
        scheduler=winner_clock,
    )
    stale_sink = wire(
        stale,
        _Actuator(),
        clock=stale_clock.clock,
        monotonic=stale_clock.monotonic,
        scheduler=stale_clock,
    )

    winner_sink.start_run()
    stale_sink.start_run()

    assert [event.kind for event in stale.history].count(EventKind.RUN_START) == 1
    assert stale.history[0].metadata["started_at_unix"] == 100.0
    assert stale_clock.handles[-1].due == 5.0


def test_first_persistent_append_failure_still_allows_memory_only_timeout(
    tmp_path: Path,
) -> None:
    database = str(tmp_path / "state.db")
    from pathlayer.store import SQLiteStateStore

    prepared = SQLiteStateStore(database)
    prepared.close()
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TRIGGER reject_tick BEFORE INSERT ON ticks
           BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END"""
    )
    connection.commit()
    connection.close()

    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True, max_duration_seconds=10)),
        store=StoreSection(path=database),
    )
    scheduler = _ClockScheduler(now=100.0)
    pipeline = build_pipeline(config, state_key="run")
    actuator = _Actuator()
    sink = wire(
        pipeline,
        actuator,
        clock=scheduler.clock,
        monotonic=scheduler.monotonic,
        scheduler=scheduler,
    )
    sink.start_run()
    assert not pipeline.durable

    scheduler.advance(10)
    assert pipeline.stopped
    assert actuator.actions == [Action.STOP]
