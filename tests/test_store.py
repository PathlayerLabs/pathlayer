"""INF-1 durable-state behavior and failure-containment tests."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import pathlayer.store as store_module
from pathlayer.config import (
    BudgetConfig,
    DetectorsConfig,
    PathlayerConfig,
    PipelineSection,
    PolicySection,
    ResourceConfig,
    StoreSection,
    build_pipeline,
)
from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.budget import BudgetState
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action
from pathlayer.store import (
    EventSerializationError,
    PersistentPipeline,
    SQLiteStateStore,
    StateClosedError,
    StateCompatibilityError,
    StateConcurrencyError,
    StateCorruptionError,
    StateIncompleteError,
    StateNotFoundError,
    StateStore,
    StateWriteError,
    configuration_fingerprint,
    decode_event,
    encode_event,
)


def _config(path: Path, *, budget: BudgetConfig | None = None) -> PathlayerConfig:
    return PathlayerConfig(
        detectors=DetectorsConfig(budget=budget or BudgetConfig()),
        store=StoreSection(path=str(path)),
    )


def _event(index: int, *, cost: float = 0.0) -> ToolEvent:
    return ToolEvent(
        "read",
        {"path": "a.py", "position": (index, index + 1)},
        result={"error": "missing"} if index % 2 else {"ok": True},
        cost_usd=cost,
        ts=float(index),
        run_id="run-1",
        metadata={"nested": [1, None, math.inf]},
    )


def _initialize_state_process(database: str, barrier: Any, queue: Any) -> None:
    try:
        barrier.wait(timeout=10)
        store = SQLiteStateStore(database)
        store.close()
        queue.put(None)
    except Exception as error:  # pragma: no cover - reported to the parent process
        queue.put(repr(error))


def _write_state_process(database: str, event_index: int, barrier: Any, queue: Any) -> None:
    try:
        config = _config(Path(database))
        pipeline = build_pipeline(config, state_key="shared")
        barrier.wait(timeout=10)
        action = pipeline.process(_event(event_index), event_id=str(event_index))
        queue.put((action.name, pipeline.durable, None))
    except Exception as error:  # pragma: no cover - reported to the parent process
        queue.put((None, None, repr(error)))


def test_event_codec_round_trips_supported_values() -> None:
    event = _event(2, cost=math.nan)
    restored = decode_event(encode_event(event))

    assert restored.tool == event.tool
    assert restored.args == event.args
    assert restored.result == event.result
    assert math.isnan(restored.cost_usd)
    assert restored.metadata == event.metadata


def test_event_codec_rejects_arbitrary_objects() -> None:
    with pytest.raises(EventSerializationError, match="unsupported value"):
        encode_event(ToolEvent("read", result=object()))


def test_invalid_runtime_cost_value_survives_replay_for_budget_diagnostics(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path / "state.db",
        budget=BudgetConfig(enabled=True, max_cost_usd=10.0, default_cost_usd=0.1),
    )
    pipeline = build_pipeline(config, state_key="invalid-cost")
    event = ToolEvent("read", cost_usd=True)
    pipeline.process(event)

    resumed = build_pipeline(config, state_key="invalid-cost")
    assert resumed.history[0].cost_usd is True
    budget = next(signal for signal in resumed.last_decision.signals if signal.name == "budget")
    assert "invalid event cost" in budget.diagnostics[0]


def test_build_without_state_key_does_not_create_database(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    pipeline = build_pipeline(_config(database))

    pipeline.process(_event(0))

    assert not database.exists()


def test_restart_replays_exact_history_and_decisions(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    config = _config(database)
    events = [_event(index) for index in range(5)]
    uninterrupted = build_pipeline(config).run(events)

    first = build_pipeline(config, state_key="same-run")
    assert first.run(events[:3]) == uninterrupted[:3]
    resumed = build_pipeline(config, state_key="same-run")

    assert resumed.history == tuple(events[:3])
    assert resumed.run(events[3:]) == uninterrupted[3:]
    assert resumed.actions == tuple(uninterrupted)
    assert resumed.last_decision is not None
    assert resumed.last_decision.index == 4


def test_event_id_is_idempotent_across_processes(tmp_path: Path) -> None:
    config = _config(tmp_path / "state.db")
    first = build_pipeline(config, state_key="run")
    action = first.process(_event(0), event_id="hook:1")
    resumed = build_pipeline(config, state_key="run")

    assert action is Action.NONE
    assert resumed.process(_event(0), event_id="hook:1") is Action.NONE
    assert len(resumed.history) == 1


@pytest.mark.parametrize("action", [Action.WARN, Action.STOP])
def test_duplicate_returns_stored_action_and_context(tmp_path: Path, action: Action) -> None:
    policy = (
        PolicySection(tier_thresholds={"warn": 0.0})
        if action is Action.WARN
        else PolicySection(stop_threshold=0.0)
    )
    config = PathlayerConfig(
        policy=policy,
        store=StoreSection(path=str(tmp_path / f"{action.name}.db")),
    )
    event = ToolEvent("read", {"path": "same.py"}, cost_usd=0.5)
    first = build_pipeline(config, state_key="run")
    assert first.process(event, event_id="stable") is action
    expected_decision = first.last_decision

    resumed = build_pipeline(config, state_key="run")
    assert resumed.process(event, event_id="stable") is action
    assert len(resumed.history) == 1
    assert resumed.last_decision == expected_decision
    assert resumed.actuation_context is not None
    assert resumed.actuation_context.action is action
    assert resumed.actuation_context.reason == expected_decision.reason
    assert resumed.actuation_context.event.cost_usd == 0.5


def test_event_id_collision_with_different_payload_is_refused(tmp_path: Path) -> None:
    config = _config(tmp_path / "state.db")
    pipeline = build_pipeline(config, state_key="run")
    pipeline.process(_event(0), event_id="same-id")

    with pytest.raises(StateConcurrencyError, match="different event"):
        pipeline.process(_event(1), event_id="same-id")

    assert len(pipeline.history) == 1


def test_corrupt_duplicate_payload_fails_open_without_leaking_store_error(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    event = _event(0)
    pipeline.process(event, event_id="same-id")
    connection = sqlite3.connect(database)
    connection.execute("UPDATE events SET payload_json = '{'")
    connection.commit()
    connection.close()

    assert pipeline.process(event, event_id="same-id") is Action.NONE
    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateCorruptionError)
    assert pipeline.history == (event, event)


def test_corrupt_signal_index_is_typed_and_duplicate_fails_open(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    event = _event(0)
    pipeline.process(event, event_id="same-id")
    connection = sqlite3.connect(database)
    connection.execute("UPDATE signals SET signal_index = 'broken' WHERE signal_index = 0")
    connection.commit()
    connection.close()

    inspector = SQLiteStateStore(database)
    with pytest.raises(StateCorruptionError, match="signal"):
        inspector.read_scope("run")
    inspector.close()

    assert pipeline.process(event, event_id="same-id") is Action.NONE
    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateCorruptionError)
    assert pipeline.history == (event, event)


@pytest.mark.parametrize("column", ["revision", "last_event_index"])
def test_corrupt_scope_counter_during_process_fails_open(tmp_path: Path, column: str) -> None:
    database = tmp_path / f"{column}.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    pipeline.process(_event(0))
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(f"UPDATE scopes SET {column} = 'broken' WHERE state_key = 'run'")
    connection.commit()
    connection.close()

    assert pipeline.process(_event(1)) is Action.NONE
    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateCorruptionError)
    assert pipeline.history == (_event(0), _event(1))


def test_corrupt_scope_revision_during_close_fails_open(tmp_path: Path) -> None:
    database = tmp_path / "close.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    pipeline.process(_event(0))
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute("UPDATE scopes SET revision = 'broken' WHERE state_key = 'run'")
    connection.commit()
    connection.close()

    pipeline.close("completed")

    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateCorruptionError)
    connection = sqlite3.connect(database)
    status, durable = connection.execute(
        "SELECT status, durable FROM scopes WHERE state_key = 'run'"
    ).fetchone()
    connection.close()
    assert status == "active"
    assert durable == 0


@pytest.mark.parametrize("field", ["event_index", "emitted_action"])
def test_corrupt_duplicate_discovered_during_append_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    database = tmp_path / f"duplicate-{field}.db"
    store = SQLiteStateStore(database)
    pipeline = build_pipeline(_config(database), state_key="run", store=store)
    event = _event(0)
    pipeline.process(event, event_id="same-id")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA ignore_check_constraints = ON")
    if field == "event_index":
        for table in ("events", "ticks", "signals"):
            connection.execute(f"UPDATE {table} SET event_index = 'broken'")
    else:
        connection.execute("UPDATE ticks SET emitted_action = 'BROKEN'")
    connection.commit()
    connection.close()
    monkeypatch.setattr(store, "lookup_event", lambda _state_key, _event_id: None)

    assert pipeline.process(event, event_id="same-id") is Action.NONE
    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateCorruptionError)
    assert pipeline.history == (event, event)


def test_concurrent_stale_pipeline_reloads_before_commit(tmp_path: Path) -> None:
    config = _config(tmp_path / "state.db")
    first = build_pipeline(config, state_key="run")
    stale = build_pipeline(config, state_key="run")

    first.process(_event(0), event_id="first")
    stale.process(_event(1), event_id="second")

    assert [event.ts for event in stale.history] == [0.0, 1.0]
    assert len(build_pipeline(config, state_key="run").history) == 2


def test_concurrent_stop_wins_without_appending_stale_event(tmp_path: Path) -> None:
    config = PathlayerConfig(
        policy=PolicySection(stop_threshold=0.0),
        store=StoreSection(path=str(tmp_path / "state.db")),
    )
    winner = build_pipeline(config, state_key="run")
    stale = build_pipeline(config, state_key="run")

    assert winner.process(ToolEvent("read"), event_id="winner") is Action.STOP
    assert stale.process(ToolEvent("edit"), event_id="stale") is Action.STOP
    assert stale.stopped
    assert stale.durable
    assert len(stale.history) == 1
    snapshot = SQLiteStateStore(tmp_path / "state.db").read_scope("run")
    assert snapshot.status == "stopped"
    assert snapshot.durable


def test_concurrent_close_fails_open_and_poisons_terminal_scope(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    config = _config(database)
    active = build_pipeline(config, state_key="run")
    closer = build_pipeline(config, state_key="run")
    closer.close("interrupted")

    event = ToolEvent("read", ts=1.0)
    assert active.process(event, event_id="late") is Action.NONE
    assert active.history == (event,)
    assert not active.durable
    assert isinstance(active.persistence_error, StateClosedError)

    snapshot = SQLiteStateStore(database).read_scope("run")
    assert snapshot.status == "interrupted"
    assert not snapshot.durable
    assert not snapshot.complete
    assert snapshot.ticks == ()
    assert active.process(ToolEvent("edit", ts=2.0), event_id="memory") is Action.NONE
    active.close("completed")
    assert SQLiteStateStore(database).read_scope("run").status == "interrupted"


def test_configuration_change_refuses_resume(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    build_pipeline(_config(database), state_key="run").process(_event(0))
    changed = _config(
        database,
        budget=BudgetConfig(enabled=True, max_cost_usd=50.0),
    )

    with pytest.raises(StateCompatibilityError, match="different configuration"):
        build_pipeline(changed, state_key="run")


def test_every_restart_prefix_matches_uninterrupted_progress_state(tmp_path: Path) -> None:
    events = [
        ToolEvent(
            "pytest",
            {"target": "tests"},
            result={"passed": passed, "failed": 5 - passed},
            ts=float(passed),
        )
        for passed in range(1, 6)
    ]
    reference = build_pipeline()
    expected: list[tuple[Action, Any]] = []
    for event in events:
        action = reference.process(event)
        expected.append((action, reference.last_decision))

    for split in range(len(events) + 1):
        config = _config(tmp_path / f"prefix-{split}.db")
        before = build_pipeline(config, state_key="run")
        for index, event in enumerate(events[:split]):
            assert before.process(event, event_id=str(index)) is expected[index][0]
        resumed = build_pipeline(config, state_key="run")
        assert resumed.history == tuple(events[:split])
        if split:
            assert resumed.last_decision == expected[split - 1][1]
        for index, event in enumerate(events[split:], start=split):
            assert resumed.process(event, event_id=str(index)) is expected[index][0]
            assert resumed.last_decision == expected[index][1]


def test_dry_run_would_be_state_survives_restart(tmp_path: Path) -> None:
    config = PathlayerConfig(
        pipeline=PipelineSection(dry_run=True),
        policy=PolicySection(stop_threshold=0.0),
        store=StoreSection(path=str(tmp_path / "state.db")),
    )
    first = build_pipeline(config, state_key="dry")
    assert first.process(ToolEvent("read"), event_id="first") is Action.NONE
    expected_decision = first.last_decision

    resumed = build_pipeline(config, state_key="dry")
    assert resumed.dry_run
    assert not resumed.stopped
    assert resumed.last_decision == expected_decision
    assert resumed.would_be_actions == first.would_be_actions


def test_terminal_scope_cannot_contaminate_a_new_run(tmp_path: Path) -> None:
    config = _config(tmp_path / "state.db")
    completed = build_pipeline(config, state_key="completed")
    completed.process(_event(0))
    completed.close("completed")
    completed.close("completed")

    with pytest.raises(StateClosedError, match="is closed"):
        completed.process(_event(1))

    with pytest.raises(StateClosedError, match="terminal"):
        build_pipeline(config, state_key="completed")
    assert build_pipeline(config, state_key="new-run").history == ()


def test_stopped_scope_rebuilds_as_stopped_without_accepting_more_events(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    config = _config(database)
    pipeline = build_pipeline(config, state_key="stopped")
    for index in range(8):
        action = pipeline.process(
            ToolEvent("read", {"path": "same.py"}, ts=float(index)),
            event_id=str(index),
        )
        if action is Action.STOP:
            break
    assert pipeline.stopped
    count = len(pipeline.history)

    resumed = build_pipeline(config, state_key="stopped")
    assert resumed.stopped
    assert resumed.post_mortem == pipeline.post_mortem
    assert resumed.process(ToolEvent("edit"), event_id="after-stop") is Action.STOP
    assert len(resumed.history) == count


def test_cost_counters_track_captured_estimated_and_invalid(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    config = _config(
        database,
        budget=BudgetConfig(
            enabled=True,
            max_cost_usd=100.0,
            default_cost_usd=0.25,
            tool_costs_usd={"read": 0.5},
        ),
    )
    pipeline = build_pipeline(config, state_key="costs")
    pipeline.process(_event(0, cost=2.0))
    pipeline.process(_event(1, cost=0.0))
    pipeline.process(_event(2, cost=math.nan))

    connection = sqlite3.connect(database)
    row = connection.execute(
        "SELECT captured_usd, priced_events, estimated_usd, invalid_costs, effective_usd "
        "FROM cost_counters WHERE state_key='costs'"
    ).fetchone()
    connection.close()

    assert row == pytest.approx((2.0, 1, 1.0, 1, 3.0))


def test_disabled_budget_tracks_actuals_without_estimates(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    config = _config(
        database,
        budget=BudgetConfig(
            enabled=False,
            default_cost_usd=0.25,
            tool_costs_usd={"read": 0.5},
        ),
    )
    pipeline = build_pipeline(config, state_key="costs")
    pipeline.process(_event(0, cost=2.0))
    pipeline.process(_event(1, cost=0.0))
    pipeline.process(_event(2, cost=math.nan))

    snapshot = SQLiteStateStore(database).read_scope("costs")
    assert snapshot.costs == BudgetState(captured_usd=2.0, priced_events=1, invalid_costs=1)


def test_signal_and_tick_snapshots_are_written_atomically(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    pipeline.process(_event(0))

    connection = sqlite3.connect(database)
    counts = tuple(
        connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("events", "ticks", "signals")
    )
    connection.close()

    assert counts[0:2] == (1, 1)
    assert counts[2] == len(pipeline.last_decision.signals)


@pytest.mark.parametrize(
    "change",
    [
        {"fused": math.nan},
        {"fused": 1.1},
        {"ts": 99.0},
        {"reason": "invented"},
    ],
)
def test_invalid_decision_is_refused_before_any_write(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    store.load_or_create("run", "fingerprint")
    event = ToolEvent("read", ts=1.0)
    pipeline = Pipeline([])
    pipeline.process(event)
    decision = pipeline.last_decision
    assert decision is not None

    with pytest.raises(EventSerializationError, match="decision cannot be serialized"):
        store.append_tick(
            "run",
            "fingerprint",
            0,
            None,
            event,
            replace(decision, **change),
            BudgetState(),
        )

    assert store.read_scope("run").ticks == ()


def test_public_scope_snapshot_is_complete_and_immutable(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    config = PathlayerConfig(
        policy=PolicySection(tier_thresholds={"warn": 0.0}),
        store=StoreSection(path=str(database)),
    )
    pipeline = build_pipeline(config, state_key="run")
    event = _event(0, cost=1.25)
    assert pipeline.process(event, event_id="stable") is Action.WARN

    snapshot = SQLiteStateStore(database).read_scope("run")
    assert snapshot.status == "active"
    assert snapshot.revision == 1
    assert snapshot.durable
    assert not snapshot.complete
    assert snapshot.costs.captured_usd == 1.25
    tick = snapshot.ticks[0]
    assert tick.event_id == "stable"
    assert tick.reason == pipeline.last_decision.reason
    assert tick.arbitrated_action is Action.WARN
    assert tick.emitted_action is Action.WARN
    assert tick.signals
    assert tick.signals[0].diagnostics == pipeline.last_decision.signals[0].diagnostics
    with pytest.raises(TypeError):
        tick.evidence["mutate"] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        tick.reason = "mutate"  # type: ignore[misc]
    mutable_copy = tick.event
    mutable_copy.args["path"] = "changed.py"
    assert tick.event.args["path"] == "a.py"
    pipeline.close("completed")
    completed = SQLiteStateStore(database).read_scope("run")
    assert completed.durable
    assert completed.complete


def test_reading_unknown_scope_has_a_dedicated_error(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")

    with pytest.raises(StateNotFoundError, match="missing scope"):
        store.read_scope("unknown")


def test_mark_nondurable_is_idempotent_and_prevents_resume(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    store.load_or_create("run", "fingerprint")

    store.mark_nondurable("run", "fingerprint")
    store.mark_nondurable("run", "fingerprint")

    snapshot = store.read_scope("run")
    assert not snapshot.durable
    assert not snapshot.complete
    with pytest.raises(StateIncompleteError, match="non-durable journal prefix"):
        store.load_or_create("run", "fingerprint")


def test_fingerprint_ignores_future_non_decision_sections(tmp_path: Path) -> None:
    config = _config(tmp_path / "state.db")
    projected = SimpleNamespace(
        detectors=config.detectors,
        canonicalization=config.canonicalization,
        policy=config.policy,
        pipeline=config.pipeline,
        store=config.store,
        observability={"export": "otlp"},
        pricing={"source": "remote"},
    )

    expected = configuration_fingerprint(config)
    assert configuration_fingerprint(projected) == expected
    changed = SimpleNamespace(**vars(projected))
    changed.pipeline = PipelineSection(dry_run=True)
    assert configuration_fingerprint(changed) != configuration_fingerprint(config)


def test_resource_duration_only_changes_fingerprint_when_resource_is_enabled(
    tmp_path: Path,
) -> None:
    disabled_a = _config(tmp_path / "a.db")
    disabled_b = replace(
        disabled_a,
        detectors=replace(
            disabled_a.detectors,
            resource=ResourceConfig(enabled=False, max_duration_seconds=7),
        ),
    )
    assert configuration_fingerprint(disabled_a) == configuration_fingerprint(disabled_b)

    enabled_a = replace(
        disabled_a,
        detectors=replace(
            disabled_a.detectors,
            resource=ResourceConfig(enabled=True, max_duration_seconds=7),
        ),
    )
    enabled_b = replace(
        enabled_a,
        detectors=replace(
            enabled_a.detectors,
            resource=ResourceConfig(enabled=True, max_duration_seconds=8),
        ),
    )
    assert configuration_fingerprint(enabled_a) != configuration_fingerprint(enabled_b)


def test_resource_max_rss_bytes_only_changes_fingerprint_when_resource_is_enabled(
    tmp_path: Path,
) -> None:
    disabled_a = _config(tmp_path / "a.db")
    disabled_b = replace(
        disabled_a,
        detectors=replace(
            disabled_a.detectors,
            resource=ResourceConfig(enabled=False, max_rss_bytes=7),
        ),
    )
    assert configuration_fingerprint(disabled_a) == configuration_fingerprint(disabled_b)

    enabled_a = replace(
        disabled_a,
        detectors=replace(
            disabled_a.detectors,
            resource=ResourceConfig(enabled=True, max_rss_bytes=7),
        ),
    )
    enabled_b = replace(
        enabled_a,
        detectors=replace(
            enabled_a.detectors,
            resource=ResourceConfig(enabled=True, max_rss_bytes=8),
        ),
    )
    assert configuration_fingerprint(enabled_a) != configuration_fingerprint(enabled_b)


def test_previous_state_semantics_scope_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    store = SQLiteStateStore(database)
    historical = "b15037528988ec774f889c83bcf27254ea2f69f23bdfc4b33177b3c9471c8a3b"
    store.load_or_create("run", historical)

    with pytest.raises(StateCompatibilityError, match="different configuration"):
        build_pipeline(_config(database), state_key="run", store=store)


def test_v3_scope_created_with_resource_enabled_is_refused_conservatively(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    store = SQLiteStateStore(database)
    store.load_or_create("run", "pre-res2-resource-fingerprint")
    config = PathlayerConfig(
        detectors=DetectorsConfig(resource=ResourceConfig(enabled=True)),
        store=StoreSection(path=str(database)),
    )

    with pytest.raises(StateCompatibilityError, match="different configuration"):
        build_pipeline(config, state_key="run", store=store)


def test_v1_schema_migrates_action_reason_transactionally(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    store = SQLiteStateStore(database)
    store.close()
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE ticks DROP COLUMN action_reason")
    connection.execute("ALTER TABLE ticks DROP COLUMN event_ts")
    connection.execute("ALTER TABLE cost_counters DROP COLUMN priced_events")
    connection.execute("ALTER TABLE scopes DROP COLUMN durable")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    migrated = SQLiteStateStore(database)
    migrated.close()
    connection = sqlite3.connect(database)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(ticks)").fetchall()}
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    connection.close()
    assert "action_reason" in columns
    assert version == 3


def test_v2_schema_backfills_observability_and_conservative_durability(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    SQLiteStateStore(database).close()
    connection = sqlite3.connect(database)
    for state_key, status, event in (
        ("active", "active", ToolEvent("paid", cost_usd=1.0, ts=math.inf)),
        ("done", "completed", ToolEvent("free", cost_usd=0.0, ts=2.0)),
    ):
        connection.execute(
            "INSERT INTO scopes(state_key, fingerprint, status, revision, "
            "last_event_index, durable) VALUES (?, 'fingerprint', ?, 1, 0, 1)",
            (state_key, status),
        )
        connection.execute(
            "INSERT INTO events VALUES (?, 0, NULL, ?)",
            (state_key, encode_event(event)),
        )
        connection.execute(
            "INSERT INTO ticks(state_key, event_index, event_ts, fused, evidence_json, "
            "arbitrated_action, emitted_action, dry_run, action_reason) "
            "VALUES (?, 0, 0, 0, '{}', 'NONE', 'NONE', 0, 'loop')",
            (state_key,),
        )
        connection.execute(
            "INSERT INTO cost_counters(state_key, captured_usd, priced_events, "
            "estimated_usd, invalid_costs, effective_usd) VALUES (?, ?, 0, 0, 0, ?)",
            (state_key, event.cost_usd, event.cost_usd),
        )
    connection.execute("ALTER TABLE ticks DROP COLUMN event_ts")
    connection.execute("ALTER TABLE cost_counters DROP COLUMN priced_events")
    connection.execute("ALTER TABLE scopes DROP COLUMN durable")
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()

    store = SQLiteStateStore(database)
    active = store.read_scope("active")
    done = store.read_scope("done")

    assert not active.durable
    assert not active.complete
    assert math.isinf(active.ticks[0].ts)
    assert active.ticks[0].reason == ""
    assert active.costs.priced_events == 1
    assert done.durable
    assert done.complete
    assert done.ticks[0].ts == 2.0
    assert done.costs.priced_events == 0


def test_v2_migration_quarantines_only_the_corrupt_scope(tmp_path: Path) -> None:
    database = tmp_path / "mixed-v2.db"
    SQLiteStateStore(database).close()
    connection = sqlite3.connect(database)
    for state_key, payload in (
        ("healthy", encode_event(ToolEvent("paid", cost_usd=2.0, ts=7.0))),
        ("corrupt", "{"),
    ):
        connection.execute(
            "INSERT INTO scopes(state_key, fingerprint, status, revision, "
            "last_event_index, durable) VALUES (?, 'fingerprint', 'completed', 1, 0, 1)",
            (state_key,),
        )
        connection.execute(
            "INSERT INTO events VALUES (?, 0, NULL, ?)",
            (state_key, payload),
        )
        connection.execute(
            "INSERT INTO ticks(state_key, event_index, event_ts, fused, evidence_json, "
            "arbitrated_action, emitted_action, dry_run, action_reason) "
            "VALUES (?, 0, 0, 0, '{}', 'NONE', 'NONE', 0, 'stale')",
            (state_key,),
        )
        connection.execute(
            "INSERT INTO cost_counters(state_key, captured_usd, priced_events, "
            "estimated_usd, invalid_costs, effective_usd) VALUES (?, 0, 0, 0, 0, 0)",
            (state_key,),
        )
    connection.execute("ALTER TABLE ticks DROP COLUMN event_ts")
    connection.execute("ALTER TABLE cost_counters DROP COLUMN priced_events")
    connection.execute("ALTER TABLE scopes DROP COLUMN durable")
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()

    store = SQLiteStateStore(database)
    healthy = store.read_scope("healthy")
    assert healthy.complete
    assert healthy.ticks[0].ts == 7.0
    assert healthy.ticks[0].reason == ""
    assert healthy.costs.priced_events == 1
    with pytest.raises(StateIncompleteError, match="non-durable"):
        store.load_or_create("corrupt", "fingerprint")
    with pytest.raises(StateCorruptionError, match="timestamp"):
        store.read_scope("corrupt")

    connection = sqlite3.connect(database)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    durability = dict(connection.execute("SELECT state_key, durable FROM scopes"))
    corrupt_ts = connection.execute(
        "SELECT event_ts FROM ticks WHERE state_key = 'corrupt'"
    ).fetchone()[0]
    connection.close()
    assert version == 3
    assert durability == {"healthy": 1, "corrupt": 0}
    assert corrupt_ts is None


def test_current_schema_open_does_not_require_a_migration_write_lock(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    SQLiteStateStore(database).close()
    holder = sqlite3.connect(database)
    holder.execute("BEGIN IMMEDIATE")
    try:
        opened = SQLiteStateStore(database, timeout=0.05)
        opened.close()
    finally:
        holder.rollback()
        holder.close()


def test_snapshot_timestamp_does_not_decode_sensitive_event_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "state.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    pipeline.process(ToolEvent("secret", result={"prompt": "sensitive"}, ts=4.5))

    def fail_decode(_payload: str) -> ToolEvent:
        raise AssertionError("event payload should stay encoded")

    monkeypatch.setattr(store_module, "decode_event", fail_decode)
    snapshot = SQLiteStateStore(database).read_scope("run")
    assert snapshot.ticks[0].ts == 4.5


@pytest.mark.parametrize("timestamp", [math.inf, -math.inf, math.nan])
def test_nonfinite_tick_timestamp_remains_lossless(tmp_path: Path, timestamp: float) -> None:
    database = tmp_path / "state.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    pipeline.process(ToolEvent("read", ts=timestamp))

    tick = SQLiteStateStore(database).read_scope("run").ticks[0]
    assert math.isnan(tick.ts) if math.isnan(timestamp) else tick.ts == timestamp
    restored = tick.event
    assert math.isnan(restored.ts) if math.isnan(timestamp) else restored.ts == timestamp


def test_event_access_rejects_timestamp_column_divergence(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    pipeline.process(ToolEvent("read", ts=1.0))
    connection = sqlite3.connect(database)
    connection.execute("UPDATE ticks SET event_ts = 2.0")
    connection.commit()
    connection.close()

    tick = SQLiteStateStore(database).read_scope("run").ticks[0]
    assert tick.ts == 2.0
    with pytest.raises(StateCorruptionError, match="timestamp disagrees"):
        _ = tick.event


def test_two_processes_serialize_first_schema_initialization(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    database = str(tmp_path / "state.db")
    processes = [
        context.Process(target=_initialize_state_process, args=(database, barrier, queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(15)

    assert [process.exitcode for process in processes] == [0, 0]
    assert [queue.get(timeout=2) for _ in processes] == [None, None]
    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    connection.close()


def test_two_processes_write_one_scope_without_overwriting_history(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    database = str(tmp_path / "state.db")
    processes = [
        context.Process(target=_write_state_process, args=(database, index, barrier, queue))
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(15)

    assert [process.exitcode for process in processes] == [0, 0]
    results = [queue.get(timeout=2) for _ in processes]
    assert all(result == (Action.NONE.name, True, None) for result in results)
    snapshot = SQLiteStateStore(database).read_scope("shared")
    assert snapshot.revision == 2
    assert {tick.event_id for tick in snapshot.ticks} == {"0", "1"}


def test_future_schema_is_refused(tmp_path: Path) -> None:
    database = tmp_path / "future.db"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 999")
    connection.close()

    with pytest.raises(StateCompatibilityError, match="newer"):
        SQLiteStateStore(database)


def test_corrupt_database_is_not_recreated(tmp_path: Path) -> None:
    database = tmp_path / "corrupt.db"
    database.write_bytes(b"not a sqlite database")

    with pytest.raises(StateCorruptionError):
        SQLiteStateStore(database)
    assert database.read_bytes() == b"not a sqlite database"


def test_locked_database_surfaces_a_write_error(tmp_path: Path) -> None:
    database = tmp_path / "locked.db"
    store = SQLiteStateStore(database, timeout=0.0)
    store.load_or_create("run", "fingerprint")
    blocker = sqlite3.connect(database)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(StateWriteError, match="locked"):
            store.close_scope("run", "fingerprint", 0, "completed")
    finally:
        blocker.rollback()
        blocker.close()


def test_locked_duplicate_lookup_fails_open(tmp_path: Path) -> None:
    database = tmp_path / "lookup-locked.db"
    store = SQLiteStateStore(database, timeout=0.01)
    pipeline = build_pipeline(_config(database), state_key="run", store=store)
    pipeline.process(_event(0), event_id="existing")
    store._connection.execute("PRAGMA journal_mode = DELETE")
    blocker = sqlite3.connect(database)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        assert pipeline.process(_event(1), event_id="new") is Action.NONE
    finally:
        blocker.rollback()
        blocker.close()

    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateWriteError)
    assert "locked" in str(pipeline.persistence_error)


def test_dead_connection_rollback_does_not_mask_original_error(tmp_path: Path) -> None:
    database = tmp_path / "closed.db"
    store = SQLiteStateStore(database)
    pipeline = build_pipeline(_config(database), state_key="run", store=store)
    store._connection.close()

    assert pipeline.process(_event(0)) is Action.NONE
    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateWriteError)
    assert "closed database" in str(pipeline.persistence_error)


def test_wal_mode_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = sqlite3.connect

    class ConnectionProxy:
        def __init__(self, connection: sqlite3.Connection) -> None:
            object.__setattr__(self, "connection", connection)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.connection, name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(self.connection, name, value)

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            if sql == "PRAGMA journal_mode = WAL":
                return self.connection.execute("SELECT 'delete'")
            return self.connection.execute(sql, *args, **kwargs)

    monkeypatch.setattr(
        store_module.sqlite3,
        "connect",
        lambda *args, **kwargs: ConnectionProxy(real_connect(*args, **kwargs)),
    )

    with pytest.raises(StateWriteError, match=r"requires WAL.*delete"):
        SQLiteStateStore(tmp_path / "no-wal.db")


def test_retry_and_budget_cross_restart_without_double_counting(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    config = PathlayerConfig(
        detectors=DetectorsConfig(
            budget=BudgetConfig(enabled=True, max_cost_usd=4.0, baseline_usd=1.0)
        ),
        policy=PolicySection(stop_threshold=0.99),
        store=StoreSection(path=str(database)),
    )
    failing = ToolEvent("fetch", {"url": "x"}, result={"error": "timeout"}, cost_usd=1.0)
    first = build_pipeline(config, state_key="run")
    assert first.process(failing, event_id="1") is Action.NONE
    assert first.process(failing, event_id="2") is Action.NONE

    resumed = build_pipeline(config, state_key="run")
    assert resumed.process(failing, event_id="3") is Action.STOP
    budget = next(signal for signal in resumed.last_decision.signals if signal.name == "budget")
    retry = next(signal for signal in resumed.last_decision.signals if signal.name == "retry")
    assert "$4.00 effective" in budget.reason
    assert retry.confidence == 1.0


def test_resource_counters_survive_cross_restart_without_double_counting(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    config = PathlayerConfig(
        detectors=DetectorsConfig(
            resource=ResourceConfig(enabled=True, max_spawn_count=4, max_delegation_depth=10)
        ),
        store=StoreSection(path=str(database)),
    )
    spawn = ToolEvent("spawn", kind=EventKind.SUBAGENT_SPAWN)
    depth = ToolEvent("delegate", metadata={"depth": 3})

    split = build_pipeline(config, state_key="run")
    split.process(spawn, event_id="1")
    split.process(depth, event_id="2")

    resumed = build_pipeline(config, state_key="run")
    resumed.process(spawn, event_id="3")

    reference = build_pipeline(config)
    reference.run([spawn, depth, spawn])

    resumed_signal = next(
        signal for signal in resumed.last_decision.signals if signal.name == "resource"
    )
    reference_signal = next(
        signal for signal in reference.last_decision.signals if signal.name == "resource"
    )
    assert resumed_signal.confidence == reference_signal.confidence == pytest.approx(0.5)


def test_hysteresis_counter_survives_restart(tmp_path: Path) -> None:
    config = PathlayerConfig(
        detectors=DetectorsConfig(
            budget=BudgetConfig(enabled=True, max_cost_usd=4.0),
        ),
        policy=PolicySection(tier_thresholds={"warn": 0.4}, escalate_ticks=2),
        store=StoreSection(path=str(tmp_path / "state.db")),
    )
    pipeline = build_pipeline(config, state_key="hysteresis")
    assert pipeline.process(ToolEvent("model", cost_usd=2.0), event_id="1") is Action.NONE

    resumed = build_pipeline(config, state_key="hysteresis")
    assert resumed.process(ToolEvent("model", cost_usd=0.1), event_id="2") is Action.WARN


class _FailOnceStore:
    def __init__(self, delegate: StateStore) -> None:
        self.delegate = delegate
        self.failed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def append_tick(self, *args: Any, **kwargs: Any) -> Any:
        if not self.failed:
            self.failed = True
            raise StateWriteError("disk full")
        return self.delegate.append_tick(*args, **kwargs)


class _LegacySignatureStore:
    """Third-party store compiled against the public pre-RES-2 signature."""

    def __init__(self, delegate: StateStore) -> None:
        self.delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def append_tick(
        self,
        state_key: str,
        fingerprint: str,
        expected_revision: int,
        event_id: str | None,
        event: ToolEvent,
        decision: Any,
        costs: BudgetState,
    ) -> Any:
        return self.delegate.append_tick(
            state_key,
            fingerprint,
            expected_revision,
            event_id,
            event,
            decision,
            costs,
        )


def test_persistent_pipeline_keeps_legacy_store_append_signature(tmp_path: Path) -> None:
    delegate = SQLiteStateStore(tmp_path / "state.db")
    pipeline = build_pipeline(
        PathlayerConfig(
            detectors=DetectorsConfig(resource=ResourceConfig(enabled=True)),
            store=StoreSection(path=str(tmp_path / "ignored.db")),
        ),
        state_key="run",
        store=_LegacySignatureStore(delegate),  # type: ignore[arg-type]
    )

    assert (
        pipeline.process(ToolEvent("pathlayer_run_start", kind=EventKind.RUN_START)) is Action.NONE
    )
    assert delegate.read_scope("run").revision == 1


def test_write_failure_keeps_action_and_permanently_degrades(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    durable = SQLiteStateStore(tmp_path / "state.db")
    failing = _FailOnceStore(durable)
    pipeline = build_pipeline(
        _config(tmp_path / "ignored.db"),
        state_key="run",
        store=failing,  # type: ignore[arg-type]
    )

    with caplog.at_level(logging.ERROR):
        assert pipeline.process(_event(0)) is Action.NONE
        assert pipeline.process(_event(1)) is Action.NONE

    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateWriteError)
    assert len(pipeline.history) == 2
    degraded = durable.read_scope("run")
    assert degraded.ticks == ()
    assert not degraded.durable
    assert sum("now memory-only" in record.message for record in caplog.records) == 1
    pipeline.close("completed")
    assert durable.read_scope("run").status == "active"


def test_sqlite_failure_rolls_back_the_whole_tick_before_fail_open(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    prepared = SQLiteStateStore(database)
    prepared.load_or_create("run", configuration_fingerprint(_config(database)))
    prepared.close()
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TRIGGER reject_tick BEFORE INSERT ON ticks
           BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END"""
    )
    connection.commit()
    connection.close()
    store = SQLiteStateStore(database)
    pipeline = build_pipeline(_config(database), state_key="run", store=store)

    assert pipeline.process(_event(0)) is Action.NONE
    assert not pipeline.durable
    connection = sqlite3.connect(database)
    counts = tuple(
        connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("events", "ticks", "signals")
    )
    revision = connection.execute("SELECT revision FROM scopes WHERE state_key='run'").fetchone()[0]
    durable = connection.execute("SELECT durable FROM scopes WHERE state_key='run'").fetchone()[0]
    connection.close()
    assert counts == (0, 0, 0)
    assert revision == 0
    assert durable == 0


def test_one_persistent_pipeline_serializes_threads(tmp_path: Path) -> None:
    config = _config(tmp_path / "state.db")
    pipeline = build_pipeline(config, state_key="run")
    events = [ToolEvent(f"tool-{index}", ts=float(index)) for index in range(20)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        actions = list(
            executor.map(
                lambda pair: pipeline.process(pair[1], event_id=str(pair[0])),
                enumerate(events),
            )
        )

    assert actions == [Action.NONE] * len(events)
    assert pipeline.durable
    assert len(pipeline.history) == len(events)
    assert SQLiteStateStore(tmp_path / "state.db").read_scope("run").revision == len(events)


class _FailLookupStore(_FailOnceStore):
    def lookup_event(self, *args: Any, **kwargs: Any) -> Any:
        raise StateWriteError("lookup failed")


class _FailAppendAndMarkerStore(_FailOnceStore):
    def mark_nondurable(self, *args: Any, **kwargs: Any) -> None:
        raise StateWriteError("cannot persist poison marker")


def test_failed_poison_marker_preserves_original_persistence_error(tmp_path: Path) -> None:
    durable = SQLiteStateStore(tmp_path / "state.db")
    pipeline = build_pipeline(
        _config(tmp_path / "ignored.db"),
        state_key="run",
        store=_FailAppendAndMarkerStore(durable),  # type: ignore[arg-type]
    )

    assert pipeline.process(_event(0)) is Action.NONE
    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateWriteError)
    assert str(pipeline.persistence_error) == "disk full"
    assert durable.read_scope("run").durable


def test_read_failure_before_evaluation_fails_open(tmp_path: Path) -> None:
    durable = SQLiteStateStore(tmp_path / "state.db")
    pipeline = build_pipeline(
        _config(tmp_path / "ignored.db"),
        state_key="run",
        store=_FailLookupStore(durable),  # type: ignore[arg-type]
    )

    assert pipeline.process(_event(0), event_id="one") is Action.NONE
    assert pipeline.history == (_event(0),)
    assert not pipeline.durable
    assert not durable.read_scope("run").durable


def test_unserializable_event_fails_open_and_stays_in_memory(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    pipeline = PersistentPipeline(lambda: Pipeline([]), store, "run", "custom-fingerprint")
    event = ToolEvent("custom", result=object())

    assert pipeline.process(event) is Action.NONE
    assert pipeline.history == (event,)
    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, EventSerializationError)
    assert not store.read_scope("run").durable
    assert pipeline.process(_event(0)) is Action.NONE
    assert len(pipeline.history) == 2


def test_cost_accounting_failure_keeps_the_evaluated_action(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")

    def fail_cost(_event: ToolEvent) -> BudgetState:
        raise ValueError("pricing unavailable")

    pipeline = PersistentPipeline(
        lambda: Pipeline([]),
        store,
        "run",
        "custom-fingerprint",
        cost_resolver=fail_cost,
    )
    event = ToolEvent("custom")

    assert pipeline.process(event) is Action.NONE
    assert pipeline.history == (event,)
    assert not pipeline.durable
    assert isinstance(pipeline.persistence_error, StateWriteError)
    snapshot = store.read_scope("run")
    assert snapshot.ticks == ()
    assert not snapshot.durable


def test_database_permissions_are_owner_only_when_supported(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permission bits are not available on Windows")
    database = tmp_path / "private" / "state.db"
    store = SQLiteStateStore(database)
    store.close()

    assert database.stat().st_mode & 0o777 == 0o600
    assert database.parent.stat().st_mode & 0o777 == 0o700


def test_wal_and_shm_permissions_are_owner_only(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permission bits are not available on Windows")
    database = tmp_path / "state.db"
    store = SQLiteStateStore(database)
    store.load_or_create("run", "fingerprint")

    for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
        assert path.exists()
        assert path.stat().st_mode & 0o777 == 0o600
    store.close()


def test_existing_parent_permissions_are_not_changed(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permission bits are not available on Windows")
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)

    store = SQLiteStateStore(parent / "state.db")
    store.close()

    assert parent.stat().st_mode & 0o777 == 0o755


class _NaNDetector(Detector):
    name = "custom_nan"

    def evaluate(self, history: Any) -> Signal:
        return Signal(self.name, math.nan)


def test_raw_nan_signal_snapshot_does_not_break_persistence(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    pipeline = PersistentPipeline(
        lambda: Pipeline([_NaNDetector()]), store, "run", "custom-fingerprint"
    )

    assert pipeline.process(ToolEvent("read")) is Action.NONE

    connection = sqlite3.connect(tmp_path / "state.db")
    confidence = connection.execute("SELECT confidence_json FROM signals").fetchone()[0]
    connection.close()
    assert json.loads(confidence) == {"$pathlayer": "float", "value": "nan"}


@pytest.mark.parametrize(
    ("statement", "match"),
    [
        ("UPDATE ticks SET fused=0.5", "fused score"),
        ("UPDATE ticks SET evidence_json='{\"progress\":0.5}'", "progress evidence"),
        ("UPDATE ticks SET action_reason='changed'", "action reason"),
        ("UPDATE ticks SET arbitrated_action='WARN'", "arbitrated action"),
        ("UPDATE ticks SET emitted_action='WARN'", "emitted action"),
        ("UPDATE ticks SET dry_run=1", "dry-run"),
        ("UPDATE signals SET confidence_json='0.5' WHERE signal_index=0", "confidence"),
        ("UPDATE signals SET reason='changed' WHERE signal_index=0", "signal 0 reason"),
        (
            "UPDATE signals SET diagnostics_json='[\"changed\"]' WHERE signal_index=0",
            "diagnostics",
        ),
    ],
)
def test_replay_refuses_any_stored_decision_divergence(
    tmp_path: Path, statement: str, match: str
) -> None:
    database = tmp_path / "state.db"
    config = _config(database)
    pipeline = build_pipeline(config, state_key="run")
    pipeline.process(_event(0))
    connection = sqlite3.connect(database)
    connection.execute(statement)
    connection.commit()
    connection.close()

    with pytest.raises(StateCompatibilityError, match=match):
        build_pipeline(config, state_key="run")


def test_missing_cost_counter_is_reported_as_corruption(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    store = SQLiteStateStore(database)
    store.load_or_create("run", "fingerprint")
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM cost_counters WHERE state_key='run'")
    connection.commit()
    connection.close()

    with pytest.raises(StateCorruptionError, match="cost counter mismatch"):
        store.read_scope("run")


def test_malformed_tick_value_is_reported_as_corruption(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    pipeline = build_pipeline(_config(database), state_key="run")
    pipeline.process(_event(0))
    connection = sqlite3.connect(database)
    connection.execute("UPDATE ticks SET fused='not-a-score'")
    connection.commit()
    connection.close()

    with pytest.raises(StateCorruptionError, match="invalid stored tick"):
        SQLiteStateStore(database).read_scope("run")


def test_non_tool_events_survive_replay(tmp_path: Path) -> None:
    config = _config(tmp_path / "state.db")
    pipeline = build_pipeline(config, state_key="run")
    event = ToolEvent("model", kind=EventKind.LLM_CALL, cost_usd=0.1)
    pipeline.process(event)

    resumed = build_pipeline(config, state_key="run")
    assert resumed.history[0].kind is EventKind.LLM_CALL


@pytest.mark.parametrize("enabled", [True, False])
def test_adapter_cost_estimates_survive_restart(tmp_path: Path, enabled: bool) -> None:
    database = tmp_path / "estimate.db"
    config = _config(
        database, budget=BudgetConfig(enabled=enabled, max_cost_usd=100.0, default_cost_usd=10.0)
    )
    first = build_pipeline(config, state_key="estimate")
    event = ToolEvent(
        "model",
        kind=EventKind.LLM_CALL,
        cost_usd=0.5,
        metadata={"cost_measurement": "estimated_total_tokens"},
    )
    first.process(event, event_id="one")
    resumed = build_pipeline(config, state_key="estimate")
    resumed.process(replace(event, cost_usd=0.0), event_id="two")
    snapshot = SQLiteStateStore(database).read_scope("estimate")
    assert snapshot.costs == BudgetState(estimated_usd=0.5)
    if enabled:
        signal = next(s for s in resumed.last_decision.signals if s.name == "budget")
        assert "$0.00 captured + $0.50 estimated" in signal.reason
