"""RES-1: tests for the ResourceDetector."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathlayer.detectors.resource import (
    DURATION_REASON_PREFIX,
    ResourceDetector,
    ResourceState,
    advance_resource_state,
    resource_delta,
)
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Action, Policy
from pathlayer.policies.base import PolicyConfig
from pathlayer.validation import Trace, inject_resource_spike, load_trace

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def _metadata(depth: int | None, rss: int | None) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if depth is not None:
        metadata["depth"] = depth
    if rss is not None:
        metadata["rss_bytes"] = rss
    return metadata


def _spawn(depth: int | None = None, rss: int | None = None) -> ToolEvent:
    return ToolEvent(tool="spawn", kind=EventKind.SUBAGENT_SPAWN, metadata=_metadata(depth, rss))


def _restart(depth: int | None = None, rss: int | None = None) -> ToolEvent:
    return ToolEvent(tool="restart", kind=EventKind.RUN_RESTART, metadata=_metadata(depth, rss))


def _tool(depth: int | None = None, rss: int | None = None) -> ToolEvent:
    return ToolEvent(tool="call", metadata=_metadata(depth, rss))


def _start() -> ToolEvent:
    return ToolEvent(
        tool="pathlayer_run_start",
        kind=EventKind.RUN_START,
        metadata={"started_at_unix": 100.0, "max_duration_seconds": 100},
    )


def _timeout() -> ToolEvent:
    return ToolEvent(
        tool="pathlayer_run_timeout",
        kind=EventKind.RUN_TIMEOUT,
        metadata={"started_at_unix": 100.0, "max_duration_seconds": 100},
    )


def test_empty_history_is_silent() -> None:
    assert ResourceDetector().evaluate([]).confidence == 0.0


def test_resource_state_field_order_is_positionally_stable() -> None:
    # RES-3 fields must be appended after every pre-existing field, never
    # interleaved: a historical positional ResourceState(spawn, restart,
    # depth, invalid_depths, timed_out) call must still mean exactly that,
    # with the two RES-3 fields defaulting to 0 rather than silently
    # consuming timed_out's slot.
    state = ResourceState(1, 2, 3, 4, True)
    assert state == ResourceState(
        spawn_count=1,
        restart_count=2,
        max_depth=3,
        invalid_depths=4,
        timed_out=True,
        peak_rss_bytes=0,
        invalid_rss=0,
    )


# --------------------------------------------------------------------------- #
# pure-fold unit tests
# --------------------------------------------------------------------------- #


def test_resource_delta_counts_spawn_and_restart_by_kind() -> None:
    assert resource_delta(_spawn()) == ResourceState(spawn_count=1)
    assert resource_delta(_restart()) == ResourceState(restart_count=1)
    assert resource_delta(_tool()) == ResourceState()


def test_resource_delta_reads_depth_from_any_event_kind() -> None:
    assert resource_delta(_tool(depth=2)) == ResourceState(max_depth=2)
    assert resource_delta(_spawn(depth=2)) == ResourceState(spawn_count=1, max_depth=2)


def test_resource_delta_missing_depth_is_zero_without_diagnostic() -> None:
    assert resource_delta(_tool()) == ResourceState()


@pytest.mark.parametrize("bad_depth", [-1, 1.5, True, "3", None])
def test_resource_delta_invalid_depth_is_ignored_and_flagged(bad_depth: object) -> None:
    event = ToolEvent(tool="call", metadata={"depth": bad_depth})
    assert resource_delta(event) == ResourceState(max_depth=0, invalid_depths=1)


def test_advance_resource_state_sums_counts_and_maxes_depth() -> None:
    state = ResourceState()
    state = advance_resource_state(state, ResourceState(spawn_count=1, max_depth=1))
    state = advance_resource_state(state, ResourceState(spawn_count=1, max_depth=3))
    state = advance_resource_state(state, ResourceState(spawn_count=1, max_depth=1))

    assert state == ResourceState(spawn_count=3, max_depth=3)


def test_resource_delta_reads_rss_from_any_event_kind() -> None:
    assert resource_delta(_tool(rss=2048)) == ResourceState(peak_rss_bytes=2048)
    assert resource_delta(_spawn(rss=2048)) == ResourceState(spawn_count=1, peak_rss_bytes=2048)


def test_resource_delta_missing_rss_is_zero_without_diagnostic() -> None:
    assert resource_delta(_tool()) == ResourceState()


@pytest.mark.parametrize("bad_rss", [-1, 1.5, True, "3", None])
def test_resource_delta_invalid_rss_is_ignored_and_flagged(bad_rss: object) -> None:
    event = ToolEvent(tool="call", metadata={"rss_bytes": bad_rss})
    assert resource_delta(event) == ResourceState(peak_rss_bytes=0, invalid_rss=1)


def test_advance_resource_state_maxes_rss() -> None:
    state = ResourceState()
    state = advance_resource_state(state, ResourceState(peak_rss_bytes=1024))
    state = advance_resource_state(state, ResourceState(peak_rss_bytes=4096))
    state = advance_resource_state(state, ResourceState(peak_rss_bytes=1024))

    assert state == ResourceState(peak_rss_bytes=4096)


# --------------------------------------------------------------------------- #
# per-counter proportional / saturating behaviour
# --------------------------------------------------------------------------- #


def test_spawn_heavy_is_proportional_then_saturates() -> None:
    detector = ResourceDetector(max_spawn_count=4)
    below = detector.evaluate([_spawn(), _spawn()])
    assert below.confidence == pytest.approx(0.5)

    at = detector.evaluate([_spawn()] * 4)
    assert at.confidence == 1.0
    assert "spawn" in at.reason

    over = detector.evaluate([_spawn()] * 6)
    assert over.confidence == 1.0


def test_deep_delegation_is_proportional_then_saturates_and_tracks_high_water_mark() -> None:
    detector = ResourceDetector(max_delegation_depth=4)
    below = detector.evaluate([_tool(depth=2)])
    assert below.confidence == pytest.approx(0.5)

    at = detector.evaluate([_tool(depth=4)])
    assert at.confidence == 1.0
    assert "depth" in at.reason

    # 1 -> 3 -> 1: the high-water mark of 3 must still be reported, not the
    # most recent reading of 1.
    sequence = detector.evaluate([_tool(depth=1), _tool(depth=3), _tool(depth=1)])
    assert sequence.confidence == pytest.approx(0.75)


def test_restart_heavy_is_proportional_then_saturates() -> None:
    detector = ResourceDetector(max_restart_count=4)
    below = detector.evaluate([_restart(), _restart()])
    assert below.confidence == pytest.approx(0.5)

    at = detector.evaluate([_restart()] * 4)
    assert at.confidence == 1.0
    assert "restart" in at.reason


def test_rss_heavy_is_proportional_then_saturates_and_tracks_high_water_mark() -> None:
    detector = ResourceDetector(max_rss_bytes=4000)
    below = detector.evaluate([_tool(rss=2000)])
    assert below.confidence == pytest.approx(0.5)

    at = detector.evaluate([_tool(rss=4000)])
    assert at.confidence == 1.0
    assert "rss" in at.reason

    # 1000 -> 3000 -> 1000: the high-water mark of 3000 must still be
    # reported, not the most recent reading of 1000.
    sequence = detector.evaluate([_tool(rss=1000), _tool(rss=3000), _tool(rss=1000)])
    assert sequence.confidence == pytest.approx(0.75)


def test_duration_is_silent_until_the_discrete_timeout_event() -> None:
    detector = ResourceDetector(max_duration_seconds=100)
    assert detector.evaluate([_start(), _tool()]).confidence == 0.0

    at = detector.evaluate([_start(), _tool(), _timeout()])
    assert at.confidence == 1.0
    assert at.reason == "run duration reached 100s hard limit"


# --------------------------------------------------------------------------- #
# independence + reason naming
# --------------------------------------------------------------------------- #


def test_each_counter_fires_independently_of_the_others() -> None:
    detector = ResourceDetector(max_spawn_count=2, max_delegation_depth=2, max_restart_count=2)

    spawn_only = detector.evaluate([_spawn(), _spawn()])
    assert spawn_only.confidence == 1.0
    assert "spawn" in spawn_only.reason

    depth_only = detector.evaluate([_tool(depth=2)])
    assert depth_only.confidence == 1.0
    assert "depth" in depth_only.reason

    restart_only = detector.evaluate([_restart(), _restart()])
    assert restart_only.confidence == 1.0
    assert "restart" in restart_only.reason


def test_each_counter_fires_independently_including_rss() -> None:
    detector = ResourceDetector(max_rss_bytes=2000)
    rss_only = detector.evaluate([_tool(rss=2000)])
    assert rss_only.confidence == 1.0
    assert "rss" in rss_only.reason


def test_mixed_case_names_the_dominant_counter_in_the_reason() -> None:
    detector = ResourceDetector(max_spawn_count=10, max_delegation_depth=10, max_restart_count=2)
    signal = detector.evaluate([_spawn(), _tool(depth=1), _restart(), _restart()])

    assert signal.confidence == 1.0
    assert "restart" in signal.reason


def test_co_occurring_breach_is_visible_in_diagnostics() -> None:
    detector = ResourceDetector(max_spawn_count=2, max_delegation_depth=10, max_restart_count=2)
    events = [_spawn(), _spawn(), _restart(), _restart()]
    signal = detector.evaluate(events)

    assert signal.confidence == 1.0
    # spawn wins the tie-break (fixed order spawn, depth, restart).
    assert "spawn" in signal.reason
    assert signal.diagnostics == ("restart also at threshold (2/2)",)


def test_duration_wins_tie_break_after_timeout() -> None:
    # RUN_TIMEOUT is the one-shot hard-ceiling fact. It must remain the
    # actionable reason even when an older RES-1/RES-3 counter is also
    # saturated; otherwise the duration-only policy override cannot
    # recognize it.
    detector = ResourceDetector(max_spawn_count=2, max_rss_bytes=100)
    events = [_spawn(), _spawn(), _tool(rss=100), _start(), _timeout()]
    signal = detector.evaluate(events)

    assert signal.confidence == 1.0
    assert signal.reason.startswith(DURATION_REASON_PREFIX)
    assert signal.diagnostics == (
        "spawn also at threshold (2/2)",
        "rss also at threshold (100/100)",
    )


def test_invalid_depths_diagnostic() -> None:
    detector = ResourceDetector()
    signal = detector.evaluate([ToolEvent(tool="call", metadata={"depth": -1})])
    assert signal.diagnostics == ("1 invalid depth value(s) ignored",)


def test_invalid_rss_diagnostic() -> None:
    detector = ResourceDetector()
    signal = detector.evaluate([ToolEvent(tool="call", metadata={"rss_bytes": -1})])
    assert signal.diagnostics == ("1 invalid rss_bytes value(s) ignored",)


# --------------------------------------------------------------------------- #
# PERF-2 contract
# --------------------------------------------------------------------------- #


def test_incremental_step_matches_batch_for_every_prefix() -> None:
    detector = ResourceDetector(
        max_spawn_count=3, max_delegation_depth=5, max_restart_count=3, max_rss_bytes=5000
    )
    events = [
        _spawn(),
        _tool(depth=1, rss=1000),
        _restart(),
        _start(),
        _timeout(),
        _tool(depth=3, rss=3000),
        _spawn(),
        _tool(depth=1, rss=1000),
        ToolEvent(tool="call", metadata={"depth": -1, "rss_bytes": -1}),
        _restart(),
    ]
    state = detector.create_state()
    for index, event in enumerate(events):
        state, incremental = detector.step(state, event)
        assert incremental == detector.evaluate(events[: index + 1])


def test_shared_detector_keeps_pipeline_states_isolated() -> None:
    detector = ResourceDetector(max_spawn_count=2)
    spawny_pipeline = Pipeline([detector], Policy())
    quiet_pipeline = Pipeline([detector], Policy())

    spawny_pipeline.run([_spawn(), _spawn()])
    quiet_pipeline.run([_tool()])

    assert spawny_pipeline.stopped
    assert not quiet_pipeline.stopped


# --------------------------------------------------------------------------- #
# cross-run continuity (in-process, no store)
# --------------------------------------------------------------------------- #


def test_split_stream_fold_equals_full_stream_evaluate() -> None:
    detector = ResourceDetector(max_spawn_count=5, max_delegation_depth=5, max_restart_count=5)
    events = [_spawn(), _tool(depth=3), _restart(), _spawn(), _tool(depth=1)]

    full = detector.evaluate(events)

    state = detector.create_state()
    for event in events[:2]:
        state, _ = detector.step(state, event)
    for event in events[2:]:
        state, split_signal = detector.step(state, event)

    assert split_signal == full


# --------------------------------------------------------------------------- #
# reason-string content
# --------------------------------------------------------------------------- #


def test_reason_names_counter_value_and_threshold() -> None:
    detector = ResourceDetector(max_spawn_count=5)
    signal = detector.evaluate([_spawn(), _spawn(), _spawn()])
    assert signal.reason == "3 sub-agent spawn(s) / 5 threshold"


def test_timeout_reason_starts_with_the_exported_prefix() -> None:
    # config.py wires pathlayer.policies.base's hard-STOP override by matching
    # signal.reason.startswith(DURATION_REASON_PREFIX) (conditional_overrides).
    # If this ever drifts, that override silently stops firing -- this test
    # exists so a reformulation of the reason string is a deliberate decision,
    # not an accident.
    detector = ResourceDetector()
    signal = detector.evaluate([_timeout()])
    assert signal.reason.startswith(DURATION_REASON_PREFIX)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
@pytest.mark.parametrize(
    "field",
    [
        "max_spawn_count",
        "max_delegation_depth",
        "max_restart_count",
        "max_duration_seconds",
        "max_rss_bytes",
    ],
)
def test_invalid_config_raises(field: str, bad: object) -> None:
    with pytest.raises(ValueError):
        ResourceDetector(**{field: bad})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pipeline/Policy integration
# --------------------------------------------------------------------------- #


def test_pipeline_stops_when_a_counter_crosses_stop_threshold_independently() -> None:
    policy = Policy()
    assert Pipeline([ResourceDetector(max_spawn_count=2)], policy).run([_spawn(), _spawn()]) == [
        Action.NONE,
        Action.STOP,
    ]
    assert Pipeline([ResourceDetector(max_delegation_depth=2)], policy).run([_tool(depth=2)]) == [
        Action.STOP
    ]
    assert Pipeline([ResourceDetector(max_restart_count=2)], policy).run(
        [_restart(), _restart()]
    ) == [Action.NONE, Action.STOP]


def test_pipeline_stops_exactly_on_timeout_event() -> None:
    pipeline = Pipeline([ResourceDetector(max_duration_seconds=100)], Policy())
    assert pipeline.process(_start()) is Action.NONE
    assert pipeline.process(_tool()) is Action.NONE
    assert pipeline.process(_timeout()) is Action.STOP


def test_direct_pipeline_resource_advisory_also_covers_timeout() -> None:
    policy = Policy(PolicyConfig(advisory_signals=frozenset({"resource"})))
    pipeline = Pipeline([ResourceDetector(max_duration_seconds=100)], policy)

    assert pipeline.process(_start()) is Action.NONE
    assert pipeline.process(_timeout()) is Action.NONE
    assert pipeline.last_decision is not None
    assert pipeline.last_decision.signals[-1].name == "resource"


def test_healthy_trace_stays_silent() -> None:
    trace = load_trace(_TRACES_DIR / "healthy" / "varied_long_run.json")
    assert ResourceDetector().evaluate(trace.events).confidence == 0.0


def test_labeled_timeout_trace_fires_and_healthy_duration_counterexample_stays_silent() -> None:
    failing = load_trace(Path(__file__).parent / "fixtures" / "runtime_timeout.json")
    healthy = load_trace(_TRACES_DIR / "healthy" / "finishes_before_runtime_limit.json")
    detector = ResourceDetector(max_duration_seconds=100)

    assert detector.evaluate(failing.events).confidence == 1.0
    assert Policy().decide([detector.evaluate(failing.events)]) is Action.STOP
    assert detector.evaluate(healthy.events).confidence == 0.0


# --------------------------------------------------------------------------- #
# VAL-4 mutation fixtures
# --------------------------------------------------------------------------- #


def test_fires_on_spawn_spike_mutation() -> None:
    base = Trace(label="healthy", events=[ToolEvent(tool="call")])
    mutated = inject_resource_spike(base, kind="spawn", start_idx=1, count=5)
    signal = ResourceDetector(max_spawn_count=5).evaluate(mutated.events)
    assert signal.confidence == 1.0
    assert Policy().decide([signal]) is Action.STOP


def test_fires_on_depth_spike_mutation() -> None:
    base = Trace(label="healthy", events=[ToolEvent(tool="call")])
    mutated = inject_resource_spike(base, kind="depth", start_idx=1, count=3)
    signal = ResourceDetector(max_delegation_depth=3).evaluate(mutated.events)
    assert signal.confidence == 1.0
    assert Policy().decide([signal]) is Action.STOP


def test_fires_on_restart_spike_mutation() -> None:
    base = Trace(label="healthy", events=[ToolEvent(tool="call")])
    mutated = inject_resource_spike(base, kind="restart", start_idx=1, count=3)
    signal = ResourceDetector(max_restart_count=3).evaluate(mutated.events)
    assert signal.confidence == 1.0
    assert Policy().decide([signal]) is Action.STOP


def test_fires_on_rss_spike_mutation() -> None:
    base = Trace(label="healthy", events=[ToolEvent(tool="call")])
    mutated = inject_resource_spike(base, kind="rss", start_idx=1, count=2_000_000_000)
    signal = ResourceDetector(max_rss_bytes=2_000_000_000).evaluate(mutated.events)
    assert signal.confidence == 1.0
    assert Policy().decide([signal]) is Action.STOP


def test_healthy_boundary_stays_silent_well_below_rss_threshold() -> None:
    # Half the threshold: comfortably below both 1.0 and the default 0.9
    # stop_threshold, so this is a genuine "not a problem" counterexample,
    # not just a hair under the saturation point.
    base = Trace(label="healthy", events=[ToolEvent(tool="call")])
    mutated = inject_resource_spike(base, kind="rss", start_idx=1, count=1_000_000_000)
    signal = ResourceDetector(max_rss_bytes=2_000_000_000).evaluate(mutated.events)
    assert signal.confidence == pytest.approx(0.5)
    assert Policy().decide([signal]) is Action.NONE
