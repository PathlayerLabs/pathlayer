"""NOP-5: comparable per-test outcomes and adversarial replay matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

import pathlayer.detectors.progress as progress_module
from pathlayer.detectors.loop import LoopDetector
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    GitProgressAnchor,
    TestProgressAnchor,
    coding_anchors,
)
from pathlayer.detectors.retry import RetryDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pipeline import Pipeline
from pathlayer.policies import Policy
from pathlayer.validation import load_trace, replay_trace

_TRACES = Path(__file__).resolve().parents[1] / "traces"


def _rich(
    outcomes: dict[str, str],
    *,
    target: str = "suite",
    scope: str = "all",
    complete: bool = True,
    exhaustive: bool = True,
    run: int = 0,
    aggregate: tuple[int, int] | None = None,
) -> ToolEvent:
    result: dict[str, object] = {
        "test_outcomes": outcomes,
        "scope_id": scope,
        "collection_complete": complete,
        "scope_exhaustive": exhaustive,
    }
    if aggregate is not None:
        result["passed"], result["failed"] = aggregate
    return ToolEvent(
        tool="run_tests",
        args={"target": target, "run": run},
        result=result,
    )


def _legacy(passed: int, failed: int, *, target: str = "suite", run: int = 0) -> ToolEvent:
    return ToolEvent(
        tool="run_tests",
        args={"target": target, "run": run},
        result={"passed": passed, "failed": failed},
    )


def _edit(index: int) -> ToolEvent:
    return ToolEvent(tool="edit_file", args={"path": "src/x.py", "patch": f"change {index}"})


def test_first_comparable_resolution_is_credited_once() -> None:
    anchor = TestProgressAnchor()
    baseline = [_rich({"a": "failed", "b": "failed"})]
    repaired = [*baseline, _rich({"a": "passed", "b": "failed"}, run=1)]
    regressed = [*repaired, _rich({"a": "failed", "b": "failed"}, run=2)]
    repaired_again = [*regressed, _rich({"a": "passed", "b": "failed"}, run=3)]

    assert anchor.measure(baseline) == 0.0
    assert anchor.measure(repaired) == 1.0
    assert anchor.measure(regressed) == 1.0
    assert anchor.measure(repaired_again) == 1.0


def test_new_target_starts_neutral_then_progresses_locally() -> None:
    anchor = TestProgressAnchor()
    history = [
        _legacy(20, 0, target="old"),
        _rich({"a": "failed", "b": "failed"}, target="new"),
    ]
    assert anchor.measure(history) == 20.0
    history.append(_rich({"a": "passed", "b": "failed"}, target="new", run=1))
    assert anchor.measure(history) == 21.0


def test_legacy_to_rich_preserves_offset_and_is_sticky_per_target() -> None:
    anchor = TestProgressAnchor()
    history = [_legacy(8, 2, target="a"), _legacy(3, 1, target="b")]
    assert anchor.measure(history) == 8.0
    history.append(_rich({"x": "failed"}, target="a"))
    assert anchor.measure(history) == 8.0
    history.append(_legacy(100, 0, target="a", run=2))
    assert anchor.measure(history) == 8.0
    history.append(_legacy(4, 0, target="b", run=2))
    assert anchor.measure(history) == 10.0
    history.append(_rich({"x": "passed"}, target="a", run=3))
    assert anchor.measure(history) == 11.0


def test_resolution_cross_scope_is_credited_once() -> None:
    anchor = TestProgressAnchor()
    history = [
        _rich({"a": "failed"}, scope="full"),
        _rich({"a": "passed"}, scope="focused", exhaustive=False, run=1),
        _rich({"a": "failed"}, scope="full", run=2),
        _rich({"a": "passed"}, scope="full", run=3),
    ]
    assert [anchor.measure(history[:i]) for i in range(1, 5)] == [0.0, 1.0, 1.0, 1.0]


def test_incomplete_collection_credits_explicit_pass_but_never_satisfies() -> None:
    anchor = TestProgressAnchor()
    history = [
        _rich({"a": "failed", "b": "failed"}),
        _rich({"a": "passed"}, complete=False, exhaustive=False, run=1),
    ]
    assert anchor.measure(history) == 1.0
    assert anchor.is_satisfied(history) is False


def test_only_complete_exhaustive_all_green_result_satisfies() -> None:
    anchor = TestProgressAnchor()
    baseline = [_rich({"a": "failed"})]
    partial_green = [*baseline, _rich({"a": "passed"}, exhaustive=False, run=1)]
    exhaustive_green = [*partial_green, _rich({"a": "passed"}, run=2)]
    assert anchor.is_satisfied(partial_green) is False
    assert anchor.is_satisfied(exhaustive_green) is True


def test_all_green_baseline_and_new_debt_lifecycle() -> None:
    anchor = TestProgressAnchor()
    green = [_rich({"a": "passed"})]
    new_debt = [*green, _rich({"a": "passed", "b": "failed"}, run=1)]
    repaired = [*new_debt, _rich({"a": "passed", "b": "passed"}, run=2)]
    assert anchor.measure(green) == 0.0
    assert anchor.is_satisfied(green) is True
    assert anchor.measure(new_debt) == 0.0
    assert anchor.is_satisfied(new_debt) is False
    assert anchor.measure(repaired) == 1.0
    assert anchor.is_satisfied(repaired) is True


@pytest.mark.parametrize(
    "second",
    [
        _rich({"renamed-a": "passed", "b": "failed"}, run=1),
        _rich({"b": "failed"}, run=1),
        _rich({"b": "failed"}, complete=False, exhaustive=False, run=1),
    ],
    ids=["renamed", "absent-complete", "absent-incomplete"],
)
def test_renamed_or_absent_failure_is_not_progress(second: ToolEvent) -> None:
    anchor = TestProgressAnchor()
    history = [_rich({"a": "failed", "b": "failed"}), second]
    assert anchor.measure(history) == 0.0
    assert anchor.is_satisfied(history) is False


def test_deleted_failure_reintroduced_as_passed_is_not_comparable_progress() -> None:
    anchor = TestProgressAnchor()
    history = [
        _rich({"a": "failed", "b": "failed"}),
        _rich({"b": "failed"}, run=1),
        _rich({"a": "passed", "b": "failed"}, run=2),
    ]
    assert anchor.measure(history) == 0.0
    assert anchor.is_satisfied(history) is False


def test_partial_scope_omission_invalidates_comparability_in_that_scope() -> None:
    anchor = TestProgressAnchor()
    history = [
        _rich({"a": "failed"}, scope="focused", exhaustive=False),
        _rich({}, scope="focused", exhaustive=False, run=1),
        _rich({"a": "passed"}, scope="focused", exhaustive=False, run=2),
    ]
    assert [anchor.measure(history[:index]) for index in range(1, 4)] == [0.0, 0.0, 0.0]


def test_rich_test_evidence_makes_weaker_anchors_defer() -> None:
    rich = _rich({"a": "failed"})
    history = [
        ToolEvent(tool="shell", result={"exit_code": 0}),
        ToolEvent(tool="edit", result={"insertions": 5, "deletions": 0}),
        rich,
    ]
    assert ExitCodeProgressAnchor().measure(history) is None
    assert GitProgressAnchor().measure(history) is None


def test_malformed_rich_claim_is_inert_without_disabling_weaker_anchors() -> None:
    malformed = ToolEvent(
        tool="run_tests",
        result={
            "test_outcomes": {"a": "unknown"},
            "scope_id": "all",
            "collection_complete": True,
            "scope_exhaustive": True,
        },
    )
    assert TestProgressAnchor().measure([malformed]) is None
    exit_history = [ToolEvent(tool="shell", result={"exit_code": 0}), malformed]
    assert ExitCodeProgressAnchor().measure(exit_history) == 1.0
    git_history = [ToolEvent(tool="git", result={"net_lines": 2}), malformed]
    assert GitProgressAnchor().measure(git_history) == 2.0
    missing_scope = ToolEvent(
        tool="run_tests",
        result={
            "test_outcomes": {"a": "failed"},
            "collection_complete": True,
            "scope_exhaustive": True,
        },
    )
    assert TestProgressAnchor().measure([missing_scope]) is None
    assert TestProgressAnchor().diagnose([malformed]) == (
        "target 'run_tests': malformed test_outcomes payload ignored",
    )

    malformed_with_aggregate = ToolEvent(
        tool="run_tests",
        result={
            "passed": 3,
            "failed": 1,
            "test_outcomes": {"a": "unknown"},
            "scope_id": "all",
            "collection_complete": True,
            "scope_exhaustive": True,
        },
    )
    assert TestProgressAnchor().measure([malformed_with_aggregate]) == 2.0
    assert ExitCodeProgressAnchor().measure([malformed_with_aggregate]) is None
    assert TestProgressAnchor().diagnose([malformed_with_aggregate]) == (
        "target 'run_tests': malformed test_outcomes payload ignored "
        "(aggregate counts still applied)",
    )


def test_non_tool_events_do_not_advance_or_consume_patience() -> None:
    detector = NoProgressDetector([TestProgressAnchor()])
    baseline = _rich({"a": "failed"})
    llm_events = [ToolEvent(tool="llm", kind=EventKind.LLM_CALL, cost_usd=1.0) for _ in range(20)]
    assert detector.evaluate([baseline, *llm_events]).confidence == 0.0


def test_state_is_hashable_and_high_scope_cardinality_does_not_duplicate_debt() -> None:
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    for index in range(70):
        state, snapshot = anchor.step(
            state,
            _rich(
                {"a": "failed"},
                scope=f"scope-{index}",
                complete=False,
                exhaustive=False,
                run=index,
            ),
        )
    assert hash(state)
    assert snapshot.level == 0.0
    rich_target = dict(state.rich_targets)["suite"]
    assert rich_target.debts == frozenset({"a"})


def test_rich_ledger_cap_is_sticky_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_module, "_MAX_TRACKED_TEST_IDS_PER_TARGET", 2)
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(state, _rich({"a": "failed"}))
    state, snapshot = anchor.step(
        state,
        _rich({"b": "failed", "c": "failed"}, run=1),
    )
    rich_target = dict(state.rich_targets)["suite"]
    assert rich_target.overflowed is True
    tracked_ids = (
        set(rich_target.debts)
        | set(rich_target.credited)
        | {test_id for _, statuses in rich_target.scope_statuses for test_id, _ in statuses}
    )
    assert tracked_ids == {"a", "b"}
    assert snapshot.satisfied is False
    assert snapshot.diagnostics == (
        "target 'suite': rich outcome capacity reached; 1 outcome dropped this event",
    )

    oversized = _rich({"a": "failed", "b": "failed", "c": "failed"})
    assert anchor.measure([oversized]) is None


def test_rich_global_caps_block_target_and_cross_target_storms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_module, "_MAX_RICH_TEST_TARGETS", 2)
    monkeypatch.setattr(progress_module, "_MAX_TRACKED_TEST_IDS_TOTAL", 2)
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(state, _rich({"a": "passed"}, target="a"))
    state, snapshot_b = anchor.step(
        state,
        _rich({"b": "failed", "c": "failed"}, target="b"),
    )
    assert sum(len(target.debts) for _, target in state.rich_targets) == 1
    assert dict(state.rich_targets)["b"].overflowed is True
    assert snapshot_b.diagnostics == (
        "target 'b': rich outcome capacity reached; 1 outcome dropped this event",
    )

    state, snapshot = anchor.step(
        state,
        _rich({"d": "passed"}, target="c"),
    )
    assert len(state.rich_targets) == 2
    assert state.rich_target_overflowed is True
    assert snapshot.satisfied is False
    assert snapshot.diagnostics == (
        "target 'c': rich test-target capacity (2) reached; new target ignored",
    )


def test_rich_scope_cap_is_sticky_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_module, "_MAX_RICH_SCOPES_PER_TARGET", 2)
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(
        state,
        _rich({"a": "failed"}, scope="one", exhaustive=False),
    )
    state, _ = anchor.step(
        state,
        _rich({}, scope="two", exhaustive=False, run=1),
    )
    state, snapshot = anchor.step(
        state,
        _rich({"a": "passed"}, scope="three", exhaustive=False, run=2),
    )
    target = dict(state.rich_targets)["suite"]
    assert len(target.scope_statuses) == 2
    assert target.overflowed is True
    assert target.credited == frozenset()
    assert snapshot.level == 0.0
    assert snapshot.satisfied is False
    assert snapshot.diagnostics == (
        "target 'suite': rich scope capacity (2) reached; scope 'three' dropped",
    )


def test_scope_status_entry_cap_counts_duplicate_ids_across_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_module, "_MAX_SCOPE_STATUS_ENTRIES_TOTAL", 2)
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(
        state,
        _rich({"a": "failed", "b": "failed"}, scope="one", exhaustive=False),
    )
    state, snapshot = anchor.step(
        state,
        _rich({"a": "passed", "b": "passed"}, scope="two", exhaustive=False, run=1),
    )
    target = dict(state.rich_targets)["suite"]
    assert sum(len(statuses) for _, statuses in target.scope_statuses) == 2
    assert target.credited == frozenset()
    assert target.overflowed is True
    assert snapshot.level == 0.0
    assert snapshot.diagnostics == (
        "target 'suite': rich outcome capacity reached; 2 outcomes dropped this event",
    )


def test_scope_status_cap_counts_other_targets_during_exhaustive_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_module, "_MAX_SCOPE_STATUS_ENTRIES_TOTAL", 3)
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(state, _rich({"x": "failed"}, target="a"))
    state, _ = anchor.step(state, _rich({"z": "failed"}, target="b"))
    state, snapshot = anchor.step(
        state,
        _rich({"x": "failed", "y": "failed", "w": "failed"}, target="a", run=1),
    )
    targets = dict(state.rich_targets)
    total_entries = sum(
        len(statuses) for target in targets.values() for _, statuses in target.scope_statuses
    )
    assert total_entries == 3
    assert targets["a"].overflowed is True
    assert snapshot.satisfied is False
    assert snapshot.diagnostics == (
        "target 'a': rich outcome capacity reached; 1 outcome dropped this event",
    )


def test_partial_scope_entry_overflow_cannot_credit_accepted_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(
        state,
        _rich({"a": "failed"}, scope="one", exhaustive=False),
    )
    monkeypatch.setattr(progress_module, "_MAX_SCOPE_STATUS_ENTRIES_TOTAL", 2)
    state, snapshot = anchor.step(
        state,
        _rich(
            {"a": "passed", "b": "passed"},
            scope="two",
            exhaustive=False,
            run=1,
        ),
    )
    target = dict(state.rich_targets)["suite"]
    assert target.overflowed is True
    assert target.credited == frozenset()
    assert snapshot.level == 0.0
    assert snapshot.diagnostics == (
        "target 'suite': rich outcome capacity reached; 1 outcome dropped this event",
    )


def test_malformed_aggregate_event_while_reverting_reports_both_diagnostics() -> None:
    # A malformed test_outcomes payload can coexist with valid aggregate counts
    # even while the target is mid-guarded-reversion: both the "rich half was
    # dropped" and the "aggregate ignored/reverted" diagnostics must surface,
    # not just the latter (the gap this test guards was previously silent).
    malformed_with_aggregate = ToolEvent(
        tool="run_tests",
        args={"target": "suite", "run": 2},
        result={
            "passed": 5,
            "failed": 0,
            "test_outcomes": {"a": "unknown"},
            "scope_id": "all",
            "collection_complete": True,
            "scope_exhaustive": True,
        },
    )
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(state, _rich({"a": "failed"}))
    state, snapshot = anchor.step(state, malformed_with_aggregate)
    assert "suite" in dict(state.rich_targets)
    assert snapshot.diagnostics == (
        "target 'suite': aggregate test evidence ignored while in rich mode (1/3 before revert)",
        "target 'suite': malformed test_outcomes payload ignored (aggregate counts still applied)",
    )


def test_legacy_streak_below_threshold_stays_rich_with_diagnostic() -> None:
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(state, _rich({"a": "failed"}))
    state, snapshot = anchor.step(state, _legacy(5, 0, run=1))
    assert "suite" in dict(state.rich_targets)
    assert snapshot.level == 0.0
    assert snapshot.diagnostics == (
        "target 'suite': aggregate test evidence ignored while in rich mode (1/3 before revert)",
    )
    state, snapshot = anchor.step(state, _legacy(5, 0, run=2))
    assert "suite" in dict(state.rich_targets)
    assert snapshot.diagnostics == (
        "target 'suite': aggregate test evidence ignored while in rich mode (2/3 before revert)",
    )


def test_legacy_streak_reset_by_rich_event_never_reverts() -> None:
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(state, _rich({"a": "failed"}))
    state, _ = anchor.step(state, _legacy(5, 0, run=1))
    state, _ = anchor.step(state, _legacy(5, 0, run=2))
    # A rich event in between breaks the streak: two more aggregate-only
    # events afterwards must not add up to the old ones and revert.
    state, _ = anchor.step(state, _rich({"a": "failed"}, run=3))
    state, _ = anchor.step(state, _legacy(5, 0, run=4))
    state, snapshot = anchor.step(state, _legacy(5, 0, run=5))
    assert "suite" in dict(state.rich_targets)
    assert snapshot.diagnostics == (
        "target 'suite': aggregate test evidence ignored while in rich mode (2/3 before revert)",
    )


def test_legacy_streak_reaches_threshold_reverts_to_aggregate() -> None:
    anchor = TestProgressAnchor()
    state = anchor.create_state()
    state, _ = anchor.step(state, _rich({"a": "failed"}))
    state, _ = anchor.step(state, _legacy(5, 0, run=1))
    state, _ = anchor.step(state, _legacy(5, 0, run=2))
    state, snapshot = anchor.step(state, _legacy(5, 1, run=3))
    assert "suite" not in dict(state.rich_targets)
    assert dict(state.legacy_latest)["suite"] == (5, 1)
    assert snapshot.level == 4.0
    assert snapshot.diagnostics == (
        "target 'suite': reverted to aggregate test evidence after 3 "
        "consecutive aggregate-only events; rich debt ledger discarded",
    )


def test_reverted_target_can_resume_rich_mode_without_resurrecting_old_credit() -> None:
    """Documents the residual re-count risk the guard bounds, not eliminates.

    A producer that oscillates rich/aggregate on purpose for the same target
    could still get the *same* debt credited more than once overall: reverting
    discards the ledger that remembers "a" was already fixed once, so a fresh
    rich lifetime after revert starts from a clean slate. This is the accepted
    trade-off (see ``_LEGACY_REVERT_STREAK``'s docstring) -- the guard's job is
    only to make a single stray aggregate-only event harmless, not to make
    deliberate oscillation gameproof.
    """
    anchor = TestProgressAnchor()
    history = [_rich({"a": "failed"}), _rich({"a": "passed"}, run=1)]
    assert anchor.measure(history) == 1.0

    history.extend(_legacy(4, 0, run=index) for index in (2, 3, 4))
    assert anchor.measure(history) == 4.0

    history.append(_rich({"a": "failed"}, run=5))
    assert anchor.measure(history) == 4.0

    history.append(_rich({"a": "passed"}, run=6))
    assert anchor.measure(history) == 5.0


def test_diagnose_is_empty_on_normal_rich_and_legacy_activity() -> None:
    # Negative case at the concrete-anchor level (not just the generic
    # Signal/log_signal plumbing): a healthy mix of legacy and rich activity,
    # with no malformed payload, no cap overflow and no reversion streak, must
    # never produce a diagnostic. A regression here would silently leak noise
    # into every healthy trace's logs.
    anchor = TestProgressAnchor()
    history = [
        _legacy(8, 2, target="old"),
        _rich({"a": "failed", "b": "failed"}, target="new"),
        _rich({"a": "passed", "b": "failed"}, target="new", run=1),
        _rich({"a": "passed", "b": "passed"}, target="new", run=2),
        _legacy(10, 0, target="old", run=1),
    ]
    for prefix_length in range(1, len(history) + 1):
        assert anchor.diagnose(history[:prefix_length]) == ()


def test_anchor_diagnose_matches_incremental_step_across_degraded_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_module, "_MAX_RICH_TEST_TARGETS", 2)
    monkeypatch.setattr(progress_module, "_MAX_RICH_SCOPES_PER_TARGET", 2)
    monkeypatch.setattr(progress_module, "_MAX_TRACKED_TEST_IDS_PER_TARGET", 2)
    anchor = TestProgressAnchor()
    events = [
        _rich({"a": "failed"}, target="a"),
        _rich({"b": "failed"}, target="b"),
        _rich({"c": "failed"}, target="c"),  # exceeds the rich-target cap
        _legacy(5, 0, target="a", run=1),  # streak 1/3 on target "a"
        _legacy(5, 0, target="a", run=2),  # streak 2/3
        _legacy(5, 1, target="a", run=3),  # streak 3/3 -> reverts target "a"
        _rich({"a": "failed"}, target="a", run=4),  # resumes rich on "a"
        _rich({"a": "failed"}, target="a", scope="s2", exhaustive=False, run=5),  # 2nd scope, ok
        _rich(
            {"a": "failed"}, target="a", scope="s3", exhaustive=False, run=6
        ),  # exceeds scope cap
        _rich({"x": "failed", "y": "failed"}, target="b", run=7),  # exceeds per-target id cap
    ]
    state = anchor.create_state()
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, incremental = anchor.step(state, event)
        assert incremental.diagnostics == anchor.diagnose(history)
        assert incremental.level == anchor.measure(history)
        assert incremental.satisfied == anchor.is_satisfied(history)


def test_diagnostics_step_matches_batch_on_no_progress_detector() -> None:
    detector = NoProgressDetector([TestProgressAnchor()])
    malformed = ToolEvent(
        tool="run_tests",
        args={"target": "suite", "run": 1},
        result={
            "test_outcomes": {"a": "unknown"},
            "scope_id": "all",
            "collection_complete": True,
            "scope_exhaustive": True,
        },
    )
    events = [
        _rich({"a": "failed"}),
        malformed,
        _legacy(5, 0, run=2),
        _legacy(5, 0, run=3),
        _legacy(5, 1, run=4),
        _rich({"a": "failed"}, run=5),
    ]
    state = detector.create_state()
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, incremental = detector.step(state, event)
        batch = detector.evaluate(history)
        assert incremental.diagnostics == batch.diagnostics
        assert incremental.confidence == pytest.approx(batch.confidence)


@pytest.mark.parametrize(
    ("relative_path", "intervened"),
    [
        ("healthy/rich_fail_to_pass_progress.json", False),
        ("healthy/rich_late_test_progress.json", False),
        ("healthy/rich_new_target_progress.json", False),
        ("healthy/rich_legacy_revert_recovers.json", False),
        ("no_progress/rich_flat_red.json", True),
        ("no_progress/rich_oscillation.json", True),
        ("no_progress/rich_deleted_failure.json", True),
    ],
)
def test_rich_trace_matrix(relative_path: str, intervened: bool) -> None:
    trace = load_trace(_TRACES / relative_path)
    result = replay_trace(trace)
    assert result.intervened is intervened
    if intervened:
        pipeline = Pipeline(
            [
                LoopDetector(),
                RetryDetector(),
                NoProgressDetector(coding_anchors()),
            ],
            Policy(),
        )
        pipeline.run(trace.events)
        assert pipeline.post_mortem is not None
        assert pipeline.post_mortem.reason.startswith("no_progress")
        assert any(
            signal.name == "no_progress" and signal.confidence >= 0.9
            for signal in pipeline.post_mortem.signals
        )
        dry_run = Pipeline(
            [NoProgressDetector(coding_anchors())],
            Policy(),
            dry_run=True,
        )
        dry_run.run(trace.events)
        assert len(dry_run.history) == len(trace.events)
        assert any(entry.action.name == "STOP" for entry in dry_run.would_be_actions)
        if relative_path.endswith("rich_flat_red.json"):
            assert result.detected_at_index == 11


def test_flat_red_boundary_and_repair_reset_are_prefix_exact() -> None:
    detector = NoProgressDetector([TestProgressAnchor()])
    baseline = _rich({"a": "failed"})
    stagnant = [baseline, *(_edit(i) for i in range(10))]
    confidences = [detector.evaluate(stagnant[:i]).confidence for i in range(1, 12)]
    assert max(confidences[:-1]) < 0.9
    assert confidences[-1] >= 0.9

    repaired_before_boundary = [baseline, *(_edit(i) for i in range(8))]
    repaired_before_boundary.append(_rich({"a": "passed", "b": "failed"}, run=1))
    assert detector.evaluate(repaired_before_boundary).confidence == 0.0


@pytest.mark.parametrize("window", [65, 129])
def test_rich_step_matches_batch_beyond_twice_window(window: int) -> None:
    detector = NoProgressDetector(
        [TestProgressAnchor()],
        window=window,
        target_rate=0.2,
        threshold=2.0,
    )
    events = [_rich({"a": "failed"})]
    events.extend(_edit(index) for index in range(2 * window))
    state = detector.create_state()
    history: list[ToolEvent] = []
    for event in events:
        history.append(event)
        state, incremental = detector.step(state, event)
        batch = detector.evaluate(history)
        assert incremental.confidence == pytest.approx(batch.confidence)
        assert incremental.reason == batch.reason
