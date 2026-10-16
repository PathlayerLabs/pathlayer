"""STATE-1: association rules, collectors, anchor and configuration."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pathlayer._pytest_evidence_protocol import MAX_OUTCOMES
from pathlayer.coding_state import (
    BUILTIN_TOOL_RULES,
    CodingStateError,
    CodingStatePlanner,
    build_state_producer,
    state_evidence_producer_from_config,
    validate_preset_config,
)
from pathlayer.coding_state_collectors import (
    CommandStateSensor,
    GitWorkspaceError,
    GitWorkspaceProbe,
    GitWorkspaceStateSensor,
    collect_command_evidence,
    collect_git_evidence,
    command_outcome_of,
    git_directory,
    git_tree_of,
    validate_command,
)
from pathlayer.config import (
    ConfigError,
    PathlayerConfig,
    StateEvidenceConfig,
    ToolRuleConfig,
    build_pipeline,
)
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.progress import TestProgressAnchor, is_advancing
from pathlayer.detectors.state_progress import (
    UNCLAIMED_GAP,
    UNCLAIMED_HOLD,
    CodingStateProgressAnchor,
)
from pathlayer.events import EventKind, ToolEvent
from pathlayer.state_evidence import (
    STATE_EVIDENCE_METADATA_KEY,
    FilesystemStateSensor,
    StateEvidenceProducer,
    StateOutcome,
    state_evidence_from_event,
)
from pathlayer.store import configuration_fingerprint
from tests.conftest import git_workspace


def planner(workspace: Path, **kwargs: Any) -> CodingStatePlanner:
    rules = {**BUILTIN_TOOL_RULES, **kwargs.pop("rules", {})}
    return CodingStatePlanner(workspace=workspace.resolve(), rules=rules, **kwargs)


def ids(sensors: Any) -> list[str]:
    return [f"{sensor.resource_type}:{sensor.resource_id}" for sensor in sensors]


# --------------------------------------------------------------------------- #
# Association rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tool,args,expected",
    [
        ("write_file", {"file_path": "src/auth.py", "text": "x"}, ["filesystem:src/auth.py"]),
        ("file_delete", {"file_path": "src/auth.py"}, ["filesystem:src/auth.py"]),
        (
            "move_file",
            {"source_path": "a.py", "destination_path": "b.py"},
            ["filesystem:a.py", "filesystem:b.py"],
        ),
        ("copy_file", {"source_path": "a.py", "destination_path": "b.py"}, ["filesystem:b.py"]),
        ("editor", {"command": "create", "path": "new.py"}, ["filesystem:new.py"]),
        ("editor", {"command": "str_replace", "path": "src/auth.py"}, ["filesystem:src/auth.py"]),
        # A call naming the same file twice is one resource, not two.
        ("move_file", {"source_path": "a.py", "destination_path": "./a.py"}, ["filesystem:a.py"]),
    ],
)
def test_a_call_naming_files_observes_exactly_those_files(
    tmp_path: Path, tool: str, args: dict[str, Any], expected: list[str]
) -> None:
    assert ids(planner(tmp_path).plan(tool=tool, args=args)) == expected


@pytest.mark.parametrize(
    "tool,args",
    [
        ("read_file", {"file_path": "src/auth.py"}),
        ("list_directory", {"dir_path": "src"}),
        ("file_search", {"dir_path": "src", "pattern": "*.py"}),
        ("editor", {"command": "view", "path": "src/auth.py"}),
        # An unknown tool, and a shape we do not recognize, claim nothing.
        ("run_shell", {"command": "ls -al"}),
        ("editor", {"command": "sync", "path": "src/auth.py"}),
        ("mystery", {"file_path": "src/auth.py"}),
    ],
)
def test_a_call_that_claims_no_state_is_not_observed(
    tmp_path: Path, tool: str, args: dict[str, Any]
) -> None:
    assert planner(tmp_path).plan(tool=tool, args=args) == ()


def test_an_absolute_path_inside_the_workspace_is_the_same_resource(tmp_path: Path) -> None:
    absolute = (tmp_path / "src" / "auth.py").resolve()
    assert ids(planner(tmp_path).plan(tool="write_file", args={"file_path": str(absolute)})) == [
        "filesystem:src/auth.py"
    ]


@pytest.mark.parametrize(
    "value",
    [
        "../outside.py",
        "/etc/passwd",
        "",
        "a\x00b",
        "back\\slash.py",
        "x" * 1100,
        None,
        42,
        ["a.py", None],
        [],
        # More files than one call can be credibly attributed to.
        [f"f{index}.py" for index in range(9)],
    ],
)
def test_an_unattributable_path_falls_back_to_the_configured_scope(
    tmp_path: Path, value: Any
) -> None:
    workspace = git_workspace(tmp_path / "ws")
    git = GitWorkspaceStateSensor(GitWorkspaceProbe(workspace))
    rules = {"bulk_write": ToolRuleConfig(effect="write", paths=("file_path",))}

    with_scope = planner(workspace, git=git, rules=rules)
    assert ids(with_scope.plan(tool="bulk_write", args={"file_path": value})) == ["git:workspace"]

    # With no configured scope there is nothing honest left to observe.
    assert planner(workspace, rules=rules).plan(tool="bulk_write", args={"file_path": value}) == ()


def test_a_partially_resolvable_call_is_attributed_to_neither_file(tmp_path: Path) -> None:
    args = {"source_path": "a.py", "destination_path": "../elsewhere.py"}
    assert planner(tmp_path).plan(tool="move_file", args=args) == ()


def test_a_declared_workspace_tool_observes_the_scope_only(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    rules = {"apply_patch": ToolRuleConfig(effect="workspace")}
    plan = planner(
        workspace, git=GitWorkspaceStateSensor(GitWorkspaceProbe(workspace)), rules=rules
    ).plan(tool="apply_patch", args={"patch": "--- a/x"})
    assert ids(plan) == ["git:workspace"]


def test_the_configured_scope_joins_every_observed_call(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    plan = planner(
        workspace,
        git=GitWorkspaceStateSensor(GitWorkspaceProbe(workspace)),
        build=CommandStateSensor(command=("true",), cwd=workspace, resource_id="build"),
    ).plan(tool="write_file", args={"file_path": "src/auth.py"})
    assert ids(plan) == ["filesystem:src/auth.py", "git:workspace", "command:build"]


def test_a_user_rule_overrides_a_builtin_one(tmp_path: Path) -> None:
    rules = {"read_file": ToolRuleConfig(effect="write", paths=("file_path",))}
    plan = planner(tmp_path, rules=rules).plan(tool="read_file", args={"file_path": "a.py"})
    assert ids(plan) == ["filesystem:a.py"]


# --------------------------------------------------------------------------- #
# Git collector
# --------------------------------------------------------------------------- #


def observe(probe: GitWorkspaceProbe, ref: str = "e1") -> Any:
    return collect_git_evidence(probe, tool_event_ref=ref, resource_id="workspace", phase="before")


def test_git_distinguishes_a_real_change_from_no_change(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    probe = GitWorkspaceProbe(workspace)

    first = observe(probe)
    unchanged = observe(probe, "e2")
    assert first.status == "observed"
    assert first.canonical_fingerprint == unchanged.canonical_fingerprint

    (workspace / "src" / "auth.py").write_text("def login():\n    return True\n", encoding="utf-8")
    changed = observe(probe, "e3")
    assert changed.canonical_fingerprint != first.canonical_fingerprint
    assert git_tree_of(changed) != git_tree_of(first)


def test_git_fingerprints_content_not_history(tmp_path: Path) -> None:
    """Committing, staging or amending is not a change of the working tree."""
    workspace = git_workspace(tmp_path / "ws")
    probe = GitWorkspaceProbe(workspace)
    (workspace / "src" / "auth.py").write_text("changed\n", encoding="utf-8")
    edited = observe(probe)

    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    assert observe(probe, "e2").canonical_fingerprint == edited.canonical_fingerprint
    subprocess.run(["git", "commit", "-qm", "second"], cwd=workspace, check=True)
    assert observe(probe, "e3").canonical_fingerprint == edited.canonical_fingerprint


@pytest.mark.parametrize(
    "change,expected_change",
    [
        (lambda ws: (ws / "untracked.py").write_text("new\n", encoding="utf-8"), True),
        (lambda ws: (ws / "src" / "auth.py").unlink(), True),
        (lambda ws: _write_ignored(ws), False),
    ],
)
def test_git_sees_what_the_repository_says_is_content(
    tmp_path: Path, change: Any, expected_change: bool
) -> None:
    workspace = git_workspace(tmp_path / "ws")
    probe = GitWorkspaceProbe(workspace)
    before = observe(probe)
    change(workspace)
    after = observe(probe, "e2")
    assert (before.canonical_fingerprint != after.canonical_fingerprint) is expected_change


def _write_ignored(workspace: Path) -> None:
    (workspace / "build").mkdir(exist_ok=True)
    (workspace / "build" / "artifact.bin").write_text("junk", encoding="utf-8")


def test_git_names_the_paths_that_changed(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    sensor = GitWorkspaceStateSensor(GitWorkspaceProbe(workspace))
    before = sensor.capture(tool_event_ref="e1", phase="before")
    (workspace / "src" / "auth.py").write_text("edited\n", encoding="utf-8")
    after = sensor.capture(tool_event_ref="e1", phase="after", before=before)

    assert after.delta == {"changed": True, "paths": ["src/auth.py"], "truncated": False}


def test_git_observation_never_writes_to_the_observed_repository(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    git_dir = workspace / ".git"
    index_before = (git_dir / "index").read_bytes()
    objects_before = sorted(path.name for path in (git_dir / "objects").rglob("*"))

    probe = GitWorkspaceProbe(workspace)
    observe(probe)
    (workspace / "src" / "auth.py").write_text("edited\n", encoding="utf-8")
    observe(probe, "e2")

    assert (git_dir / "index").read_bytes() == index_before
    assert sorted(path.name for path in (git_dir / "objects").rglob("*")) == objects_before
    assert not (git_dir / "index.lock").exists()


def test_a_workspace_without_git_fails_at_build_and_never_mid_run(tmp_path: Path) -> None:
    with pytest.raises(GitWorkspaceError, match="not a git repository"):
        git_directory(tmp_path)
    # A repository that disappears under a live run degrades explicitly instead.
    workspace = git_workspace(tmp_path / "ws")
    probe = GitWorkspaceProbe(workspace)
    assert observe(probe).status == "observed"
    broken = GitWorkspaceProbe(tmp_path / "plain")
    (tmp_path / "plain").mkdir()
    assert observe(broken).status == "collection_failed"
    assert broken.changed_paths("a" * 40, "b" * 40) is None


def test_a_git_fingerprint_that_disagrees_with_itself_is_rejected(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    evidence = observe(GitWorkspaceProbe(workspace))
    for mutation in [
        {"raw_fingerprint": {"tree": "nope"}},
        {"raw_fingerprint": {"other": "a" * 40}},
        {"raw_fingerprint": "a" * 40},
        {"canonical_fingerprint": "tree:" + "b" * 40},
    ]:
        with pytest.raises(ValueError, match="invalid fingerprint"):
            git_tree_of(replace(evidence, **mutation))


# --------------------------------------------------------------------------- #
# Command collector
# --------------------------------------------------------------------------- #


def run_command(command: tuple[str, ...], tmp_path: Path, **kwargs: Any) -> Any:
    return collect_command_evidence(
        command,
        cwd=tmp_path,
        tool_event_ref="e1",
        resource_id="build",
        phase="before",
        **kwargs,
    )


def test_command_evidence_reports_success_failure_and_absence(tmp_path: Path) -> None:
    passing = run_command((sys.executable, "-c", "raise SystemExit(0)"), tmp_path)
    failing = run_command((sys.executable, "-c", "raise SystemExit(2)"), tmp_path)
    assert command_outcome_of(passing) == "success"
    assert command_outcome_of(failing) == "failure"
    # Exit statuses are not distinguished: churn between error codes is not progress.
    other = run_command((sys.executable, "-c", "raise SystemExit(3)"), tmp_path)
    assert other.canonical_fingerprint == failing.canonical_fingerprint

    assert run_command(("pathlayer-does-not-exist",), tmp_path).status == "collection_failed"
    slow = run_command(
        (sys.executable, "-c", "import time; time.sleep(30)"), tmp_path, timeout_seconds=0.5
    )
    assert slow.status == "collection_failed"


def test_command_delta_reports_the_transition(tmp_path: Path) -> None:
    marker = tmp_path / "ok"
    command = (
        sys.executable,
        "-c",
        f"import os,sys; sys.exit(0 if os.path.exists({str(marker)!r}) else 1)",
    )
    sensor = CommandStateSensor(command=command, cwd=tmp_path, resource_id="build")
    before = sensor.capture(tool_event_ref="e1", phase="before")
    marker.write_text("", encoding="utf-8")
    after = sensor.capture(tool_event_ref="e1", phase="after", before=before)
    assert after.delta == {"changed": True, "before": "failure", "after": "success"}


@pytest.mark.parametrize("command", ["not a list", (), ("",), ("a\x00b",), ("x" * 5000,), (1,)])
def test_a_malformed_command_is_rejected(command: Any) -> None:
    with pytest.raises(ValueError):
        validate_command(command)


def test_a_malformed_command_timeout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_command(("true",), tmp_path, timeout_seconds=0)


def test_a_command_fingerprint_that_disagrees_with_itself_is_rejected(tmp_path: Path) -> None:
    evidence = run_command((sys.executable, "-c", "pass"), tmp_path)
    for mutation in [{"raw_fingerprint": {"outcome": "maybe"}}, {"canonical_fingerprint": "other"}]:
        with pytest.raises(ValueError, match="invalid fingerprint"):
            command_outcome_of(replace(evidence, **mutation))


# --------------------------------------------------------------------------- #
# Producer planning
# --------------------------------------------------------------------------- #


def test_a_producer_needs_exactly_one_of_sensors_or_planner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either sensors or a planner"):
        StateEvidenceProducer()
    with pytest.raises(ValueError, match="either sensors or a planner"):
        StateEvidenceProducer(
            [FilesystemStateSensor(tmp_path / "a", resource_id="a")], planner=planner(tmp_path)
        )
    with pytest.raises(ValueError, match="plans its sensors per tool call"):
        StateEvidenceProducer(planner=planner(tmp_path)).capture_before("e1")


def test_a_planned_call_carries_only_its_own_resources(tmp_path: Path) -> None:
    producer = StateEvidenceProducer(planner=planner(tmp_path))
    (tmp_path / "a.py").write_text("one", encoding="utf-8")

    assert producer.plan_capture("e1", tool="read_file", args={"file_path": "a.py"}) is None
    capture = producer.plan_capture("e1", tool="write_file", args={"file_path": "a.py"})
    assert capture is not None
    (tmp_path / "a.py").write_text("two", encoding="utf-8")
    event = producer.attach_after(ToolEvent(tool="write_file", run_id="e1"), capture)

    envelope = state_evidence_from_event(event)
    assert envelope is not None
    assert envelope.outcome is StateOutcome.CHANGED
    assert [record.resource_id for record in envelope.records] == ["a.py", "a.py"]


def test_a_withdrawn_capture_keeps_its_resources_and_claims_nothing(tmp_path: Path) -> None:
    producer = StateEvidenceProducer(planner=planner(tmp_path))
    capture = producer.plan_capture("e1", tool="write_file", args={"file_path": "a.py"})
    assert capture is not None
    withdrawn = producer.unavailable_for(capture)
    event = producer.attach_after(ToolEvent(tool="write_file", run_id="e1"), withdrawn)

    envelope = state_evidence_from_event(event)
    assert envelope is not None
    assert envelope.outcome is StateOutcome.UNAVAILABLE
    assert {record.resource_id for record in envelope.records} == {"a.py"}


class _BrokenPlanner:
    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour

    def plan(self, *, tool: str, args: Any) -> Any:
        if self.behaviour == "raise":
            raise RuntimeError("planner is broken")
        if self.behaviour == "duplicate":
            sensor = FilesystemStateSensor(Path("a"), resource_id="a")
            return [sensor, sensor]
        return [
            FilesystemStateSensor(Path(f"f{index}"), resource_id=f"f{index}") for index in range(65)
        ]


@pytest.mark.parametrize("behaviour", ["raise", "duplicate", "too many"])
def test_an_unusable_plan_leaves_the_call_unobserved(
    behaviour: str, caplog: pytest.LogCaptureFixture
) -> None:
    producer = StateEvidenceProducer(planner=_BrokenPlanner(behaviour))
    with caplog.at_level(logging.WARNING):
        assert producer.plan_capture("e1", tool="write_file", args={}) is None
    assert caplog.records


# --------------------------------------------------------------------------- #
# Anchor
# --------------------------------------------------------------------------- #


def observed_event(
    tmp_path: Path, name: str, content: str, *, run_id: str = "e1", tool: str = "write_file"
) -> ToolEvent:
    """A real planned capture around a real write, as the adapter would record it."""
    producer = StateEvidenceProducer(planner=planner(tmp_path))
    capture = producer.plan_capture(run_id, tool=tool, args={"file_path": name})
    assert capture is not None
    (tmp_path / name).write_text(content, encoding="utf-8")
    return producer.attach_after(ToolEvent(tool=tool, run_id=run_id), capture)


def test_novelty_is_credited_once_per_resource_state(tmp_path: Path) -> None:
    anchor = CodingStateProgressAnchor()
    first = observed_event(tmp_path, "a.py", "one")
    second = observed_event(tmp_path, "b.py", "one", run_id="e2")
    # Same content, different resource: a new state of b.py, credited.
    assert anchor.measure([first]) == 1.0
    assert anchor.measure([first, second]) == 2.0
    # Returning a.py to a content already seen is not progress.
    back = observed_event(tmp_path, "a.py", "", run_id="e3")
    forward = observed_event(tmp_path, "a.py", "one", run_id="e4")
    assert anchor.measure([first, second, back]) == 3.0
    assert anchor.measure([first, second, back, forward]) == 3.0


def test_a_rewrite_with_identical_content_is_not_progress(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("same", encoding="utf-8")
    events = [observed_event(tmp_path, "a.py", "same", run_id=f"e{i}") for i in range(4)]
    assert CodingStateProgressAnchor().measure(events) == 0.0


@pytest.mark.parametrize("mode", [UNCLAIMED_GAP, UNCLAIMED_HOLD])
def test_an_unclaimed_call_has_no_level_before_anything_was_observed(
    tmp_path: Path, mode: str
) -> None:
    read = ToolEvent(tool="read_file", args={"file_path": "a.py"}, run_id="e1")
    assert CodingStateProgressAnchor(unclaimed=mode).measure([read]) is None


def test_reading_never_looks_like_stagnation_to_the_detector(tmp_path: Path) -> None:
    """The false stop this preset must not cause: a long exploration phase.

    Reading files after an edit is normal work, and the ``gap`` anchor -- the
    only one ``build_pipeline`` gives ``NoProgressDetector`` -- never turns it
    into a stalled step.
    """
    write = observed_event(tmp_path, "a.py", "one")
    reads = [
        ToolEvent(tool="read_file", args={"file_path": "a.py"}, run_id=f"r{i}") for i in range(30)
    ]
    gap = CodingStateProgressAnchor(unclaimed=UNCLAIMED_GAP)
    assert gap.measure([write, *reads]) is None
    assert NoProgressDetector([gap]).evaluate([write, *reads]).confidence == 0.0


def test_holding_keeps_a_baseline_for_suppression_and_stays_out_of_no_progress(
    tmp_path: Path,
) -> None:
    """Why the two modes exist, stated as the behaviour that separates them.

    Loop suppression compares the level to the previous event's: a gap between
    two edits would leave the second edit with no baseline and no evidence of
    progress. Holding fixes that -- and is exactly why the same anchor must
    never reach ``NoProgressDetector``, where a held level is a stalled step.
    """
    write = observed_event(tmp_path, "a.py", "one")
    read = ToolEvent(tool="read_file", args={"file_path": "a.py"}, run_id="r1")
    second = observed_event(tmp_path, "a.py", "two", run_id="e3")
    hold = CodingStateProgressAnchor(unclaimed=UNCLAIMED_HOLD)

    assert hold.measure([write, read]) == 1.0
    assert is_advancing([hold], [write, read, second])
    assert not is_advancing(
        [CodingStateProgressAnchor(unclaimed=UNCLAIMED_GAP)], [write, read, second]
    )
    assert NoProgressDetector([hold]).evaluate([write, *[read] * 30]).confidence == 1.0


@pytest.mark.parametrize("mode", [UNCLAIMED_GAP, UNCLAIMED_HOLD])
def test_an_unavailable_observation_is_a_gap_under_both_modes(tmp_path: Path, mode: str) -> None:
    producer = StateEvidenceProducer(planner=planner(tmp_path))
    capture = producer.plan_capture("e1", tool="write_file", args={"file_path": "a.py"})
    assert capture is not None
    event = producer.attach_after(
        ToolEvent(tool="write_file", run_id="e1"), producer.unavailable_for(capture)
    )
    anchor = CodingStateProgressAnchor(unclaimed=mode)
    assert anchor.measure([event]) is None
    assert anchor.diagnose([event]) == ("state evidence unavailable",)


def test_a_failed_collection_is_a_gap_not_an_unchanged_state(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    producer = StateEvidenceProducer(planner=planner(tmp_path))
    capture = producer.plan_capture("e1", tool="write_file", args={"file_path": "dir"})
    assert capture is not None
    event = producer.attach_after(ToolEvent(tool="write_file", run_id="e1"), capture)

    assert state_evidence_from_event(event).outcome is StateOutcome.COLLECTION_FAILED
    anchor = CodingStateProgressAnchor()
    assert anchor.measure([event]) is None
    assert anchor.diagnose([event]) == ("state evidence collection_failed",)


def test_the_anchor_steps_aside_for_objective_evidence(tmp_path: Path) -> None:
    """An observed effect must never keep a stalled objective quiet."""
    novel = [observed_event(tmp_path, "a.py", str(index), run_id=f"e{index}") for index in range(4)]
    tests = replace(novel[1], result={"passed": 1, "failed": 3})
    proof = replace(novel[2], metadata={**novel[2].metadata, "progress_proof": {"checkpoints": {}}})

    for objective in (tests, proof):
        history = [novel[0], objective, novel[3]]
        deferring = CodingStateProgressAnchor(defer_to_objective=True)
        assert deferring.measure([novel[0]]) == 1.0
        assert deferring.measure(history) is None
        assert deferring.diagnose(history[:2]) == ("deferring to objective evidence",)
        # ...and stays aside afterwards, so it cannot flip back on the next call.
        assert deferring.measure([*history, novel[3]]) is None
        assert CodingStateProgressAnchor().measure(history) == 3.0


def test_effect_novelty_cannot_mask_a_stalled_objective(tmp_path: Path) -> None:
    """The regression the deferral exists to prevent, through the real detector."""
    events = [
        replace(
            observed_event(tmp_path, "a.py", f"attempt {index}", run_id=f"e{index}"),
            result={"passed": 0, "failed": 1},
        )
        for index in range(12)
    ]
    anchors = [TestProgressAnchor(), CodingStateProgressAnchor(defer_to_objective=True)]
    assert NoProgressDetector(anchors).evaluate(events).confidence == 1.0
    # Without the deferral, every fresh wrong state would look like progress.
    naive = [TestProgressAnchor(), CodingStateProgressAnchor()]
    assert NoProgressDetector(naive).evaluate(events).confidence == 0.0


def test_unsupported_or_inconsistent_evidence_is_a_gap(tmp_path: Path) -> None:
    event = observed_event(tmp_path, "a.py", "one")
    anchor = CodingStateProgressAnchor()

    unsupported = _mutate_record(event, (0, 1), {"collector_version": "someone-elses@9"})
    assert anchor.measure([unsupported]) is None
    assert anchor.diagnose([unsupported]) == ("unsupported or invalid state evidence",)

    inconsistent = _mutate_record(event, (1,), {"canonical_fingerprint": "sha256:" + "0" * 64})
    assert anchor.measure([inconsistent]) is None


def _mutate_record(
    event: ToolEvent, indices: tuple[int, ...], changes: dict[str, Any]
) -> ToolEvent:
    envelope = json.loads(json.dumps(event.metadata[STATE_EVIDENCE_METADATA_KEY]))
    for index in indices:
        envelope["records"][index].update(changes)
    if indices == (1,):
        # Keep the envelope self-consistent so the reader accepts it and the
        # anchor is what rejects the fingerprint.
        envelope["outcome"] = StateOutcome.CHANGED.value
    return replace(event, metadata={**event.metadata, STATE_EVIDENCE_METADATA_KEY: envelope})


def test_capacity_is_reached_once_and_then_stays_quiet(tmp_path: Path) -> None:
    anchor = CodingStateProgressAnchor(max_states=2)
    events = [
        observed_event(tmp_path, "a.py", str(index), run_id=f"e{index}") for index in range(3)
    ]
    assert anchor.measure(events[:1]) == 1.0
    assert anchor.measure(events[:2]) is None
    assert anchor.diagnose(events[:2]) == ("state evidence capacity reached",)
    assert anchor.measure(events) is None
    assert anchor.diagnose(events) == ()


def test_step_matches_measure_and_diagnose_for_every_prefix(tmp_path: Path) -> None:
    history = [
        observed_event(tmp_path, "a.py", "one"),
        ToolEvent(tool="read_file", run_id="e2"),
        observed_event(tmp_path, "a.py", "two", run_id="e3"),
        ToolEvent(tool="model", kind=EventKind.LLM_CALL, run_id="e4"),
        replace(observed_event(tmp_path, "a.py", "three", run_id="e5"), result={"passed": 1}),
        observed_event(tmp_path, "a.py", "four", run_id="e6"),
    ]
    for mode in (UNCLAIMED_GAP, UNCLAIMED_HOLD):
        for defer in (False, True):
            anchor = CodingStateProgressAnchor(unclaimed=mode, defer_to_objective=defer)
            state = anchor.create_state()
            detector = NoProgressDetector([anchor])
            detector_state = detector.create_state()
            for index, event in enumerate(history):
                state, snapshot = anchor.step(state, event)
                prefix = history[: index + 1]
                assert snapshot.level == anchor.measure(prefix)
                assert snapshot.diagnostics == anchor.diagnose(prefix)
                detector_state, signal = detector.step(detector_state, event)
                assert signal == detector.evaluate(prefix)


@pytest.mark.parametrize(
    "kwargs", [{"unclaimed": "maybe"}, {"max_states": 1}, {"max_states": True}]
)
def test_bad_anchor_configuration_is_rejected(kwargs: Any) -> None:
    with pytest.raises(ValueError):
        CodingStateProgressAnchor(**kwargs)


def test_the_anchor_rejects_a_foreign_state(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        CodingStateProgressAnchor().step(None, observed_event(tmp_path, "a.py", "one"))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def config_mapping(root: Path, **overrides: Any) -> dict[str, Any]:
    section: dict[str, Any] = {
        "enabled": True,
        "preset": "coding",
        "workspace": str(root),
        **overrides,
    }
    return {"detectors": {"state_evidence": section}}


def test_a_short_configuration_wires_both_halves_of_the_preset(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    config = PathlayerConfig.from_mapping(config_mapping(workspace))

    producer = state_evidence_producer_from_config(config)
    assert producer is not None
    pipeline = build_pipeline(config)

    no_progress = next(d for d in pipeline._detectors if d.name == "no_progress")
    anchors = {type(anchor).__name__ for anchor in no_progress._anchors}
    assert "CodingStateProgressAnchor" in anchors
    suppression = [
        anchor
        for anchor in pipeline._progress_anchors
        if isinstance(anchor, CodingStateProgressAnchor)
    ]
    assert [anchor.unclaimed for anchor in suppression] == [UNCLAIMED_HOLD]
    gap = next(a for a in no_progress._anchors if isinstance(a, CodingStateProgressAnchor))
    assert (gap.unclaimed, gap.defer_to_objective) == (UNCLAIMED_GAP, True)


def test_zero_config_stays_exactly_as_it_was(tmp_path: Path) -> None:
    default = PathlayerConfig()
    assert default.detectors.state_evidence.enabled is False
    assert state_evidence_producer_from_config(default) is None
    pipeline = build_pipeline(default)
    assert not any(isinstance(a, CodingStateProgressAnchor) for a in pipeline._progress_anchors)
    # The fingerprint of every pre-STATE-1 configuration is untouched.
    assert configuration_fingerprint(default) == configuration_fingerprint(
        PathlayerConfig.from_mapping({"detectors": {"state_evidence": {"enabled": False}}})
    )


def test_enabling_the_preset_changes_the_durable_fingerprint(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    enabled = PathlayerConfig.from_mapping(config_mapping(workspace))
    other_scope = PathlayerConfig.from_mapping(config_mapping(workspace, git=False))
    assert configuration_fingerprint(enabled) != configuration_fingerprint(PathlayerConfig())
    assert configuration_fingerprint(enabled) != configuration_fingerprint(other_scope)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"preset": "browser"}, "unknown state evidence preset"),
        # A budget the collectors cannot spend would turn every observation
        # into a collection_failed at run time (Kropemann's review).
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"tests": ["-k", "x"]}, "invalid pytest nodeid"),
        ({"tests": ["a::b", "a::b"]}, "must not repeat"),
        ({"build": {"id": "b"}}, "requires 'command'"),
        ({"build": {"command": ["true"]}}, "requires an 'id'"),
        ({"build": {"id": "b", "command": [""]}}, "invalid 'detectors.state_evidence.build"),
        ({"tools": {"t": {"effect": "sideways"}}}, "invalid effect"),
        ({"tools": {"t": {"effect": "write"}}}, "names no path argument"),
        ({"tools": {"t": {"effect": "read", "paths": ["p"]}}}, "not 'write'"),
    ],
)
def test_a_malformed_preset_fails_the_build_not_the_run(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    workspace = git_workspace(tmp_path / "ws")
    config = PathlayerConfig.from_mapping(config_mapping(workspace, **overrides))
    with pytest.raises(CodingStateError, match=message):
        build_pipeline(config)
    with pytest.raises(CodingStateError, match=message):
        state_evidence_producer_from_config(config)


def test_a_workspace_without_git_fails_the_capture_side_unless_git_is_off(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    config = PathlayerConfig.from_mapping(config_mapping(plain))
    with pytest.raises(CodingStateError, match="not a git repository"):
        state_evidence_producer_from_config(config)
    assert state_evidence_producer_from_config(
        PathlayerConfig.from_mapping(config_mapping(plain, git=False))
    )


def test_a_recorded_run_can_be_replayed_where_the_workspace_does_not_exist(tmp_path: Path) -> None:
    """The consumer side reads recorded evidence, so it must build anywhere.

    A post-mortem replay on another machine has no workspace and no repository;
    only the capture side needs them.
    """
    config = PathlayerConfig.from_mapping(config_mapping(tmp_path / "gone-for-good"))
    assert build_pipeline(config)
    with pytest.raises(CodingStateError, match="not an existing directory"):
        state_evidence_producer_from_config(config)


def test_a_workspace_rule_requires_the_git_scope(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    config = PathlayerConfig.from_mapping(
        config_mapping(plain, git=False, tools={"patch": {"effect": "workspace"}})
    )
    with pytest.raises(CodingStateError, match="requires"):
        build_state_producer(config.detectors.state_evidence)


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    for section in [
        {"enabled": True, "typo": 1},
        {"build": {"nope": 1}},
        {"tools": {"t": {"x": 1}}},
    ]:
        with pytest.raises(ConfigError, match="unknown key"):
            PathlayerConfig.from_mapping({"detectors": {"state_evidence": section}})


def test_an_enabled_preset_without_no_progress_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    workspace = git_workspace(tmp_path / "ws")
    config = PathlayerConfig.from_mapping(
        {
            **config_mapping(workspace),
            "detectors": {
                **config_mapping(workspace)["detectors"],
                "no_progress": {"enabled": False},
            },
        }
    )
    with caplog.at_level(logging.WARNING):
        build_pipeline(config)
    assert "nowhere to wire" in caplog.text


def test_a_configured_selection_is_observed_around_a_write(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    (workspace / "test_unit.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    config = PathlayerConfig.from_mapping(
        config_mapping(workspace, tests=["test_unit.py::test_ok"], timeout_seconds=120.0)
    )
    producer = state_evidence_producer_from_config(config)
    assert producer is not None

    capture = producer.plan_capture("e1", tool="write_file", args={"file_path": "src/auth.py"})
    assert capture is not None
    (workspace / "src" / "auth.py").write_text("edited\n", encoding="utf-8")
    event = producer.attach_after(ToolEvent(tool="write_file", run_id="e1"), capture)

    envelope = state_evidence_from_event(event)
    assert envelope is not None
    assert [(r.resource_type, r.resource_id) for r in envelope.records[:6:2]] == [
        ("filesystem", "src/auth.py"),
        ("git", "workspace"),
        ("tests", "selection"),
    ]
    assert envelope.outcome is StateOutcome.CHANGED


def test_a_wide_change_lists_a_bounded_sample_of_paths(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path / "ws")
    sensor = GitWorkspaceStateSensor(GitWorkspaceProbe(workspace))
    before = sensor.capture(tool_event_ref="e1", phase="before")
    for index in range(70):
        (workspace / f"file_{index:03d}.py").write_text(f"{index}\n", encoding="utf-8")
    after = sensor.capture(tool_event_ref="e1", phase="after", before=before)

    assert after.delta is not None
    assert after.delta["changed"] is True
    assert after.delta["truncated"] is True
    assert len(after.delta["paths"]) == 64


def test_a_repository_borrowing_objects_is_still_observable(tmp_path: Path) -> None:
    """A shared clone keeps its objects elsewhere; the probe must still read them."""
    origin = git_workspace(tmp_path / "origin")
    borrower = tmp_path / "borrower"
    subprocess.run(
        ["git", "clone", "-q", "--shared", str(origin), str(borrower)], cwd=tmp_path, check=True
    )
    probe = GitWorkspaceProbe(borrower)
    first = observe(probe)
    assert first.status == "observed"
    (borrower / "src" / "auth.py").write_text("edited\n", encoding="utf-8")
    assert observe(probe, "e2").canonical_fingerprint != first.canonical_fingerprint


def test_an_unusable_workspace_is_reported_as_configuration(tmp_path: Path) -> None:
    file_not_directory = tmp_path / "notes.txt"
    file_not_directory.write_text("x", encoding="utf-8")
    section = StateEvidenceConfig(enabled=True, workspace=str(file_not_directory))
    with pytest.raises(CodingStateError, match="not an existing directory"):
        build_state_producer(section)


def test_a_relative_workspace_resolves_against_the_given_base(tmp_path: Path) -> None:
    git_workspace(tmp_path / "ws")
    producer = build_state_producer(
        StateEvidenceConfig(enabled=True, workspace="ws"), base_dir=tmp_path
    )
    capture = producer.plan_capture("e1", tool="write_file", args={"file_path": "src/auth.py"})
    assert capture is not None
    assert [record.resource_id for record in capture.before] == ["src/auth.py", "workspace"]


def test_a_repository_that_breaks_mid_run_degrades_explicitly(tmp_path: Path) -> None:
    """git working at build time is no guarantee it still works at call 200."""
    workspace = git_workspace(tmp_path / "ws")
    probe = GitWorkspaceProbe(workspace)
    assert observe(probe).status == "observed"

    for path in sorted((workspace / ".git").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (workspace / ".git").rmdir()

    failed = observe(probe, "e2")
    assert failed.status == "collection_failed"
    assert failed.canonical_fingerprint is None
    assert probe.changed_paths("a" * 40, "b" * 40) is None


def test_a_symlink_loop_is_not_attributable(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "loop").symlink_to(workspace / "loop")
    assert planner(workspace).plan(tool="write_file", args={"file_path": "loop/x.py"}) == ()


def test_too_many_configured_nodeids_are_rejected() -> None:
    section = StateEvidenceConfig(
        enabled=True, tests=tuple(f"t.py::test_{index}" for index in range(MAX_OUTCOMES + 1))
    )
    with pytest.raises(CodingStateError, match="at most"):
        validate_preset_config(section)


@pytest.mark.parametrize("timeout", [0, 0.0, -1.0, float("inf"), float("nan")])
def test_an_unspendable_timeout_is_a_configuration_error(tmp_path: Path, timeout: float) -> None:
    """The collectors require a positive budget, so the config must too.

    Accepting `timeout_seconds: 0` would let a whole run report degraded
    collectors instead of the configuration mistake that caused them.
    """
    workspace = git_workspace(tmp_path / "ws")
    with pytest.raises(ConfigError, match="timeout_seconds"):
        build_pipeline(
            PathlayerConfig.from_mapping(config_mapping(workspace, timeout_seconds=timeout))
        )
