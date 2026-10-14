"""VAL-7: session continuity must not become shared runtime lifecycle."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pathlayer.config import PathlayerConfig, PipelineSection, StoreSection, build_pipeline
from pathlayer.detectors.proof import compute_spec_hash, validate_proof_spec
from pathlayer.detectors.session import continuity_key, session_history
from pathlayer.events import EventKind, ToolEvent
from pathlayer.policies import Action
from pathlayer.session import SessionJournal
from pathlayer.store import configuration_fingerprint, encode_event


def config(root: Path) -> PathlayerConfig:
    return PathlayerConfig(
        pipeline=PipelineSection(session_stitching=True),
        store=StoreSection(path=str(root / "state.db")),
    )


def observation(run: str, sequence: int = 0, *, passed: bool = False) -> ToolEvent:
    spec = {
        "schema_version": 1,
        "task_id": "task",
        "proof_id": "tests",
        "scope_id": "workspace",
        "kind": "checkpoint_set",
        "required": ["test"],
    }
    validated = validate_proof_spec(spec)
    assert validated is not None
    return ToolEvent(
        tool="shell",
        args={"cmd": "build"},
        run_id=run,
        metadata={
            "error": "compiler error",
            "proof_spec": spec,
            "progress_proof": {
                "spec_hash": compute_spec_hash(validated),
                "sequence": sequence,
                "checkpoints": {"test": passed},
            },
        },
    )


def capture(
    journal: SessionJournal,
    event: ToolEvent,
    *,
    session: str = "s",
    run: str | None = None,
    identity: str = "e",
    source: str = "test",
) -> ToolEvent:
    return journal.enrich(
        event,
        source=source,
        session_key=session,
        run_id=run or event.run_id or "r",
        event_id=identity,
    )


def test_three_runs_repeat_and_replay_same_decisions(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint=configuration_fingerprint(cfg))
    emitted = []
    events = []
    for run in ("z", "a", "m"):
        event = capture(journal, observation(run))
        events.append(event)
        pipeline = build_pipeline(cfg)
        emitted.append(pipeline.process(event))
        replay = build_pipeline(cfg)
        assert replay.process(event) == emitted[-1]
        assert replay.last_decision == pipeline.last_decision
    assert emitted == [Action.NONE, Action.NONE, Action.STOP]
    report = journal.report(source="test", session_key="s")
    assert len(report["incidents"]) == 1
    assert report["incidents"][0]["run_ids"] == ["z", "a", "m"]
    assert report["incidents"][0]["occurrences"] == 3
    assert encode_event(events[0]) == encode_event(capture(journal, observation("z")))


@pytest.mark.parametrize(
    "gap", ["missing", "invalid", "unavailable", "passed", "different", "partial"]
)
def test_gap_or_progress_never_inherits_a_previous_incident(tmp_path: Path, gap: str) -> None:
    cfg = config(tmp_path)
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint=configuration_fingerprint(cfg))
    capture(journal, observation("a"))
    event = observation("b")
    if gap == "missing":
        event.metadata = {}
    elif gap == "invalid":
        event.metadata["progress_proof"]["spec_hash"] = "invalid"
    elif gap == "unavailable":
        event.metadata["progress_proof_unobserved"] = {}
    elif gap == "passed":
        event = observation("b", passed=True)
    elif gap == "different":
        event.args = {"cmd": "repair"}
    else:
        event.metadata["progress_proof"]["checkpoints"] = {"other": False}
    capture(journal, event)
    last = capture(journal, observation("c"))
    assert session_history(last) == []
    assert build_pipeline(cfg).process(last) is Action.NONE
    assert journal.report(source="test", session_key="s")["incidents"] == []


def test_successful_repetitions_across_turns_never_stitch(tmp_path: Path) -> None:
    # The user asks again every turn and the same read-only check succeeds
    # with the task still unfinished: user-driven polling, not a stuck agent.
    cfg = config(tmp_path)
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint=configuration_fingerprint(cfg))
    for turn in range(12):
        event = observation(f"turn-{turn}")
        event.metadata.pop("error")
        event.args = {"cmd": "gh pr checks 12"}
        event = capture(journal, event)
        assert continuity_key(event) is None
        assert not session_history(event)
        assert build_pipeline(cfg).process(event) is Action.NONE
    assert journal.report(source="test", session_key="s")["incidents"] == []


def test_a_new_error_or_a_success_restarts_the_window(tmp_path: Path) -> None:
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    capture(journal, observation("a"))
    linker = observation("b")
    linker.metadata["error"] = "linker error"
    assert not session_history(capture(journal, linker))
    succeeded = observation("c")
    succeeded.metadata.pop("error")
    capture(journal, succeeded)
    assert not session_history(capture(journal, observation("d")))
    assert len(session_history(capture(journal, observation("e")))) == 2


def test_source_and_session_isolation_restart_and_idempotency(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    first = SessionJournal(path, fingerprint="f")
    a = capture(first, observation("a"))
    resumed = SessionJournal(path, fingerprint="f")
    b = capture(resumed, observation("b"))
    assert len(session_history(b)) == 2
    assert capture(resumed, observation("a")) == a
    assert not session_history(capture(resumed, observation("c"), session="other"))
    assert not session_history(capture(resumed, observation("c"), source="other"))
    with pytest.raises(ValueError, match="different payload"):
        capture(resumed, observation("a", passed=True))
    resumed.close_session(source="test", session_key="s")
    with pytest.raises(ValueError, match="closed"):
        capture(resumed, observation("c"))


def runs(event: ToolEvent) -> list[str | None]:
    return [prior.run_id for prior in session_history(event)]


def test_configuration_change_restarts_the_window_both_ways(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    original = SessionJournal(path, fingerprint="f")
    capture(original, observation("a"))
    assert runs(capture(original, observation("b"))) == ["a", "b"]
    changed = SessionJournal(path, fingerprint="changed")
    assert not session_history(capture(changed, observation("c")))
    assert runs(capture(changed, observation("d"))) == ["c", "d"]
    reverted = SessionJournal(path, fingerprint="f")
    assert not session_history(capture(reverted, observation("e")))
    assert runs(capture(reverted, observation("g"))) == ["e", "g"]


def test_disabled_configuration_ignores_captured_context(tmp_path: Path) -> None:
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    for run in ("a", "b", "c"):
        event = capture(journal, observation(run))
        assert build_pipeline().process(event) is Action.NONE


def test_duplicate_capture_is_atomic_across_writers(tmp_path: Path) -> None:
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    with ThreadPoolExecutor(max_workers=4) as pool:
        events = list(pool.map(lambda _: capture(journal, observation("a")), range(8)))
    assert all(event == events[0] for event in events)
    last = capture(journal, observation("b"))
    assert len(session_history(last)) == 2


def test_langgraph_real_tool_callbacks_continue_across_invocations(tmp_path: Path) -> None:
    from langchain_core.tools import tool

    from pathlayer.adapters.langgraph import LangGraphAdapter
    from pathlayer.state_evidence import FilesystemStateSensor, StateEvidenceProducer

    artifact = tmp_path / "output.txt"
    artifact.write_text("unchanged")

    @tool
    def attempt() -> dict[str, str]:
        """Attempt a build which fails without changing its output."""
        return {"error": "build failed"}

    cfg = config(tmp_path)
    sensor = FilesystemStateSensor(artifact, resource_id="output.txt")
    actions = []
    for run in ("one", "two", "three"):
        pipeline = build_pipeline(cfg)
        adapter = LangGraphAdapter.from_config(
            pipeline.sink,
            cfg,
            session_key="thread-123",
            logical_run_id=run,
            state_evidence=StateEvidenceProducer([sensor]),
        )
        adapter.attach()
        attempt.invoke({}, config=adapter.as_config())
        adapter.detach()
        actions.append(pipeline.actions[-1])
    assert actions[-1] is Action.STOP


def test_openclaw_adapter_restart_and_per_run_replay(tmp_path: Path) -> None:
    from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
    from pathlayer.adapters.openclaw_run_store import RunStore
    from pathlayer.validation.traces import load_trace

    cfg = config(tmp_path)

    class Proof:
        def attach(self, event: ToolEvent) -> dict[str, Any]:
            proof = observation(event.run_id or "none").metadata
            attached = {k: v for k, v in proof.items() if k != "error"}
            event.metadata.update(attached)
            return attached

    for run in ("one", "two", "three"):
        recorder = OpenClawRunRecorder(tmp_path / "runs", config=cfg)
        recorder.startup()
        recorder._proof_producer = Proof()  # type: ignore[assignment]
        ctx = {"runId": run, "sessionKey": "s"}
        recorder.process_payload({"hook": "before_agent_run", "ctx": ctx, "event": {}})
        recorder.process_payload(
            {
                "hook": "after_tool_call",
                "ctx": ctx,
                "event": {
                    "toolName": "shell",
                    "params": {"cmd": "build"},
                    "error": "failed",
                },
            }
        )
        recorder.process_payload({"hook": "agent_end", "ctx": ctx, "event": {"success": True}})
        directory = RunStore(tmp_path / "runs").resolve_run(run)
        assert directory is not None
        trace = load_trace(directory.trace_path)
        replay = build_pipeline(cfg)
        for event in trace.events:
            replay.process(event)
        decisions = [json.loads(line) for line in directory.decisions_path.read_text().splitlines()]
        assert replay.stopped == decisions[-1]["stopped"]
    assert replay.stopped


def test_config_opt_in_and_non_tool_events(tmp_path: Path) -> None:
    legacy = PipelineSection(10, True)
    assert legacy.dry_run and not legacy.session_stitching
    cfg = PathlayerConfig.from_mapping({"pipeline": {"session_stitching": True}})
    assert cfg.pipeline.session_stitching
    with pytest.raises(ValueError):
        PathlayerConfig.from_mapping({"pipeline": {"session_stitching": "yes"}})
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    event = replace(observation("r"), kind=EventKind.LLM_CALL)
    assert capture(journal, event) == event
    assert continuity_key(event) is None


def test_failed_capture_gap_survives_restart_and_keeps_local_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    path = tmp_path / "sessions.db"
    journal = SessionJournal(path, fingerprint="f")
    capture(journal, observation("a"))
    with monkeypatch.context() as patch:

        def fail() -> Any:
            raise sqlite3.OperationalError("database locked")

        patch.setattr(journal, "_connect", fail)
        event = journal.capture(
            observation("b", passed=True), source="test", session_key="s", run_id="b", event_id="e"
        )
        assert not session_history(event)
        assert event.metadata["progress_proof"]["checkpoints"]["test"]
    assert len(list(tmp_path.glob("*.gap"))) == 1  # the database was unreachable
    resumed = SessionJournal(path, fingerprint="f")
    actions = []
    for run in ("c", "d", "e"):
        event = resumed.capture(
            observation(run), source="test", session_key="s", run_id=run, event_id="e"
        )
        assert "a" not in runs(event)  # never bridges the missed observation
        actions.append(build_pipeline(config(tmp_path)).process(event))
    assert runs(event) == ["c", "d", "e"]
    assert actions == [Action.NONE, Action.NONE, Action.STOP]
    assert not list(tmp_path.glob("*.gap"))


def test_close_before_first_capture(tmp_path: Path) -> None:
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    journal.close_session(source="test", session_key="s")
    with pytest.raises(ValueError, match="closed"):
        capture(journal, observation("a"))


def test_serialization_failure_also_breaks_the_window(tmp_path: Path) -> None:
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    capture(journal, observation("a"))
    event = observation("b", passed=True)
    event.result = object()
    returned = journal.capture(event, source="test", session_key="s", run_id="b", event_id="e")
    assert returned.result is event.result
    assert not session_history(returned)
    assert not list(tmp_path.glob("*.gap"))  # the database journaled the gap directly
    resumed = SessionJournal(journal.path, fingerprint="f")
    assert not session_history(capture(resumed, observation("c")))
    assert runs(capture(resumed, observation("d"))) == ["c", "d"]


def test_gap_markers_survive_glob_metacharacters_in_the_store_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    path = tmp_path / "state[1].db.sessions"
    journal = SessionJournal(path, fingerprint="f")
    capture(journal, observation("a"))
    with monkeypatch.context() as patch:

        def fail() -> Any:
            raise sqlite3.OperationalError("database locked")

        patch.setattr(journal, "_connect", fail)
        journal.capture(observation("b"), source="test", session_key="s", run_id="b", event_id="e")
    resumed = SessionJournal(path, fingerprint="f")
    assert not session_history(capture(resumed, observation("c")))
    assert not list(tmp_path.glob("*.gap"))


def test_nan_stitched_reading_never_displaces_the_local_signal(tmp_path: Path) -> None:
    from pathlayer.detectors.base import Detector, Signal
    from pathlayer.detectors.session import SessionAwareDetector

    class Stub(Detector):
        name = "retry"
        event_kinds = frozenset({EventKind.TOOL_CALL})

        def evaluate(self, history: Any) -> Signal:
            # The local history is the single event; its session context is two.
            confidence = 0.5 if len(history) == 1 else float("nan")
            return Signal(self.name, confidence=confidence, reason="stub")

    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    capture(journal, observation("a"))
    event = capture(journal, observation("b"))
    assert len(session_history(event)) == 2
    assert SessionAwareDetector(Stub()).evaluate([event]).confidence == 0.5


def test_capture_failures_warn_once_per_streak(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")

    def attempt(run: str, *, broken: bool) -> None:
        event = observation(run)
        if broken:
            event.result = object()
        journal.capture(event, source="test", session_key="s", run_id=run, event_id="e")

    with caplog.at_level(logging.WARNING, logger="pathlayer.session"):
        for run, broken in (("a", True), ("b", True), ("c", False), ("d", True)):
            attempt(run, broken=broken)
    warnings = [r for r in caplog.records if "continuity window broken" in r.getMessage()]
    assert len(warnings) == 2


def test_capacity_and_malformed_context_abstain(tmp_path: Path) -> None:
    from copy import deepcopy

    from pathlayer.detectors.session import (
        MAX_CONTEXT_BYTES,
        MAX_SESSION_EVENTS,
        SESSION_METADATA_KEY,
    )

    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    for i in range(MAX_SESSION_EVENTS):
        event = capture(journal, observation(str(i)))
    assert len(session_history(event)) == MAX_SESSION_EVENTS
    assert not session_history(capture(journal, observation("overflow")))
    for mutation in ("version", "duplicate", "malformed", "oversize", "current", "nested"):
        bad = deepcopy(event)
        ctx = bad.metadata[SESSION_METADATA_KEY]
        if mutation == "version":
            ctx["version"] = True
        elif mutation == "duplicate":
            ctx["observations"][1]["index"] = ctx["observations"][0]["index"]
        elif mutation == "malformed":
            ctx["observations"][0]["event"] = "not json"
        elif mutation == "oversize":
            ctx["observations"][0]["event"] = "x" * (MAX_CONTEXT_BYTES + 1)
        elif mutation == "current":
            bad.args = {"cmd": "different"}
        else:
            prior = observation("0")
            prior.metadata[SESSION_METADATA_KEY] = {}
            ctx["observations"][0]["event"] = encode_event(prior)
        assert not session_history(bad)


def test_partial_progress_and_canonical_masks(tmp_path: Path) -> None:
    from pathlayer.detectors.canonical import ToolMask

    journal = SessionJournal(
        tmp_path / "sessions.db",
        fingerprint="f",
        tool_masks={"shell": ToolMask(mask=frozenset({"attempt"}))},
    )
    for i in range(3):
        event = observation(str(i))
        event.args["attempt"] = i
        event = capture(journal, event)
    assert len(session_history(event, journal.tool_masks)) == 3
    assert not session_history(event)  # masks are part of the replay configuration
    changed = observation("new")
    changed.args["cmd"] = "repair"
    assert not session_history(capture(journal, changed), journal.tool_masks)

    def partial(run: str, passed: bool) -> ToolEvent:
        event = observation(run)
        spec = event.metadata["proof_spec"]
        spec["required"] = ["test", "artifact"]
        validated = validate_proof_spec(spec)
        assert validated is not None
        proof = event.metadata["progress_proof"]
        proof["spec_hash"] = compute_spec_hash(validated)
        proof["checkpoints"] = {"test": passed, "artifact": False}
        return event

    capture(journal, partial("p1", False))
    assert not session_history(capture(journal, partial("p2", True)))


@pytest.mark.parametrize("change", ["changed", "unavailable", "malformed", "scope"])
def test_state_evidence_changes_break_continuity(tmp_path: Path, change: str) -> None:
    from pathlayer.state_evidence import (
        STATE_EVIDENCE_METADATA_KEY,
        FilesystemStateSensor,
        StateEvidenceProducer,
    )

    artifact = tmp_path / "artifact"
    artifact.write_text("before")
    producer = StateEvidenceProducer([FilesystemStateSensor(artifact, resource_id="output")])
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")

    def measured(run: str) -> ToolEvent:
        event = observation(run)
        event.metadata = {"error": "failed"}
        before = producer.capture_before(tool_event_ref=run)
        return producer.attach_after(event, before)

    capture(journal, measured("a"))
    event = measured("b")
    envelope = event.metadata[STATE_EVIDENCE_METADATA_KEY]
    if change == "changed":
        before = producer.capture_before(tool_event_ref="b")
        artifact.write_text("after")
        event = producer.attach_after(observation("b"), before)
    elif change == "unavailable":
        envelope["outcome"] = "unavailable"
    elif change == "malformed":
        envelope["records"][0]["fingerprint"] = "invalid"
    else:
        for record in envelope["records"]:
            record["resource_id"] = "different"
    assert not session_history(capture(journal, event))
    assert not session_history(capture(journal, measured("c")))


def test_run_budget_and_stop_never_leak(tmp_path: Path) -> None:
    from pathlayer.config import BudgetConfig

    cfg = config(tmp_path)
    cfg.detectors.budget = BudgetConfig(enabled=True, max_cost_usd=1)
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint=configuration_fingerprint(cfg))
    for run in ("a", "b"):
        pipeline = build_pipeline(cfg, state_key=run)
        assert (
            pipeline.process(ToolEvent(tool="model", kind=EventKind.LLM_CALL, cost_usd=0.4))
            is Action.NONE
        )
        assert pipeline.process(capture(journal, observation(run))) is Action.NONE
    stopped = build_pipeline(cfg, state_key="c")
    assert stopped.process(capture(journal, observation("c"))) is Action.STOP
    fresh = build_pipeline(cfg, state_key="d")
    assert fresh.process(capture(journal, observation("d", passed=True))) is Action.NONE
    assert not fresh.stopped


def test_report_rebuild_is_order_independent_and_read_only(tmp_path: Path, capsys: Any) -> None:
    from pathlayer.session import main, project_session
    from pathlayer.validation.traces import Trace, dump_trace

    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    events = [capture(journal, observation(run)) for run in ("z", "a", "m")]
    expected = journal.report(source="test", session_key="s")
    assert project_session(reversed(events), source="test", session_key="s") == expected
    path = tmp_path / "trace.json"
    dump_trace(Trace(label="loop", events=events), path)
    before = path.read_bytes()
    assert main(["--source", "test", "--session-key", "s", "--json", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert main(["--source", "test", "--session-key", "s", str(path)]) == 0
    assert "3 observations across 3 runs" in capsys.readouterr().out
    assert path.read_bytes() == before


def test_openclaw_unfinished_raw_recovery_preserves_proof_and_all_decisions(tmp_path: Path) -> None:
    from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder

    cfg = config(tmp_path)
    recorder = OpenClawRunRecorder(tmp_path / "runs", config=cfg)

    class Proof:
        def attach(self, event: ToolEvent) -> dict[str, Any]:
            metadata = observation("r").metadata
            metadata.pop("error")
            event.metadata.update(metadata)
            return metadata

    recorder._proof_producer = Proof()  # type: ignore[assignment]
    ctx = {"runId": "r", "sessionKey": "s"}
    recorder.process_payload({"hook": "before_agent_run", "ctx": ctx, "event": {}})
    recorder.process_payload(
        {
            "hook": "after_tool_call",
            "ctx": ctx,
            "event": {"toolName": "shell", "params": {"cmd": "build"}, "error": "failed"},
        }
    )
    active = recorder._runs["r"]
    raw_before = active.directory.raw_events_path.read_bytes()
    decisions_before = active.directory.decisions_path.read_bytes()
    recovered = OpenClawRunRecorder(tmp_path / "runs", config=cfg)
    replay = build_pipeline(cfg)
    _, events = recovered._restore_recorded_events(active.directory, replay)
    assert [encode_event(event) for event in events] == [
        encode_event(event) for event in active.events
    ]
    assert replay.actions == active.pipeline.actions
    assert replay.last_decision == active.pipeline.last_decision
    recovered.startup()
    assert active.directory.raw_events_path.read_bytes() == raw_before
    assert active.directory.decisions_path.read_bytes() == decisions_before


def test_openclaw_crash_before_session_commit_breaks_continuity_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder

    cfg = config(tmp_path)
    recorder = OpenClawRunRecorder(tmp_path / "runs", config=cfg)
    assert recorder._session_journal is not None
    recorder._session_journal.enrich(
        observation("a"), source="openclaw", session_key="s", run_id="a", event_id="e"
    )
    ctx = {"runId": "b", "sessionKey": "s"}
    recorder.process_payload({"hook": "before_agent_run", "ctx": ctx, "event": {}})

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated process death")

    monkeypatch.setattr(recorder._session_journal, "capture", crash)
    with pytest.raises(RuntimeError, match="death"):
        recorder.process_payload(
            {"hook": "after_tool_call", "ctx": ctx, "event": {"toolName": "repair", "params": {}}}
        )
    recovered = OpenClawRunRecorder(tmp_path / "runs", config=cfg)
    recovered.startup()
    assert recovered._session_journal is not None
    actions = []
    for run in ("c", "d", "e"):
        event = recovered._session_journal.capture(
            observation(run), source="openclaw", session_key="s", run_id=run, event_id="e"
        )
        assert "a" not in runs(event)  # the interrupted suffix is never bridged
        actions.append(build_pipeline(cfg).process(event))
    assert runs(event) == ["c", "d", "e"]
    assert actions == [Action.NONE, Action.NONE, Action.STOP]


def test_langgraph_failed_proof_abstains_even_with_unchanged_state(tmp_path: Path) -> None:
    from langchain_core.tools import tool

    from pathlayer.adapters.langgraph import LangGraphAdapter
    from pathlayer.state_evidence import FilesystemStateSensor, StateEvidenceProducer

    artifact = tmp_path / "output"
    artifact.write_text("unchanged")
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")

    class FailedProof:
        def attach(self, event: ToolEvent) -> dict[str, Any]:
            raise RuntimeError("failed probe")

    @tool
    def attempt() -> dict[str, str]:
        """Attempt a build."""
        return {"error": "failed"}

    for run in ("a", "b", "c"):
        pipeline = build_pipeline(config(tmp_path))
        adapter = LangGraphAdapter(
            pipeline.sink,
            session_journal=journal,
            session_key="s",
            logical_run_id=run,
            progress_proof=FailedProof(),  # type: ignore[arg-type]
            state_evidence=StateEvidenceProducer(
                [FilesystemStateSensor(artifact, resource_id="o")]
            ),
        )
        adapter.attach()
        attempt.invoke({}, config=adapter.as_config())
        adapter.detach()
        assert pipeline.actions[-1] is Action.NONE
        assert not session_history(pipeline.history[-1])


@pytest.mark.parametrize("case", ["repeat", "progress", "unavailable", "polling"])
def test_labeled_session_traces(tmp_path: Path, case: str) -> None:
    from pathlayer.validation.traces import load_trace

    trace = load_trace(Path(__file__).parents[1] / "traces" / "val7" / f"{case}.json")
    pipeline = build_pipeline(config(tmp_path))
    for event in trace.events:
        action = pipeline.process(event)
    assert action.name == trace.metadata["expected_action"]
    assert (action is Action.NONE) == (trace.label == "healthy")


def test_stale_proof_is_a_gap_within_a_logical_run(tmp_path: Path) -> None:
    journal = SessionJournal(tmp_path / "sessions.db", fingerprint="f")
    capture(journal, observation("a", sequence=5))
    capture(journal, observation("b", sequence=1))
    stale = capture(journal, observation("b", sequence=1), identity="another")
    assert not session_history(stale)
    assert not session_history(capture(journal, observation("c")))
