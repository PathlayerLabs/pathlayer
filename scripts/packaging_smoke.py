"""Exercise the installed public API from outside the repository checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from readme_quickstart import extract_quickstart

_NETWORK_EVENTS = {
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyaddr",
    "socket.gethostbyname",
    "socket.getnameinfo",
    "socket.sendmsg",
    "socket.sendto",
}


def _deny_network(event: str, args: tuple[object, ...]) -> None:
    del args
    if event in _NETWORK_EVENTS:
        raise RuntimeError(
            f"network access is forbidden during the Pathlayer runtime smoke test: {event}"
        )


# Audit hooks block outbound socket operations without replacing socket.socket,
# which must remain a class so importing ssl and http.server keeps working.
sys.addaudithook(_deny_network)

import pathlayer  # noqa: E402
from pathlayer import (  # noqa: E402
    ConfigError,
    SessionOutcome,
    TickRecord,
    build_pipeline,
    build_session_report,
    load_config,
)
from pathlayer.adapters import InMemoryAdapter  # noqa: E402
from pathlayer.events import ToolEvent  # noqa: E402
from pathlayer.policies import Action, Actuator, wire  # noqa: E402
from pathlayer.postmortem import PostMortem  # noqa: E402


class RecordingActuator(Actuator):
    """Record the real action transitions produced by ``wire``."""

    def __init__(self) -> None:
        self.applied: list[tuple[Action, PostMortem | None]] = []

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        self.applied.append((action, post_mortem))


def _assert_network_blocked() -> None:
    import socket

    try:
        socket.getaddrinfo("localhost", 80)
    except RuntimeError as exc:
        assert "network access is forbidden" in str(exc)
    else:
        raise AssertionError("network audit hook is inactive")


def _healthy_events() -> list[ToolEvent]:
    plan: list[tuple[str, dict[str, object]]] = [
        ("list_dir", {"path": "src"}),
        ("read_file", {"path": "src/main.py"}),
        ("edit_file", {"path": "src/main.py", "patch": "add guard"}),
        ("run_tests", {"target": "tests/test_main.py"}),
        ("git_commit", {"message": "add guard"}),
    ]
    return [
        ToolEvent(tool=tool, args=args, ts=float(index)) for index, (tool, args) in enumerate(plan)
    ]


def _loop_events() -> list[ToolEvent]:
    return [
        ToolEvent(
            tool=("list_dir", "read_file")[index % 2],
            args={"path": "src/main.py"},
            ts=float(index),
        )
        for index in range(40)
    ]


def _exercise_core() -> None:
    expected_api = {
        "ConfigError",
        "PathlayerConfig",
        "PersistentPipeline",
        "Pipeline",
        "PipelineDecision",
        "PostMortem",
        "RunAwareEventSink",
        "SQLiteStateStore",
        "SessionOutcome",
        "SessionReport",
        "StateStoreError",
        "TickRecord",
        "WouldBeAction",
        "__version__",
        "build_pipeline",
        "build_session_report",
        "load_config",
    }
    assert set(pathlayer.__all__) == expected_api
    for name in expected_api:
        getattr(pathlayer, name)
    assert pathlayer.__version__ == version("pathlayer")
    assert Path.cwd().joinpath(".pathlayer").exists() is False

    healthy_pipeline = build_pipeline(load_config())
    healthy_actuator = RecordingActuator()
    InMemoryAdapter(wire(healthy_pipeline, healthy_actuator), _healthy_events()).attach()
    assert healthy_pipeline.stopped is False
    assert healthy_actuator.applied == []

    loop_pipeline = build_pipeline(load_config())
    loop_actuator = RecordingActuator()
    InMemoryAdapter(wire(loop_pipeline, loop_actuator), _loop_events()).attach()
    assert loop_pipeline.stopped is True
    assert loop_pipeline.post_mortem is not None
    assert loop_actuator.applied
    action, post_mortem = loop_actuator.applied[-1]
    assert action is Action.STOP
    assert post_mortem is loop_pipeline.post_mortem
    assert Path.cwd().joinpath(".pathlayer").exists() is False


_JOURNAL: tuple[dict[str, object], ...] = (
    {"index": 0, "ts": 0.0, "signals": [{"name": "loop", "confidence": 0.0}], "fused": 0.0},
    {"index": 1, "ts": 1.0, "signals": [{"name": "loop", "confidence": 0.45}], "fused": 0.45},
    {
        "index": 2,
        "ts": 2.0,
        "signals": [{"name": "loop", "confidence": 0.95, "reason": "same call 20x"}],
        "fused": 0.95,
        "action": "STOP",
        "reason": "loop: same call 20x",
    },
)


def _exercise_session_report() -> None:
    """Rebuild a session report from a journal, as a user would after a run.

    The whole point of OBS-2 is that a report is reconstructible from a durable
    journal alone, so the journal is written to disk and read back rather than
    passed in memory. It goes into a temporary directory: nothing here may
    create ``.pathlayer`` in the caller's working directory.
    """
    assert Path.cwd().joinpath(".pathlayer").exists() is False
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run-smoke"
        run_dir.mkdir()
        journal = run_dir / "decisions.jsonl"
        journal.write_text(
            "".join(json.dumps(TickRecord.from_dict(row).to_dict()) + "\n" for row in _JOURNAL),
            encoding="utf-8",
        )
        ticks = [
            TickRecord.from_dict(json.loads(line))
            for line in journal.read_text(encoding="utf-8").splitlines()
        ]

        report = build_session_report(
            run_id="run-smoke",
            status="completed",
            ticks=ticks,
            events_recorded=4,
            started_at="2026-01-01T00:00:00+00:00",
        )
        assert isinstance(report, pathlayer.SessionReport)
        # A STOP was emitted, so the run is stopped even though the manifest
        # status says the process reached its documented end.
        assert report.outcome is SessionOutcome.STOPPED
        assert report.stop_reason == "loop: same call 20x"
        assert (report.events_recorded, report.events_analyzed) == (4, 3)

        # The chronology: which detector peaked where, and what was emitted when.
        activity = report.detectors[0]
        assert (activity.name, activity.peak_confidence, activity.peak_index) == ("loop", 0.95, 2)
        assert [(a.action, a.index, a.emitted) for a in report.actions] == [(Action.STOP, 2, True)]
        assert (report.peak_fused, report.peak_fused_index) == (0.95, 2)
        rendered = report.format()
        assert "outcome: stopped" in rendered
        assert "event 2" in rendered

        # Nothing priced this run: the report must say so instead of billing $0.
        assert report.cost.measured is False
        assert report.cost.total_usd is None
        assert "not measured" in report.cost.describe()
        assert "$0.0000" not in rendered

        # The report survives the JSON round trip a user persists it through.
        serialized = run_dir / "report.json"
        serialized.write_text(json.dumps(report.to_dict()), encoding="utf-8")
        restored = json.loads(serialized.read_text(encoding="utf-8"))
        assert restored["outcome"] == SessionOutcome.STOPPED.value
        assert restored["cost"]["total_usd"] is None
    assert Path.cwd().joinpath(".pathlayer").exists() is False


def _exercise_yaml() -> None:
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "config.yaml"
        config.write_text("pipeline:\n  dry_run: true\n", encoding="utf-8")
        assert load_config(config).pipeline.dry_run is True


def _exercise_scaffolder_entrypoint() -> None:
    executable_name = (
        "pathlayer-scaffold-proof.exe" if os.name == "nt" else "pathlayer-scaffold-proof"
    )
    executable = Path(sys.executable).with_name(executable_name)
    assert executable.is_file(), f"missing installed entry point: {executable}"
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "proof"
        completed = subprocess.run(
            [
                str(executable),
                "--task-id",
                "package-smoke",
                "--name",
                "Package smoke proof",
                "--objective",
                "Verify the installed scaffolder.",
                "--checkpoint",
                "installed",
                "--output-dir",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "created" in completed.stdout
        assert json.loads((output / "proof_spec.json").read_text(encoding="utf-8"))["required"] == [
            "installed"
        ]
        probe = subprocess.run(
            [sys.executable, str(output / "probe.py")], capture_output=True, text=True
        )
        assert probe.returncode != 0
        assert "NotImplementedError" in probe.stderr


def _exercise_missing_yaml_extra() -> None:
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "config.yaml"
        config.write_text("pipeline:\n  dry_run: true\n", encoding="utf-8")
        try:
            load_config(config)
        except ConfigError as exc:
            assert "pathlayer[yaml]" in str(exc)
        else:
            raise AssertionError("YAML config unexpectedly loaded without pathlayer[yaml]")


def _exercise_langgraph() -> None:
    from pathlayer.adapters.langgraph import LangGraphAdapter
    from pathlayer.policies.langgraph_actuator import LangGraphActuator

    pipeline = build_pipeline(load_config())
    actuator = LangGraphActuator()
    actuator.attach()
    adapter = LangGraphAdapter(wire(pipeline, actuator))
    adapter.attach()
    assert adapter.as_config()
    assert actuator.control is not None
    adapter.detach()


def _exercise_readme(readme: Path) -> None:
    """Execute the exact marked quickstart against the installed distribution."""
    assert Path.cwd().joinpath(".pathlayer").exists() is False
    code = extract_quickstart(readme)
    namespace = {"__name__": "__pathlayer_readme_quickstart__"}
    exec(compile(code, str(readme), "exec"), namespace)
    assert Path.cwd().joinpath(".pathlayer").exists() is False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extras",
        choices=("core", "yaml", "langgraph", "all"),
        default="core",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        help="execute the exact marked core quickstart from this README",
    )
    args = parser.parse_args()

    _assert_network_blocked()
    _exercise_core()
    _exercise_session_report()
    _exercise_scaffolder_entrypoint()
    if args.readme is not None:
        _exercise_readme(args.readme)
    if args.extras == "core":
        assert importlib.util.find_spec("yaml") is None
        assert importlib.util.find_spec("langgraph") is None
        _exercise_missing_yaml_extra()
    if args.extras in {"yaml", "all"}:
        _exercise_yaml()
        if args.extras == "yaml":
            assert importlib.util.find_spec("langgraph") is None
    if args.extras in {"langgraph", "all"}:
        _exercise_langgraph()
    print(f"packaging-smoke-ok version={pathlayer.__version__} extras={args.extras}")


if __name__ == "__main__":
    main()
