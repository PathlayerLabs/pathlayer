"""Exercise the installed public API from outside the repository checkout."""

from __future__ import annotations

import argparse
import importlib.util
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
from pathlayer import ConfigError, build_pipeline, load_config  # noqa: E402
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
        "SQLiteStateStore",
        "StateStoreError",
        "WouldBeAction",
        "__version__",
        "build_pipeline",
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


def _exercise_yaml() -> None:
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "config.yaml"
        config.write_text("pipeline:\n  dry_run: true\n", encoding="utf-8")
        assert load_config(config).pipeline.dry_run is True


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
