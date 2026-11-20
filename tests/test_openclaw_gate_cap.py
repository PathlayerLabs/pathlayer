"""The gate's cap on identical tool calls in flight (OpenClaw).

OpenClaw runs the tool calls of one assistant message in parallel and decides
every ``before_tool_call`` query of the batch before any call runs (2026.7.1:
all hooks start together; 2026.10.10 and later: the batch is prepared, then
launched). A STOP that needs observed failures therefore cannot stop such a
batch: in the 2026-11-18 retest, 20 identical calls ran although Pathlayer
stopped the run at the third failure. The cap blocks a call when the run already
has ``max_identical_in_flight`` identical calls (same tool, same arguments, as
fingerprinted by the plugin) running.

The labeled scenarios in ``tests/fixtures/openclaw_gate/`` are the evidence for
the default threshold: the cap must never fire on a ``healthy`` one, and must
fire on every ``loop`` one. Only the cap's own blocks are counted here: a STOP
selected by a detector also blocks, and is tested elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pathlayer.adapters import openclaw_cli
from pathlayer.adapters.openclaw_recorder import (
    DEFAULT_MAX_IDENTICAL_IN_FLIGHT,
    OpenClawRunRecorder,
)
from pathlayer.adapters.openclaw_report import build_run_report
from pathlayer.adapters.openclaw_run_store import RunDirectory, RunStore

_SCENARIOS = sorted((Path(__file__).parent / "fixtures" / "openclaw_gate").glob("*.json"))
_RUN = "run-a"
_CTX = {"runId": _RUN, "sessionKey": "sess-1", "agentId": "agent-1"}


def _digest(tool: str, params: dict[str, Any]) -> str:
    identity = json.dumps({"tool": tool, "params": params}, sort_keys=True)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _play(recorder: OpenClawRunRecorder, steps: list[Any]) -> list[dict[str, Any] | None]:
    """Feed one scenario; return the gate's answer to each query, in order."""
    answers: list[dict[str, Any] | None] = []
    for step in steps:
        if step == "start":
            recorder.process_payload({"hook": "before_agent_run", "event": {}, "ctx": _CTX})
        elif "end" in step:
            recorder.process_payload(
                {"hook": "agent_end", "event": {"success": step["end"]}, "ctx": _CTX}
            )
        elif "query" in step:
            event = {
                "toolName": step["tool"],
                "toolCallId": step["query"],
                "runId": _RUN,
                "paramsDigest": _digest(step["tool"], step["params"]),
            }
            answers.append(
                recorder.tool_call_decision(
                    {"hook": "before_tool_call", "event": event, "ctx": _CTX}
                )
            )
        else:
            event = {
                "toolName": step["tool"],
                "toolCallId": step["result"],
                "params": step["params"],
                "result": "error" if "error" in step else "ok",
                "runId": _RUN,
            }
            if "error" in step:
                event["error"] = step["error"]
            recorder.process_payload({"hook": "after_tool_call", "event": event, "ctx": _CTX})
    return answers


def _cap_blocks(answers: list[dict[str, Any] | None]) -> int:
    return sum(
        1
        for answer in answers
        if answer is not None and "identical calls in flight" in answer["blockReason"]
    )


def _run(root: Path) -> RunDirectory:
    run = RunStore(root).resolve_run(_RUN)
    assert run is not None
    return run


def _gate(run: RunDirectory) -> list[dict[str, Any]]:
    return [json.loads(line) for line in run.gate_path.read_text(encoding="utf-8").splitlines()]


def test_the_scenarios_cover_both_labels() -> None:
    labels = [json.loads(path.read_text(encoding="utf-8"))["label"] for path in _SCENARIOS]
    assert labels.count("healthy") >= 5
    assert labels.count("loop") >= 2
    assert set(labels) == {"healthy", "loop"}


@pytest.mark.parametrize("path", _SCENARIOS, ids=lambda path: path.stem)
def test_the_cap_fires_on_loops_only(tmp_path: Path, path: Path) -> None:
    scenario = json.loads(path.read_text(encoding="utf-8"))
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()

    answers = _play(recorder, scenario["steps"])

    if scenario["label"] == "healthy":
        assert _cap_blocks(answers) == 0
    else:
        assert _cap_blocks(answers) > 0


def test_a_burst_runs_only_the_cap(tmp_path: Path) -> None:
    scenario = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "openclaw_gate"
            / "loop_burst_of_20_identical_failing_calls.json"
        ).read_text(encoding="utf-8")
    )
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()

    answers = _play(recorder, scenario["steps"])

    assert answers[:DEFAULT_MAX_IDENTICAL_IN_FLIGHT] == [None] * DEFAULT_MAX_IDENTICAL_IN_FLIGHT
    assert _cap_blocks(answers) == 20 - DEFAULT_MAX_IDENTICAL_IN_FLIGHT
    blocked = answers[DEFAULT_MAX_IDENTICAL_IN_FLIGHT]
    assert blocked is not None
    assert blocked["blockReason"].startswith(
        "Pathlayer blocked this call (identical calls in flight: 3 exec call(s)"
    )
    gate = _gate(_run(tmp_path))
    assert [line["cap_exceeded"] for line in gate] == [False] * 3 + [True] * 17
    assert all(line["params_digest"] == gate[0]["params_digest"] for line in gate)
    report = build_run_report(_run(tmp_path))
    assert report.tool_call_gate is not None
    assert len(report.tool_call_gate.blocked) == 17


def test_a_finished_call_frees_its_slot(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    query = {"query": "c0", "tool": "exec", "params": {"command": "make"}}
    result = {"result": "c0", "tool": "exec", "params": {"command": "make"}}
    steps: list[Any] = ["start"]
    for index in range(3):
        steps.append({**query, "query": f"c{index}"})
    steps.append({**result, "result": "c0"})
    steps.append({**query, "query": "c3"})
    steps.append({**query, "query": "c4"})

    answers = _play(recorder, steps)

    # c3 takes the slot c0 freed; c4 is the fourth identical call in flight.
    assert answers[:4] == [None] * 4
    assert _cap_blocks(answers[4:]) == 1


def test_a_passive_recording_logs_the_cap_without_blocking(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path)
    recorder.startup()
    steps: list[Any] = [
        "start",
        *[{"query": f"c{i}", "tool": "exec", "params": {"command": "ls"}} for i in range(5)],
    ]

    answers = _play(recorder, steps)

    assert answers == [None] * 5
    gate = _gate(_run(tmp_path))
    assert [line["cap_exceeded"] for line in gate] == [False] * 3 + [True] * 2
    assert all(line["decision"] == "allow" for line in gate)
    report = build_run_report(_run(tmp_path))
    assert report.tool_call_gate is not None
    assert report.tool_call_gate.would_have_blocked == 2


@pytest.mark.parametrize("digest", [None, "", 42])
def test_a_query_without_a_fingerprint_is_never_capped(tmp_path: Path, digest: Any) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True)
    recorder.startup()
    recorder.process_payload({"hook": "before_agent_run", "event": {}, "ctx": _CTX})
    for index in range(6):
        event = {"toolName": "exec", "toolCallId": f"c{index}", "paramsDigest": digest}
        assert (
            recorder.tool_call_decision({"hook": "before_tool_call", "event": event, "ctx": _CTX})
            is None
        )


def test_a_zero_cap_disables_it(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, enforce=True, max_identical_in_flight=0)
    recorder.startup()
    steps: list[Any] = [
        "start",
        *[{"query": f"c{i}", "tool": "exec", "params": {"command": "ls"}} for i in range(10)],
    ]
    assert _play(recorder, steps) == [None] * 10


@pytest.mark.parametrize("value", [-1, True])
def test_an_invalid_cap_is_refused(tmp_path: Path, value: Any) -> None:
    with pytest.raises(ValueError, match="max_identical_in_flight"):
        OpenClawRunRecorder(tmp_path, max_identical_in_flight=value)


def test_the_cli_sets_the_cap(tmp_path: Path) -> None:
    parser = openclaw_cli.build_parser()
    args = parser.parse_args(
        ["record", "--runs-root", str(tmp_path), "--enforce", "--max-identical-in-flight", "0"]
    )
    assert args.max_identical_in_flight == 0
    default = parser.parse_args(["record", "--runs-root", str(tmp_path)])
    assert default.max_identical_in_flight == DEFAULT_MAX_IDENTICAL_IN_FLIGHT
    with pytest.raises(SystemExit):
        parser.parse_args(["record", "--max-identical-in-flight", "-2"])
