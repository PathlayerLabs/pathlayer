"""Tests for the OpenClaw capture adapter (ADP-1, issue #40).

Two layers, matching how the module is split:

- ``translate_openclaw_event`` is a pure function tested directly against
  payload shapes following the public plugin hook reference
  (https://docs.openclaw.ai/plugins/hooks) -- no network, no OpenClaw needed.
- ``OpenClawAdapter`` is exercised end-to-end over real loopback HTTP with a
  "recorded OpenClaw session" fixture (a plausible sequence of hook firings),
  which doubles as the ADP-1 smoke test required by the issue's Definition of
  Done ("running an OpenClaw agent produces a valid trace, covered by a smoke
  test on a recorded OpenClaw session"): since a live OpenClaw cannot run in
  CI, a recorded session fixture is the offline equivalent.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from pathlayer import Pipeline
from pathlayer.adapters.openclaw import DEFAULT_PORT, OpenClawAdapter, translate_openclaw_event
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.validation.traces import Trace, dump_trace, load_trace

# --------------------------------------------------------------------------- #
# translate_openclaw_event: pure translation, no OpenClaw needed.
# --------------------------------------------------------------------------- #


def test_after_tool_call_is_translated() -> None:
    events = translate_openclaw_event(
        {
            "hook": "after_tool_call",
            "event": {
                "toolName": "read_file",
                "params": {"path": "src/main.py"},
                "result": "def main(): ...",
                "runId": "run-1",
                "toolCallId": "call-1",
                "durationMs": 42,
            },
            "ctx": {"agentId": "agent-1", "sessionKey": "sess-1"},
        }
    )
    assert len(events) == 1
    event = events[0]
    assert event.tool == "read_file"
    assert event.args == {"path": "src/main.py"}
    assert event.result == "def main(): ..."
    assert event.kind is EventKind.TOOL_CALL
    assert event.run_id == "run-1"
    assert event.metadata == {
        "agent_id": "agent-1",
        "session_key": "sess-1",
        "tool_call_id": "call-1",
        "duration_ms": 42,
    }


def test_after_tool_call_error_is_captured_in_metadata() -> None:
    events = translate_openclaw_event(
        {
            "hook": "after_tool_call",
            "event": {"toolName": "run_tests", "params": {}, "error": "1 failed"},
            "ctx": {},
        }
    )
    assert events[0].metadata["error"] == "1 failed"
    assert events[0].result is None


def test_run_id_prefers_ctx_over_event_and_never_tool_call_id() -> None:
    events = translate_openclaw_event(
        {
            "hook": "after_tool_call",
            "event": {
                "toolName": "x",
                "params": {},
                "runId": "event-run",
                "toolCallId": "call-9",
            },
            "ctx": {"runId": "ctx-run"},
        }
    )
    assert events[0].run_id == "ctx-run"
    assert events[0].metadata["tool_call_id"] == "call-9"

    events = translate_openclaw_event(
        {
            "hook": "after_tool_call",
            "event": {"toolName": "x", "params": {}, "runId": "event-run"},
            "ctx": {},
        }
    )
    assert events[0].run_id == "event-run"

    events = translate_openclaw_event(
        {
            "hook": "after_tool_call",
            "event": {"toolName": "x", "params": {}, "toolCallId": "call-9"},
            "ctx": {},
        }
    )
    assert events[0].run_id is None
    assert events[0].metadata["tool_call_id"] == "call-9"


@pytest.mark.parametrize(
    "hook",
    [
        "session_start",
        "session_end",
        "before_tool_call",
        "subagent_spawned",
        "gateway_start",
        "before_agent_run",
        "agent_end",
    ],
)
def test_unhandled_hooks_are_ignored_not_errors(hook: str) -> None:
    # Only after_tool_call becomes ToolEvents; lifecycle hooks are handled by
    # the run recorder and must not raise during translation.
    assert translate_openclaw_event({"hook": hook, "event": {}, "ctx": {}}) == []


def test_non_dict_payload_raises() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        translate_openclaw_event("not a dict")


def test_after_tool_call_missing_tool_name_raises() -> None:
    with pytest.raises(ValueError, match="toolName"):
        translate_openclaw_event({"hook": "after_tool_call", "event": {"params": {}}, "ctx": {}})


def test_missing_params_defaults_to_empty_args() -> None:
    events = translate_openclaw_event(
        {"hook": "after_tool_call", "event": {"toolName": "noop"}, "ctx": {}}
    )
    assert events[0].args == {}


def test_falsy_tool_name_raises() -> None:
    # A present-but-empty/null toolName is just as invalid as a missing key.
    for bad_name in (None, ""):
        with pytest.raises(ValueError, match="toolName"):
            translate_openclaw_event(
                {"hook": "after_tool_call", "event": {"toolName": bad_name}, "ctx": {}}
            )


def test_non_dict_truthy_event_raises_missing_tool_name() -> None:
    with pytest.raises(ValueError, match="toolName"):
        translate_openclaw_event(
            {"hook": "after_tool_call", "event": ["not", "a", "dict"], "ctx": {}}
        )


def test_non_dict_ctx_falls_back_to_empty() -> None:
    events = translate_openclaw_event(
        {"hook": "after_tool_call", "event": {"toolName": "x"}, "ctx": "not-a-dict"}
    )
    assert events[0].run_id is None
    assert events[0].metadata == {}


def test_non_dict_params_falls_back_to_empty_args() -> None:
    events = translate_openclaw_event(
        {
            "hook": "after_tool_call",
            "event": {"toolName": "x", "params": ["not", "a", "dict"]},
            "ctx": {},
        }
    )
    assert events[0].args == {}


# --------------------------------------------------------------------------- #
# OpenClawAdapter: real loopback HTTP, wired to the translator above.
# --------------------------------------------------------------------------- #


def test_default_port_is_documented_constant() -> None:
    assert OpenClawAdapter(lambda _e: None, port=0)._http._requested_port == 0
    assert DEFAULT_PORT == 8787


def _post(port: int, body: dict[str, Any]) -> int:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/ingest",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return int(response.status)


def test_adapter_end_to_end_over_http() -> None:
    events: list[ToolEvent] = []
    adapter = OpenClawAdapter(events.append, port=0)
    adapter.attach()
    try:
        status = _post(
            adapter.port,
            {
                "source": "openclaw",
                "payload": {
                    "hook": "after_tool_call",
                    "event": {"toolName": "read_file", "params": {"path": "a.py"}},
                    "ctx": {"sessionKey": "sess-1"},
                },
            },
        )
        assert status == 200
    finally:
        adapter.detach()

    assert len(events) == 1
    assert events[0].tool == "read_file"


# A plausible sequence of hook firings for one OpenClaw agent run.
_RECORDED_OPENCLAW_SESSION: list[dict[str, Any]] = [
    {
        "hook": "before_agent_run",
        "event": {"prompt": "read README"},
        "ctx": {"sessionKey": "sess-1", "agentId": "agent-1", "runId": "run-1"},
    },
    {
        "hook": "after_tool_call",
        "event": {
            "toolName": "read_file",
            "params": {"path": "src/main.py"},
            "result": "def main(): ...",
            "toolCallId": "call-1",
            "durationMs": 12,
        },
        "ctx": {"sessionKey": "sess-1", "agentId": "agent-1", "runId": "run-1"},
    },
    {
        "hook": "after_tool_call",
        "event": {
            "toolName": "edit_file",
            "params": {"path": "src/main.py", "patch": "add guard"},
            "result": {"ok": True},
            "toolCallId": "call-2",
            "durationMs": 8,
        },
        "ctx": {"sessionKey": "sess-1", "agentId": "agent-1", "runId": "run-1"},
    },
    {
        "hook": "after_tool_call",
        "event": {
            "toolName": "run_tests",
            "params": {"target": "tests/test_main.py"},
            "error": "1 failed, 3 passed",
            "toolCallId": "call-3",
            "durationMs": 950,
        },
        "ctx": {"sessionKey": "sess-1", "agentId": "agent-1", "runId": "run-1"},
    },
    {
        "hook": "agent_end",
        "event": {"success": True, "durationMs": 5000},
        "ctx": {"sessionKey": "sess-1", "agentId": "agent-1", "runId": "run-1"},
    },
]


def test_recorded_openclaw_session_smoke_test(tmp_path: Path) -> None:
    events: list[ToolEvent] = []
    adapter = OpenClawAdapter(events.append, port=0)
    adapter.attach()
    try:
        for entry in _RECORDED_OPENCLAW_SESSION:
            status = _post(adapter.port, {"source": "openclaw", "payload": entry})
            assert status == 200
    finally:
        adapter.detach()

    # Lifecycle hooks are handled by the run recorder; only tool calls become
    # ToolEvents through translate_openclaw_event / OpenClawAdapter.
    assert [e.tool for e in events] == ["read_file", "edit_file", "run_tests"]
    assert [e.ts for e in events] == [0, 1, 2]
    assert events[2].metadata["error"] == "1 failed, 3 passed"

    # DoD: "produces a valid trace (VAL-1 schema)".
    trace = Trace(
        label="healthy",
        description="Recorded OpenClaw session (ADP-1 smoke test)",
        events=events,
    )
    path = tmp_path / "openclaw_session.json"
    dump_trace(trace, path)
    reloaded = load_trace(path)
    assert reloaded.events == events


def test_recorded_session_feeds_the_real_pipeline() -> None:
    # Same fixture, but through a real Pipeline + LoopDetector -- proves the
    # adapter's output is consumable end-to-end, not just structurally valid.
    pipeline = Pipeline([LoopDetector()])
    adapter = OpenClawAdapter(pipeline.sink, port=0)
    adapter.attach()
    try:
        for entry in _RECORDED_OPENCLAW_SESSION:
            _post(adapter.port, {"source": "openclaw", "payload": entry})
    finally:
        adapter.detach()

    assert len(pipeline.history) == 3
    assert [e.tool for e in pipeline.history] == ["read_file", "edit_file", "run_tests"]
