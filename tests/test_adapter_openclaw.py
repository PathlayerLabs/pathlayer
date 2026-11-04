"""Tests for the OpenClaw capture adapter (ADP-1).

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
from pathlayer.adapters.openclaw import (
    DEFAULT_PORT,
    OpenClawAdapter,
    translate_openclaw_event,
    translate_openclaw_llm_event,
)
from pathlayer.detectors.loop import LoopDetector
from pathlayer.events import EventKind, ToolEvent
from pathlayer.pricing import ModelPrice
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


def test_llm_total_tokens_are_priced_as_an_explicit_estimate() -> None:
    events = translate_openclaw_llm_event(
        {
            "hook": "llm_output",
            "event": {"model": "house", "usage": {"total": 100}},
            "ctx": {"runId": "run-1", "sessionKey": "session-1"},
        },
        prices={"house": ModelPrice(10_000.0, 20_000.0)},
        input_ratio=0.75,
    )
    assert len(events) == 1
    event = events[0]
    assert event.kind is EventKind.LLM_CALL
    assert event.cost_usd == pytest.approx(1.25)
    assert event.metadata == {
        "input_tokens": 75,
        "output_tokens": 25,
        "reported_total_tokens": 100,
        "input_ratio": 0.75,
        "token_split": "input_ratio",
        "cost_measurement": "estimated_total_tokens",
        "session_key": "session-1",
    }


def test_llm_total_missing_or_unknown_price_is_never_a_fabricated_cost() -> None:
    missing = {"hook": "llm_output", "event": {"model": "house", "usage": {}}, "ctx": {}}
    assert translate_openclaw_llm_event(missing) == []

    unknown = translate_openclaw_llm_event(
        {"hook": "llm_output", "event": {"model": "unknown", "usage": {"total": 3}}, "ctx": {}}
    )
    assert unknown[0].cost_usd == 0.0
    assert unknown[0].metadata["cost_measurement"] == "unmeasured_unknown_model"


# The usage OpenClaw 2026.7.1 reported for a live run on Bedrock: four model
# calls, almost all of the tokens served from or written to the prompt cache.
_LIVE_USAGE = {"input": 31, "output": 621, "cacheRead": 79260, "cacheWrite": 20182, "total": 100094}
_BEDROCK_SONNET = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"


def _llm_payload(usage: dict[str, Any], model: str = _BEDROCK_SONNET) -> dict[str, Any]:
    return {
        "hook": "llm_output",
        "event": {"model": model, "usage": usage},
        "ctx": {"runId": "run-1"},
    }


def test_llm_reported_split_prices_each_counter_at_its_rate() -> None:
    [event] = translate_openclaw_llm_event(_llm_payload(_LIVE_USAGE))
    # Sonnet 4.5: $3 input, $15 output, $0.30 cache read, $3.75 cache write.
    expected = (31 * 3.0 + 621 * 15.0 + 79260 * 0.30 + 20182 * 3.75) / 1_000_000
    assert event.cost_usd == pytest.approx(expected)
    assert event.metadata == {
        "input_tokens": 31,
        "output_tokens": 621,
        "cache_read_tokens": 79260,
        "cache_write_tokens": 20182,
        "token_split": "reported",
        "reported_total_tokens": 100094,
        "cost_measurement": "estimated_total_tokens",
    }


def test_llm_split_pricing_can_be_turned_off_for_runs_recorded_before_it() -> None:
    # The total split prices every token at the input or output rate: for this
    # cached run, about five times the split estimate.
    [event] = translate_openclaw_llm_event(_llm_payload(_LIVE_USAGE), split_usage=False)
    assert event.metadata["token_split"] == "input_ratio"
    assert event.cost_usd == pytest.approx((80075 * 3.0 + 20019 * 15.0) / 1_000_000)


def test_llm_counters_omitted_at_zero_count_as_zero() -> None:
    # OpenClaw leaves a counter out when it is zero (here: no cache write).
    [event] = translate_openclaw_llm_event(
        _llm_payload({"input": 9, "output": 95, "cacheRead": 20514})
    )
    assert event.metadata["cache_write_tokens"] == 0
    assert "reported_total_tokens" not in event.metadata
    assert event.cost_usd == pytest.approx((9 * 3.0 + 95 * 15.0 + 20514 * 0.30) / 1_000_000)


@pytest.mark.parametrize("bad", [-5, True, 1.5, "10"])
def test_llm_malformed_split_counter_falls_back_to_the_total(bad: object) -> None:
    [event] = translate_openclaw_llm_event(_llm_payload({"input": 10, "output": bad, "total": 100}))
    assert event.metadata["token_split"] == "input_ratio"
    assert "cache_read_tokens" not in event.metadata
    # Without a usable total either, nothing is invented.
    assert translate_openclaw_llm_event(_llm_payload({"input": 10, "output": bad})) == []


def test_llm_cache_tokens_without_a_cache_rate_stay_unpriced() -> None:
    usage = {"input": 1000, "output": 100, "cacheRead": 4000, "cacheWrite": 500}
    no_cache = {"house": ModelPrice(1.0, 2.0)}
    [event] = translate_openclaw_llm_event(_llm_payload(usage, "house"), prices=no_cache)
    assert event.cost_usd == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1_000_000)
    assert event.metadata["unpriced_cache_tokens"] == 4500
    assert event.metadata["cost_measurement"] == "estimated_total_tokens"

    read_only = {"house": ModelPrice(1.0, 2.0, cache_read_usd_per_mtok=0.1)}
    [event] = translate_openclaw_llm_event(_llm_payload(usage, "house"), prices=read_only)
    assert event.metadata["unpriced_cache_tokens"] == 500


def test_llm_reasoning_tokens_are_evidence_not_billed_twice() -> None:
    usage = {"input": 10, "output": 100, "reasoningTokens": 40}
    [event] = translate_openclaw_llm_event(_llm_payload(usage))
    assert event.metadata["reasoning_tokens"] == 40
    assert event.cost_usd == pytest.approx((10 * 3.0 + 100 * 15.0) / 1_000_000)


def test_llm_split_for_an_unknown_model_is_unmeasured() -> None:
    [event] = translate_openclaw_llm_event(_llm_payload(_LIVE_USAGE, "house"))
    assert event.cost_usd == 0.0
    assert event.metadata["cost_measurement"] == "unmeasured_unknown_model"
    assert "unpriced_cache_tokens" not in event.metadata


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


@pytest.mark.parametrize(
    "ratio", [-0.1, 1.1, float("nan"), float("inf"), -float("inf"), True, False, "0.8", None]
)
def test_invalid_input_ratio_rejected_before_any_events(ratio: Any) -> None:
    with pytest.raises(ValueError, match="input_ratio"):
        OpenClawAdapter(Pipeline([]), input_ratio=ratio)
    with pytest.raises(ValueError, match="input_ratio"):
        translate_openclaw_llm_event({"hook": "llm_output", "event": {}}, input_ratio=ratio)


@pytest.mark.parametrize("ratio, expected_input", [(0, 0), (1, 5), (0.8, 4)])
def test_input_ratio_boundaries_preserve_total(ratio: float, expected_input: int) -> None:
    event = translate_openclaw_llm_event(
        {"hook": "llm_output", "event": {"model": "house", "usage": {"total": 5}}},
        input_ratio=ratio,
    )[0]
    assert event.metadata["input_tokens"] == expected_input
    assert event.metadata["input_tokens"] + event.metadata["output_tokens"] == 5
