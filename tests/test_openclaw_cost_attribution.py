"""OBS-5: captured cost split into attributed and unattributed, by cause.

Two families of tests, kept apart on purpose:

- **Association losses Pathlayer caused** -- a cost hook that lost the race with
  ``agent_end`` was dropped, a run without ``trace.json`` reported no cost, a
  blank ``ctx.runId`` shadowed a valid ``event.runId``. Each is a regression test.
- **Cost that is legitimately outside any turn** -- the source named no run. It
  must stay unattributed, with its reason and the source's own trigger, and must
  never be attached to a run by any other correlation (session, timing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_report import (
    build_cost_attribution,
    build_run_report,
    build_store_cost_attribution,
    main,
)
from pathlayer.adapters.openclaw_run_store import RunDirectory, RunStore
from pathlayer.config import BudgetConfig, DetectorsConfig, PathlayerConfig, PricingSection
from pathlayer.pipeline import Pipeline
from pathlayer.postmortem import (
    TRIGGER_ABSENT,
    UNATTRIBUTED_MISSING_RUN_ID,
    CostAttribution,
    CostShare,
    UnattributedCost,
)
from pathlayer.pricing import ModelPrice
from pathlayer.validation.traces import load_trace

# 1000 total tokens at the default 0.8 input ratio: 800 * $1/M + 200 * $2/M.
_CALL_USD = 0.0012
_PRICED = PathlayerConfig(
    pricing=PricingSection(models={"model": ModelPrice(1.0, 2.0), "free": ModelPrice(0.0, 0.0)})
)
# The recorder's own snapshot shape for one model, to corrupt one rate at a time.
_RATES = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0}
# Every file a finished run is made of, except the two OBS-5 may touch.
_IMMUTABLE = (
    "manifest.json",
    "request.json",
    "events.raw.jsonl",
    "trace.json",
    "decisions.jsonl",
    "final.json",
)


def _ctx(run_id: str | None, *, session: str = "sess-1", trigger: str | None = None) -> dict:
    ctx: dict[str, Any] = {"sessionKey": session, "agentId": "agent-1"}
    if run_id is not None:
        ctx["runId"] = run_id
    if trigger is not None:
        ctx["trigger"] = trigger
    return ctx


def _before(run_id: str) -> dict[str, Any]:
    return {"hook": "before_agent_run", "event": {"prompt": "fix it"}, "ctx": _ctx(run_id)}


def _end(run_id: str) -> dict[str, Any]:
    return {"hook": "agent_end", "event": {"success": True, "durationMs": 10}, "ctx": _ctx(run_id)}


def _tool(run_id: str) -> dict[str, Any]:
    return {
        "hook": "after_tool_call",
        "event": {"toolName": "read", "params": {}, "toolCallId": "c1", "runId": run_id},
        "ctx": _ctx(run_id),
    }


def _llm(
    run_id: str | None,
    *,
    event_run_id: str | None = None,
    model: str = "model",
    total: int | None = 1000,
    usage: dict[str, int] | None = None,
    session: str = "sess-1",
    trigger: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"model": model}
    if usage is not None:
        event["usage"] = usage
    elif total is not None:
        event["usage"] = {"total": total}
    if event_run_id is not None:
        event["runId"] = event_run_id
    return {
        "hook": "llm_output",
        "event": event,
        "ctx": _ctx(run_id, session=session, trigger=trigger),
    }


def _run(root: Path, run_id: str) -> RunDirectory:
    run = RunStore(root).resolve_run(run_id)
    assert run is not None
    return run


def _orphans(root: Path) -> list[RunDirectory]:
    return [run for run in RunStore(root).list_runs() if run.original_run_id.startswith("orphan-")]


def _snapshot(run: RunDirectory) -> dict[str, bytes]:
    return {
        name: (run.root / name).read_bytes() for name in _IMMUTABLE if (run.root / name).is_file()
    }


def _late_lines(run: RunDirectory) -> list[dict[str, Any]]:
    if not run.late_events_path.is_file():
        return []
    return [json.loads(line) for line in run.late_events_path.read_text().splitlines()]


def _assert_partition(attribution: CostAttribution) -> None:
    captured = attribution.captured
    parts = attribution.attributed.plus(attribution.unattributed_total)
    assert captured == parts
    if captured.total_usd is not None:
        assert captured.total_usd == pytest.approx(
            (attribution.attributed.total_usd or 0.0)
            + (attribution.unattributed_total.total_usd or 0.0)
        )


# --------------------------------------------------------------------------- #
# Association losses Pathlayer caused (regressions)
# --------------------------------------------------------------------------- #


def test_llm_output_after_agent_end_is_kept_and_attributed_late(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    before = _snapshot(run)

    recorder.process_payload(_llm("r1"))

    assert _snapshot(run) == before
    late = _late_lines(run)
    assert len(late) == 1
    assert late[0]["after_status"] == "completed"
    assert late[0]["payload"] == _llm("r1")
    attribution = build_cost_attribution(run)
    assert attribution.attributed.costed_events == 2
    assert attribution.attributed.total_usd == pytest.approx(2 * _CALL_USD)
    assert attribution.late.total_usd == pytest.approx(_CALL_USD)
    assert attribution.unattributed == ()
    # The report on disk is rebuilt from the annex, and says what was not analyzed.
    report = json.loads(run.report_path.read_text())
    assert report["cost_attribution"]["late"]["total_usd"] == pytest.approx(_CALL_USD)
    assert any("after this run had ended" in note for note in report["notes"])
    # The in-run component is preserved; the headline total includes the annex.
    assert report["cost"]["estimated_usd"] == pytest.approx(_CALL_USD)
    assert report["cost"]["post_run_estimated_usd"] == pytest.approx(_CALL_USD)
    assert report["cost"]["total_usd"] == pytest.approx(2 * _CALL_USD)


def test_llm_output_after_a_restart_interruption_is_kept_and_attributed_late(
    tmp_path: Path,
) -> None:
    first = OpenClawRunRecorder(tmp_path, config=_PRICED)
    first.process_payload(_before("r1"))
    first.process_payload(_llm("r1"))
    restarted = OpenClawRunRecorder(tmp_path, config=_PRICED)
    restarted.startup()
    run = _run(tmp_path, "r1")
    assert run.read_manifest()["status"] == "interrupted"
    before = _snapshot(run)

    restarted.process_payload(_llm("r1"))

    assert _snapshot(run) == before
    assert [line["after_status"] for line in _late_lines(run)] == ["interrupted"]
    assert build_run_report(run).cost.total_usd == pytest.approx(2 * _CALL_USD)
    assert build_cost_attribution(run).late.total_usd == pytest.approx(_CALL_USD)


def test_llm_output_racing_a_completion_on_the_live_run_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The payload resolved the run while it was live, then waited on its lock
    # while agent_end completed it.
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    active = recorder._runs["r1"]
    recorder.process_payload(_end("r1"))
    monkeypatch.setattr(recorder, "_get_or_create_run", lambda *_args: active)

    recorder.process_payload(_llm("r1"))

    run = _run(tmp_path, "r1")
    assert len(_late_lines(run)) == 1
    assert build_cost_attribution(run).late.costed_events == 1


def test_llm_output_for_a_finished_run_still_in_memory_waits_on_its_lock(
    tmp_path: Path,
) -> None:
    # The run's manifest is terminal while its entry is still in the live map:
    # the late hook is journaled under the run's own lock, then the entry dropped.
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    active = recorder._runs["r1"]
    recorder.process_payload(_end("r1"))
    recorder._runs["r1"] = active

    recorder.process_payload(_llm("r1"))

    assert "r1" not in recorder._runs
    assert build_cost_attribution(_run(tmp_path, "r1")).late.total_usd == pytest.approx(_CALL_USD)


def test_a_late_hook_without_cost_is_still_ignored(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    before = _snapshot(run)

    recorder.process_payload(_tool("r1"))
    recorder.process_payload(_end("r1"))

    assert _snapshot(run) == before
    assert not run.late_events_path.exists()


def test_a_late_llm_output_never_reaches_a_detector(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")

    recorder.process_payload(_llm("r1"))

    assert load_trace(run.trace_path).events == []
    assert not run.decisions_path.exists() or run.decisions_path.read_text() == ""


def test_a_run_without_trace_json_still_reports_its_llm_cost(tmp_path: Path) -> None:
    # Budget detector disabled (the default): before OBS-5 the cost of a run
    # that never reached agent_end was invisible in its report.
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1"))
    OpenClawRunRecorder(tmp_path, config=_PRICED).startup()

    report = build_run_report(_run(tmp_path, "r1"))

    assert report.cost.estimated_usd == pytest.approx(_CALL_USD)
    assert report.cost.total_usd == pytest.approx(_CALL_USD)
    assert "read back from the raw hook journal" in report.cost.source


def test_a_run_without_trace_json_and_without_llm_cost_is_still_unmeasured(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_tool("r1"))
    OpenClawRunRecorder(tmp_path, config=_PRICED).startup()

    report = build_run_report(_run(tmp_path, "r1"))

    assert report.cost.total_usd is None
    assert "no trace.json" in report.cost.source


def test_a_blank_ctx_run_id_does_not_shadow_the_event_run_id(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("", event_run_id="r1"))
    recorder.process_payload(_end("r1"))

    attribution = build_cost_attribution(_run(tmp_path, "r1"))
    assert attribution.attributed.total_usd == pytest.approx(_CALL_USD)
    assert _orphans(tmp_path) == []


# --------------------------------------------------------------------------- #
# Cost that is legitimately outside any turn
# --------------------------------------------------------------------------- #


def test_cost_without_any_run_id_is_never_attached_to_the_sessions_run(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    # Same session, same moment, no runId: correlation is not causality.
    recorder.process_payload(_llm(None, session="sess-1"))
    recorder.process_payload(_end("r1"))

    run_attribution = build_cost_attribution(_run(tmp_path, "r1"))
    assert run_attribution.attributed.costed_events == 0
    assert run_attribution.unattributed == ()
    store = build_store_cost_attribution(RunStore(tmp_path).list_runs())
    assert store.attributed.total_usd is None
    assert store.unattributed == (
        UnattributedCost(
            reason=UNATTRIBUTED_MISSING_RUN_ID,
            trigger=TRIGGER_ABSENT,
            share=CostShare(estimated_usd=pytest.approx(_CALL_USD), costed_events=1),
        ),
    )
    _assert_partition(store)


def test_unattributed_cost_is_grouped_by_the_sources_own_trigger(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_llm(None, trigger="cron"))
    recorder.process_payload(_llm(None, trigger="cron"))
    recorder.process_payload(_llm(None, trigger="heartbeat"))
    recorder.process_payload(_llm(None))

    store = build_store_cost_attribution(RunStore(tmp_path).list_runs())

    by_trigger = {entry.trigger: entry.share for entry in store.unattributed}
    assert {entry.reason for entry in store.unattributed} == {UNATTRIBUTED_MISSING_RUN_ID}
    assert set(by_trigger) == {"cron", "heartbeat", TRIGGER_ABSENT}
    assert by_trigger["cron"].costed_events == 2
    assert by_trigger["cron"].total_usd == pytest.approx(2 * _CALL_USD)
    assert by_trigger["heartbeat"].total_usd == pytest.approx(_CALL_USD)
    assert store.unattributed_total.total_usd == pytest.approx(4 * _CALL_USD)
    _assert_partition(store)


def test_a_run_named_by_the_source_without_a_recorded_start_is_still_attributed(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_end("r1"))

    attribution = build_cost_attribution(_run(tmp_path, "r1"))

    assert attribution.attributed.total_usd == pytest.approx(_CALL_USD)
    assert attribution.without_turn_start.total_usd == pytest.approx(_CALL_USD)
    assert attribution.unattributed == ()


# --------------------------------------------------------------------------- #
# Nothing invented, consistent with the existing pricing
# --------------------------------------------------------------------------- #


def test_unknown_model_and_missing_counter_are_counted_never_priced(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1", model="mystery"))
    recorder.process_payload(_llm("r1", total=None))
    recorder.process_payload(_end("r1"))

    attribution = build_cost_attribution(_run(tmp_path, "r1"))

    assert attribution.attributed.costed_events == 0
    assert attribution.attributed.unmeasured_events == 2
    assert attribution.attributed.total_usd is None


def test_a_configured_free_model_is_a_measured_zero(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1", model="free"))
    recorder.process_payload(_end("r1"))

    attribution = build_cost_attribution(_run(tmp_path, "r1"))

    assert attribution.attributed.costed_events == 1
    assert attribution.attributed.total_usd == 0.0


def test_a_run_recorded_without_a_price_table_is_never_repriced(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, pipeline_factory=lambda: Pipeline([]))
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    assert run.read_manifest()["llm_pricing_snapshot"] is None

    attribution = build_cost_attribution(run)

    assert attribution.attributed.costed_events == 0
    assert attribution.attributed.unmeasured_events == 1


def test_attribution_uses_the_prices_pinned_at_recording(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_llm("r1", total=4000))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    traced = sum(event.cost_usd or 0.0 for event in load_trace(run.trace_path).events)

    attribution = build_cost_attribution(run)

    assert attribution.attributed.total_usd == pytest.approx(traced)
    assert attribution.attributed.total_usd == pytest.approx(5 * _CALL_USD)


# The split OpenClaw reports, mostly cache reads: 100 * $1 + 50 * $2 + 850 * $0.1
# per million at the cached rates below, against _CALL_USD for the total split.
_SPLIT = {"input": 100, "output": 50, "cacheRead": 850, "total": 1000}
_SPLIT_USD = 0.000285
_CACHED = PathlayerConfig(pricing=PricingSection(models={"model": ModelPrice(1.0, 2.0, 0.1, 1.25)}))


def test_a_reported_split_is_priced_at_the_pinned_cache_rates(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_CACHED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1", usage=_SPLIT))
    recorder.process_payload(_end("r1"))
    recorder.process_payload(_llm("r1", usage=_SPLIT))  # late, as OpenClaw 2026.7 sends it
    run = _run(tmp_path, "r1")
    snapshot = run.read_manifest()["llm_pricing_snapshot"]
    assert snapshot["split_usage"] is True
    assert snapshot["models"]["model"]["cache_read_usd_per_mtok"] == 0.1
    traced = sum(event.cost_usd or 0.0 for event in load_trace(run.trace_path).events)

    attribution = build_cost_attribution(run)

    assert traced == pytest.approx(_SPLIT_USD)
    assert attribution.attributed.total_usd == pytest.approx(2 * _SPLIT_USD)
    assert attribution.late.total_usd == pytest.approx(_SPLIT_USD)


def _gpt_4o_run(root: Path, usage: dict[str, int]) -> RunDirectory:
    # gpt-4o is in the default table without a cache rate.
    recorder = OpenClawRunRecorder(root, config=PathlayerConfig())
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1", model="gpt-4o", usage=usage))
    recorder.process_payload(_end("r1"))
    return _run(root, "r1")


def test_a_partially_priced_call_says_which_cache_tokens_it_left_out(tmp_path: Path) -> None:
    run = _gpt_4o_run(tmp_path, {"input": 1_000, "cacheRead": 1_000_000})

    attribution = build_cost_attribution(run)
    report = build_run_report(run)

    # 1,000 input tokens at $2.50 per million; the million cache reads have no rate.
    assert attribution.attributed == CostShare(
        estimated_usd=pytest.approx(0.0025),
        costed_events=1,
        partially_priced_events=1,
        unpriced_cache_tokens=1_000_000,
    )
    assert "1 partially priced; 1,000,000 cache token(s) without a known rate" in (
        attribution.attributed.describe()
    )
    assert report.cost.source.count("estimated from") == 1
    assert "estimated from the reported input, output and cache token counters" in (
        report.cost.source
    )
    assert "input/output ratio" not in report.cost.source
    assert (
        "1,000,000 cache token(s) over 1 model call(s) are not priced: no cache rate is "
        "known for gpt-4o" in report.cost.source
    )


def test_a_call_with_only_unpriced_cache_tokens_is_unmeasured(tmp_path: Path) -> None:
    run = _gpt_4o_run(tmp_path, {"cacheRead": 1_000_000})

    attribution = build_cost_attribution(run)
    report = build_run_report(run)

    assert attribution.attributed == CostShare(unmeasured_events=1, unpriced_cache_tokens=1_000_000)
    assert attribution.attributed.total_usd is None
    assert report.cost.estimated_usd is None
    assert "1,000,000 cache token(s) over 1 model call(s) are not priced" in report.cost.source


def test_post_run_usage_names_its_method_and_unpriced_cache_tokens(tmp_path: Path) -> None:
    # OpenClaw 2026.7 reports the usage after agent_end: the report still says
    # how that spend was priced and which cache tokens it left out.
    recorder = OpenClawRunRecorder(tmp_path, config=PathlayerConfig())
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    recorder.process_payload(
        _llm("r1", model="gpt-4o", usage={"input": 1_000, "cacheRead": 1_000_000})
    )
    recorder.process_payload(_llm("r1", model="gpt-4o", usage={"cacheRead": 500}))

    report = build_run_report(_run(tmp_path, "r1"))

    assert report.cost.post_run_estimated_usd == pytest.approx(0.0025)
    assert report.cost_attribution is not None
    assert report.cost_attribution.late == CostShare(
        estimated_usd=pytest.approx(0.0025),
        costed_events=1,
        unmeasured_events=1,
        partially_priced_events=1,
        unpriced_cache_tokens=1_000_500,
    )
    assert (
        "$0.002500 estimated from the reported input, output and cache token counters at "
        "local rates, received after the run" in report.cost.source
    )
    assert (
        "1,000,500 cache token(s) over 2 usage report(s) received after the run are not "
        "priced: no cache rate is known for gpt-4o" in report.cost.source
    )


def test_post_run_usage_with_only_unpriced_cache_tokens_is_still_named(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=PathlayerConfig())
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    recorder.process_payload(_llm("r1", model="gpt-4o", usage={"cacheRead": 500}))

    report = build_run_report(_run(tmp_path, "r1"))

    assert report.cost.post_run_estimated_usd is None
    assert "estimated from" not in report.cost.source
    assert "500 cache token(s) over 1 usage report(s) received after the run" in report.cost.source


def test_a_total_only_call_names_the_input_ratio_method(tmp_path: Path) -> None:
    run = _gpt_4o_run(tmp_path, {"total": 1_000})

    report = build_run_report(run)

    assert "estimated from total tokens and an input/output ratio" in report.cost.source
    assert "cache token" not in report.cost.source


def test_a_run_recorded_before_split_pricing_keeps_its_total_split(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_CACHED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1", usage=_SPLIT))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    # The snapshot a 0.1.0 recorder pinned: no rule, no cache rates.
    run.update_manifest(
        llm_pricing_snapshot={"input_ratio": 0.8, "models": {"model": dict(_RATES)}}
    )

    attribution = build_cost_attribution(run)

    assert attribution.attributed.total_usd == pytest.approx(_CALL_USD)


def test_a_run_recorded_before_wrapped_ids_keeps_them_unmeasured(tmp_path: Path) -> None:
    # Pathlayer 0.1.0 did not unwrap Bedrock ids: its runs priced nothing for
    # them, and a report rebuilt today must not invent that figure.
    bedrock = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    recorder = OpenClawRunRecorder(tmp_path, config=PathlayerConfig())
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    recorder.process_payload(_llm("r1", model=bedrock))
    run = _run(tmp_path, "r1")
    snapshot = run.read_manifest()["llm_pricing_snapshot"]
    assert snapshot["unwrap_model_ids"] is True
    assert build_cost_attribution(run).late.costed_events == 1

    # The snapshot a 0.1.0 recorder pinned: no split, no unwrapping.
    run.update_manifest(
        llm_pricing_snapshot={
            "input_ratio": 0.8,
            "models": {"claude-sonnet-4": {"input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0}},
        }
    )

    late = build_cost_attribution(run).late
    assert late == CostShare(unmeasured_events=1)


def test_a_corrupt_price_snapshot_is_unmeasured(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")

    for snapshot in (
        {"input_ratio": True, "models": {}},
        {"input_ratio": 0.8, "models": []},
        {"input_ratio": 0.8, "models": {"model": "x"}},
        {"input_ratio": 0.8, "models": {"model": {"input_usd_per_mtok": 1.0}}},
        {"input_ratio": 5.0, "models": {}},
        {"input_ratio": float("nan"), "models": {}},
        {"input_ratio": -0.1, "models": {}},
        *(
            {"input_ratio": 0.8, "models": {"model": {**_RATES, side: bad}}}
            for side in (
                "input_usd_per_mtok",
                "output_usd_per_mtok",
                "cache_read_usd_per_mtok",
                "cache_write_usd_per_mtok",
            )
            for bad in (-1.0, float("nan"), float("inf"), "1.0", True)
        ),
    ):
        run.update_manifest(llm_pricing_snapshot=snapshot)
        attribution = build_cost_attribution(run)
        assert attribution.attributed.costed_events == 0
        assert attribution.attributed.unmeasured_events == 1


def test_a_run_recorded_before_0_1_1_keeps_post_run_usage_out_of_its_total(
    tmp_path: Path,
) -> None:
    # 0.1.0 attributed usage received after agent_end to the run but left it out
    # of the total: a report rebuilt today keeps that total.
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    recorder.process_payload(_llm("r1"))
    run = _run(tmp_path, "r1")
    assert build_run_report(run).cost.post_run_estimated_usd == pytest.approx(_CALL_USD)
    run.update_manifest(llm_pricing_snapshot={"input_ratio": 0.8, "models": {"model": _RATES}})

    report = build_run_report(run)

    assert report.cost.post_run_estimated_usd is None
    assert report.cost.total_usd is None
    assert report.cost_attribution is not None
    assert report.cost_attribution.late.total_usd == pytest.approx(_CALL_USD)
    assert any("recorded before Pathlayer 0.1.1" in note for note in report.notes)
    assert "received after the run" not in report.cost.source


def test_post_run_usage_without_a_usable_snapshot_is_not_blamed_on_0_1_0(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    recorder.process_payload(_llm("r1"))
    run = _run(tmp_path, "r1")
    snapshot = run.read_manifest()["llm_pricing_snapshot"]
    snapshot["models"]["model"]["input_usd_per_mtok"] = -1  # unusable as a whole
    run.update_manifest(llm_pricing_snapshot=snapshot)

    report = build_run_report(run)

    assert report.cost.post_run_estimated_usd is None
    assert not any("before Pathlayer 0.1.1" in note for note in report.notes)
    assert any("Unpriced usage stays unmeasured" in note for note in report.notes)


def test_a_corrupt_hook_journal_costs_the_attribution_not_the_report(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    lines = run.raw_events_path.read_text().splitlines(keepends=True)
    run.raw_events_path.write_text("{not json\n" + "".join(lines))

    report = build_run_report(run)

    assert report.cost_attribution is None
    assert report.cost.estimated_usd == pytest.approx(_CALL_USD)  # from trace.json
    assert any("cost attribution is unavailable" in note for note in report.notes)
    assert any("leaves out any usage received after the run" in note for note in report.notes)
    assert report.to_dict()["cost_attribution"] is None
    with pytest.raises(ValueError):
        build_cost_attribution(run)


# --------------------------------------------------------------------------- #
# Report and CLI
# --------------------------------------------------------------------------- #


def _mixed_store(root: Path) -> None:
    recorder = OpenClawRunRecorder(root, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_end("r1"))
    recorder.process_payload(_llm("r1"))  # late
    recorder.process_payload(_llm(None, trigger="cron"))


def test_cli_cost_attribution_separates_captured_attributed_unattributed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _mixed_store(tmp_path)

    assert main(["--all", "--cost-attribution", "--runs-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "cost attribution over 2 run(s)" in out
    assert "captured: $0.0036 over 3 event(s)" in out
    assert "attributed to a run: $0.0024 over 2 event(s)" in out
    assert "received after the run ended: $0.0012 over 1 event(s)" in out
    assert "unattributed: $0.0012 over 1 event(s)" in out
    assert f"{UNATTRIBUTED_MISSING_RUN_ID} (trigger cron): $0.0012" in out

    assert main(["--all", "--cost-attribution", "--json", "--runs-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"] == 2
    assert payload["captured"]["total_usd"] == pytest.approx(3 * _CALL_USD)
    assert payload["attributed"]["total_usd"] == pytest.approx(2 * _CALL_USD)
    assert payload["unattributed"]["total_usd"] == pytest.approx(_CALL_USD)
    assert payload["unattributed_by_reason"] == [
        {
            "reason": UNATTRIBUTED_MISSING_RUN_ID,
            "trigger": "cron",
            "total_usd": pytest.approx(_CALL_USD),
            "measured_usd": 0.0,
            "estimated_usd": pytest.approx(_CALL_USD),
            "costed_events": 1,
            "unmeasured_events": 0,
            "partially_priced_events": 0,
            "unpriced_cache_tokens": 0,
        }
    ]


def test_cli_cost_attribution_over_no_run_measures_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--all", "--cost-attribution", "--runs-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "over 0 run(s)" in out
    assert "captured: no costed event" in out


def test_single_run_report_prints_its_attribution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _mixed_store(tmp_path)

    assert main(["r1", "--runs-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "cost attribution:\n  captured: $0.0024 over 2 event(s)" in out
    assert "  attributed to a run: $0.0024 over 2 event(s)" in out
    assert "  unattributed: no costed event" in out
    assert "note: 1 model usage report(s) reached Pathlayer after this run had ended" in out


def test_the_attribution_partition_holds_and_reports_unmeasured_events() -> None:
    attribution = CostAttribution(
        attributed=CostShare(estimated_usd=0.5, costed_events=1, unmeasured_events=1),
        without_turn_start=CostShare(unmeasured_events=1),
        unattributed=(UnattributedCost("r", "t", CostShare(measured_usd=0.25, costed_events=1)),),
        runs=1,
    )
    _assert_partition(attribution)
    assert attribution.captured.total_usd == pytest.approx(0.75)
    lines = attribution.describe_lines()
    assert lines[1] == "attributed to a run: $0.5000 over 1 event(s), 1 unmeasured"
    assert "start was never recorded: no costed event, 1 unmeasured" in lines[2]


@pytest.mark.parametrize("ordering", ["in_run", "post_run"])
@pytest.mark.parametrize("budget_enabled", [False, True])
def test_2026_7_hook_order_reports_model_and_same_total(
    tmp_path: Path,
    ordering: str,
    budget_enabled: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "openclaw_2026_7" / f"{ordering}.json"
    payloads = json.loads(fixture.read_text())
    config = PathlayerConfig(detectors=DetectorsConfig(budget=BudgetConfig(enabled=budget_enabled)))
    recorder = OpenClawRunRecorder(tmp_path, config=config, enforce=True)
    for payload in payloads:
        recorder.process_payload(payload)
    run = _run(tmp_path, "fixture-run")
    manifest = run.read_manifest()
    assert manifest["openclaw_model"] == "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert manifest["openclaw_provider"] == "bedrock"
    report = build_run_report(run)
    assert report.cost.total_usd == pytest.approx(0.021)
    assert report.cost.post_run_estimated_usd == (
        pytest.approx(0.021) if ordering == "post_run" else None
    )
    assert report.cost.estimated_usd == (
        pytest.approx(0.021) if ordering == "in_run" else (0.0 if budget_enabled else None)
    )
    assert run.late_events_path.exists() == (ordering == "post_run")
    assert not manifest["stopped"]
    assert main(["fixture-run", "--runs-root", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "$0.0210 effective" in text
    assert (
        "estimated received after run" if ordering == "post_run" else "estimated during run"
    ) in text


@pytest.mark.parametrize(
    "model,total,expected",
    [
        ("model", 1000, _CALL_USD),
        ("free", 1000, 0.0),
        ("mystery", 1000, None),
        ("model", None, None),
    ],
)
def test_post_run_only_cost_preserves_unknown_and_free(
    tmp_path: Path,
    model: str,
    total: int | None,
    expected: float | None,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    before = _snapshot(run)
    recorder.process_payload(_llm("r1", model=model, total=total))
    report = build_run_report(run)
    assert report.cost.total_usd == expected
    assert report.cost.post_run_estimated_usd == expected
    assert report.cost.estimated_usd is None
    assert _snapshot(run) == before
    assert report.cost_attribution is not None
    assert report.cost_attribution.late.unmeasured_events == (1 if expected is None else 0)


def test_multiple_post_run_usage_reports_are_added_once_each(tmp_path: Path) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    before = _snapshot(run)
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_llm("r1", total=2000))
    for _ in range(2):
        assert build_run_report(run).cost.total_usd == pytest.approx(3 * _CALL_USD)
    assert _snapshot(run) == before
    assert json.loads(run.report_path.read_text())["cost"]["total_usd"] == pytest.approx(
        3 * _CALL_USD
    )


def test_corrupt_late_annex_keeps_in_run_cost_without_inventing_post_run_cost(
    tmp_path: Path,
) -> None:
    recorder = OpenClawRunRecorder(tmp_path, config=_PRICED)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_end("r1"))
    run = _run(tmp_path, "r1")
    run.late_events_path.write_text("{broken\n")
    report = build_run_report(run)
    assert report.cost.total_usd == pytest.approx(_CALL_USD)
    assert report.cost.post_run_estimated_usd is None
    assert any("cost attribution is unavailable" in note for note in report.notes)


def test_post_run_cost_adds_to_tool_estimate_and_baseline(tmp_path: Path) -> None:
    config = PathlayerConfig(
        pricing=_PRICED.pricing,
        detectors=DetectorsConfig(
            budget=BudgetConfig(
                enabled=True,
                baseline_usd=0.5,
                tool_costs_usd={"read": 0.1},
            )
        ),
    )
    recorder = OpenClawRunRecorder(tmp_path, config=config)
    recorder.process_payload(_before("r1"))
    recorder.process_payload(_tool("r1"))
    recorder.process_payload(_llm("r1"))
    recorder.process_payload(_end("r1"))
    recorder.process_payload(_llm("r1"))
    cost = build_run_report(_run(tmp_path, "r1")).cost
    assert cost.baseline_usd == 0.5
    assert cost.estimated_usd == pytest.approx(0.1 + _CALL_USD)
    assert cost.post_run_estimated_usd == pytest.approx(_CALL_USD)
    assert cost.total_usd == pytest.approx(0.6 + 2 * _CALL_USD)
