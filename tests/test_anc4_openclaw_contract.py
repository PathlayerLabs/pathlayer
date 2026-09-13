"""ANC-4: a compiled coding contract runs unchanged in the OpenClaw recorder.

The recorder is the ANC-2 runtime that already probes live and durably
journals each proof. A compiled contract is an ordinary ANC bundle, so the
only thing to prove is that it plugs into that path as-is: checkpoint ids from
the contract, the exact live proof journaled, and a restart that restores it
without re-running the probe against today's workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pathlayer.adapters.openclaw_recorder import OpenClawRunRecorder
from pathlayer.adapters.openclaw_run_store import RunStore
from pathlayer.coding_contracts import (
    CodingContract,
    FileChanges,
    FileExists,
    write_coding_contract,
)
from pathlayer.config import PathlayerConfig


def _ctx(run_id: str) -> dict[str, Any]:
    return {"runId": run_id, "sessionKey": "sess-1", "agentId": "agent-1"}


def _before(run_id: str) -> dict[str, Any]:
    return {
        "hook": "before_agent_run",
        "event": {"prompt": "write the report", "messages": []},
        "ctx": _ctx(run_id),
    }


def _tool(run_id: str, call: str) -> dict[str, Any]:
    return {
        "hook": "after_tool_call",
        "event": {
            "toolName": "write",
            "params": {"path": "report.md"},
            "result": "ok",
            "toolCallId": call,
            "runId": run_id,
        },
        "ctx": _ctx(run_id),
    }


def test_compiled_contract_is_probed_journaled_and_restored_by_openclaw(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "notes.md").write_text("draft\n", encoding="utf-8")
    installed = write_coding_contract(
        CodingContract(
            task="write-report",
            objective="Produce the report.",
            workspace=workspace,
            progress=(FileExists("report.md"), FileChanges("notes.md")),
        ),
        output_dir=tmp_path / "bundle",
    )
    config = PathlayerConfig.from_mapping(
        {"detectors": {"progress_proof": installed.progress_proof_config()}}
    )
    runs_root = tmp_path / "runs"

    recorder = OpenClawRunRecorder(runs_root, config=config)
    recorder.process_payload(_before("run-1"))
    (workspace / "report.md").write_text("# Report\n", encoding="utf-8")  # the agent's effect
    recorder.process_payload(_tool("run-1", "call-1"))

    live_event = recorder._runs["run-1"].events[0]
    live_proof = live_event.metadata["progress_proof"]
    assert live_proof["checkpoints"] == {
        "file_changes:notes.md": False,
        "file_exists:report.md": True,
    }
    assert live_event.metadata["proof_spec"]["required"] == list(
        installed.compiled.bundle.proof_spec.required
    )

    run = RunStore(runs_root).resolve_run("run-1")
    assert run is not None
    records = [json.loads(line) for line in run.raw_events_path.read_text().splitlines()]
    sidecars = [r["payload"] for r in records if r.get("hook") == "pathlayer_progress_proof"]
    assert [sidecar["progress_proof"] for sidecar in sidecars] == [live_proof]

    # Today's workspace no longer matches; a restore must replay the journaled
    # proof, not re-probe, or the durable state would diverge.
    (workspace / "report.md").unlink()
    restarted = OpenClawRunRecorder(runs_root, config=config)
    restarted.startup()

    manifest = run.read_manifest()
    assert manifest["status"] == "interrupted"
    assert manifest.get("persistence_error") is None
    assert manifest.get("state_durable") is not False
