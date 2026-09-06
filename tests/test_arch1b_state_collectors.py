"""ARCH-1B filesystem/pytest evidence collectors and local export."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import pathlayer.validation.state_collectors as collectors_module
from pathlayer.detectors.state_evidence import StateEvidence
from pathlayer.validation import _pytest_evidence_plugin as pytest_plugin
from pathlayer.validation import (
    collect_filesystem_evidence,
    collect_pytest_evidence,
    dump_state_evidence_export,
    load_state_evidence_export,
    unavailable_state_evidence,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "state_evidence" / "arch1b"
_TEST_ID = "case_target.py::test_normalize"


def _filesystem(
    path: Path,
    *,
    phase: str = "before",
    before: StateEvidence | None = None,
    ref: str = "call-edit-target",
    resource_id: str = "target.py",
) -> StateEvidence:
    return collect_filesystem_evidence(
        path,
        tool_event_ref=ref,
        resource_id=resource_id,
        phase=phase,  # type: ignore[arg-type]
        before=before,
    )


def _tests(
    cwd: Path,
    *,
    phase: str = "before",
    before: StateEvidence | None = None,
    test_ids: tuple[str, ...] = (_TEST_ID,),
    ref: str = "call-edit-target",
) -> StateEvidence:
    return collect_pytest_evidence(
        test_ids,
        cwd=cwd,
        tool_event_ref=ref,
        resource_id="selected-normalizer-tests",
        phase=phase,  # type: ignore[arg-type]
        before=before,
    )


def _scenario_records(name: str) -> tuple[StateEvidence, ...]:
    root = _FIXTURES / name
    before_root = root / "before"
    after_root = root / "after"
    filesystem_before = _filesystem(before_root / "target.py")
    tests_before = _tests(before_root)
    filesystem_after = _filesystem(
        after_root / "target.py", phase="after", before=filesystem_before
    )
    tests_after = _tests(after_root, phase="after", before=tests_before)
    return filesystem_before, tests_before, filesystem_after, tests_after


def test_filesystem_unchanged_and_changed_transitions(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"before")
    before = _filesystem(target)

    unchanged = _filesystem(target, phase="after", before=before)
    target.write_bytes(b"after")
    changed = _filesystem(target, phase="after", before=before)

    assert before.raw_fingerprint == {
        "exists": True,
        "sha256": "6db7d803e74f1ffa7d8f5adc0bf95b3e15bf4c8373fffadf546227cc6c6742cb",
    }
    assert unchanged.delta == {
        "changed": False,
        "hash_before": before.canonical_fingerprint,
        "hash_after": before.canonical_fingerprint,
    }
    assert changed.delta == {
        "changed": True,
        "hash_before": before.canonical_fingerprint,
        "hash_after": changed.canonical_fingerprint,
    }


def test_filesystem_created_and_deleted_transitions(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    missing = _filesystem(target)
    assert missing.raw_fingerprint == {"exists": False, "sha256": None}
    assert missing.canonical_fingerprint == "missing"

    target.write_text("created", encoding="utf-8")
    created = _filesystem(target, phase="after", before=missing)
    assert created.delta == {
        "changed": True,
        "hash_before": None,
        "hash_after": created.canonical_fingerprint,
    }

    existing = _filesystem(target, ref="call-delete-target")
    target.unlink()
    deleted = _filesystem(
        target,
        phase="after",
        before=existing,
        ref="call-delete-target",
    )
    assert deleted.delta == {
        "changed": True,
        "hash_before": existing.canonical_fingerprint,
        "hash_after": None,
    }


def test_filesystem_fingerprint_ignores_path_and_mtime(tmp_path: Path) -> None:
    first = tmp_path / "one" / "target.txt"
    second = tmp_path / "two" / "renamed.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")
    os.utime(first, (1, 1))
    os.utime(second, (2_000_000_000, 2_000_000_000))

    first_evidence = _filesystem(first)
    second_evidence = _filesystem(second, ref="call-other", resource_id="renamed.txt")

    assert first_evidence.canonical_fingerprint == second_evidence.canonical_fingerprint


@pytest.mark.parametrize(
    "kind",
    [
        "directory",
        pytest.param(
            "symlink",
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="symlink_to requires SeCreateSymbolicLinkPrivilege on Windows",
            ),
        ),
    ],
)
def test_filesystem_non_regular_resource_is_collection_failed(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "resource"
    if kind == "directory":
        target.mkdir()
    else:
        source = tmp_path / "source"
        source.write_text("content", encoding="utf-8")
        target.symlink_to(source)

    evidence = _filesystem(target)

    assert evidence.status == "collection_failed"
    assert evidence.raw_fingerprint is None
    assert evidence.canonical_fingerprint is None


def test_filesystem_path_with_embedded_nul_is_collection_failed(tmp_path: Path) -> None:
    # lstat()/os.open() raise ValueError (not OSError) for a NUL byte in the
    # path; without catching it too this crashes the collector instead of
    # degrading to collection_failed like any other unusable resource.
    target = tmp_path / "resource\x00name"

    evidence = _filesystem(target)

    assert evidence.status == "collection_failed"
    assert evidence.raw_fingerprint is None
    assert evidence.canonical_fingerprint is None


def test_filesystem_read_error_is_collection_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.write_text("content", encoding="utf-8")

    def fail_open(path: object, flags: int) -> int:
        del path, flags
        raise PermissionError("denied")

    monkeypatch.setattr(os, "open", fail_open)
    assert _filesystem(target).status == "collection_failed"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="symlink_to requires SeCreateSymbolicLinkPrivilege on Windows",
)
def test_filesystem_replacement_with_symlink_is_collection_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    secret = tmp_path / "secret"
    target.write_text("public", encoding="utf-8")
    secret.write_text("private", encoding="utf-8")
    original_open = os.open

    def replace_then_open(path: object, flags: int) -> int:
        target.unlink()
        target.symlink_to(secret)
        return original_open(path, flags)

    monkeypatch.setattr(os, "open", replace_then_open)
    assert _filesystem(target).status == "collection_failed"


def test_unavailable_before_remains_explicit_and_prevents_delta(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("content", encoding="utf-8")
    before = unavailable_state_evidence(
        tool_event_ref="call-edit-target",
        resource_type="filesystem",
        resource_id="target.py",
        phase="before",
    )

    after = _filesystem(target, phase="after", before=before)

    assert before.status == "unavailable"
    assert after.status == "observed"
    assert after.delta is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase": "after"},
        {
            "phase": "before",
            "before": unavailable_state_evidence(
                tool_event_ref="call-edit-target",
                resource_type="filesystem",
                resource_id="target.py",
                phase="before",
            ),
        },
    ],
)
def test_capture_rejects_missing_or_extra_prior(tmp_path: Path, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="before"):
        _filesystem(tmp_path / "target", **kwargs)  # type: ignore[arg-type]


def test_capture_rejects_prior_from_another_event(tmp_path: Path) -> None:
    target = tmp_path / "target"
    prior = _filesystem(target, ref="call-one")
    with pytest.raises(ValueError, match="does not match"):
        _filesystem(target, phase="after", before=prior, ref="call-two")


def test_pytest_captures_failure_then_success_from_real_runs() -> None:
    records = _scenario_records("healthy_effect")
    tests_before, tests_after = records[1], records[3]

    assert tests_before.status == "observed"
    assert tests_before.canonical_fingerprint == {"failing": [_TEST_ID], "passing": []}
    assert tests_after.canonical_fingerprint == {"failing": [], "passing": [_TEST_ID]}
    assert tests_after.delta == {
        "newly_passing": [_TEST_ID],
        "newly_failing": [],
        "still_failing": [],
    }


def test_pytest_captures_healthy_regression_and_sorts_outcomes(tmp_path: Path) -> None:
    tests_file = tmp_path / "test_sample.py"
    tests_file.write_text(
        "def test_z():\n    assert True\n\ndef test_a():\n    assert True\n",
        encoding="utf-8",
    )
    selected = ("test_sample.py::test_z", "test_sample.py::test_a")
    before = _tests(tmp_path, test_ids=selected)
    tests_file.write_text(
        "def test_z():\n    assert False\n\ndef test_a():\n    assert True\n",
        encoding="utf-8",
    )

    after = _tests(tmp_path, phase="after", before=before, test_ids=selected)

    assert before.canonical_fingerprint == {
        "failing": [],
        "passing": ["test_sample.py::test_a", "test_sample.py::test_z"],
    }
    assert after.delta == {
        "newly_passing": [],
        "newly_failing": ["test_sample.py::test_z"],
        "still_failing": [],
    }


def test_pytest_runs_same_stable_order_for_reordered_selection(tmp_path: Path) -> None:
    marker = tmp_path / "ordered"
    (tmp_path / "test_order.py").write_text(
        "from pathlib import Path\n\n"
        "def test_a():\n    Path('ordered').write_text('ready')\n\n"
        "def test_b():\n    assert Path('ordered').read_text() == 'ready'\n",
        encoding="utf-8",
    )
    a = "test_order.py::test_a"
    b = "test_order.py::test_b"
    before = _tests(tmp_path, test_ids=(b, a))
    marker.unlink()

    after = _tests(tmp_path, phase="after", before=before, test_ids=(a, b))

    assert (
        before.canonical_fingerprint
        == after.canonical_fingerprint
        == {
            "failing": [],
            "passing": [a, b],
        }
    )
    assert after.delta == {"newly_passing": [], "newly_failing": [], "still_failing": []}


def test_pytest_skip_is_an_observed_non_pass(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n\n@pytest.mark.skip\ndef test_skipped():\n    pass\n",
        encoding="utf-8",
    )
    evidence = _tests(tmp_path, test_ids=("test_sample.py::test_skipped",))
    assert evidence.status == "observed"
    assert evidence.canonical_fingerprint == {
        "failing": ["test_sample.py::test_skipped"],
        "passing": [],
    }


def test_pytest_setup_error_is_an_observed_non_pass(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def broken():\n    raise RuntimeError('setup failed')\n\n"
        "def test_error(broken):\n    pass\n",
        encoding="utf-8",
    )
    evidence = _tests(tmp_path, test_ids=("test_sample.py::test_error",))
    assert evidence.status == "observed"
    assert evidence.canonical_fingerprint == {
        "failing": ["test_sample.py::test_error"],
        "passing": [],
    }


def test_pytest_timeout_is_collection_failed(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "import time\n\ndef test_slow():\n    time.sleep(10)\n",
        encoding="utf-8",
    )
    evidence = collect_pytest_evidence(
        ("test_sample.py::test_slow",),
        cwd=tmp_path,
        tool_event_ref="call-edit-target",
        resource_id="slow-test",
        phase="before",
        timeout_seconds=0.01,
    )
    assert evidence.status == "collection_failed"


def test_pytest_timeout_terminates_descendants(tmp_path: Path) -> None:
    started = tmp_path / "descendant-started"
    marker = tmp_path / "descendant-survived"
    (tmp_path / "test_sample.py").write_text(
        "import subprocess\nimport sys\nimport time\n\n"
        "def test_spawn():\n"
        "    subprocess.Popen([sys.executable, '-c', "
        '"import time; from pathlib import Path; '
        "Path('descendant-started').write_text('started'); time.sleep(1); "
        "Path('descendant-survived').write_text('alive')\"])\n"
        "    time.sleep(10)\n",
        encoding="utf-8",
    )
    evidence = collect_pytest_evidence(
        ("test_sample.py::test_spawn",),
        cwd=tmp_path,
        tool_event_ref="call-edit-target",
        resource_id="spawn-test",
        phase="before",
        timeout_seconds=0.5,
    )
    assert started.exists()
    time.sleep(1.1)
    assert evidence.status == "collection_failed"
    assert not marker.exists()


def test_pytest_interruption_terminates_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(
        poll=lambda: None,
        wait=lambda *, timeout: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    terminated: list[object] = []
    monkeypatch.setattr(
        collectors_module,
        "_terminate_and_reap_pytest",
        lambda child: terminated.append(child),
    )

    with pytest.raises(KeyboardInterrupt):
        collectors_module._wait_for_pytest(process, 1.0)

    assert terminated == [process]


def test_pytest_missing_selection_is_collection_failed(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text("def test_present():\n    pass\n", encoding="utf-8")
    evidence = _tests(tmp_path, test_ids=("test_sample.py::test_missing",))
    assert evidence.status == "collection_failed"


def test_pytest_ignores_ambient_plugin_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_PLUGINS", "definitely_missing_pathlayer_plugin")
    evidence = _tests(_FIXTURES / "healthy_effect" / "before")
    assert evidence.status == "observed"


def test_pytest_failed_before_capture_keeps_after_delta_unknown(tmp_path: Path) -> None:
    before = _tests(tmp_path, test_ids=("test_sample.py::test_present",))
    assert before.status == "collection_failed"
    (tmp_path / "test_sample.py").write_text("def test_present():\n    pass\n", encoding="utf-8")

    after = _tests(
        tmp_path,
        phase="after",
        before=before,
        test_ids=("test_sample.py::test_present",),
    )

    assert after.status == "observed"
    assert after.delta is None
    export = tmp_path / "evidence.json"
    dump_state_evidence_export((before, after), export)
    assert load_state_evidence_export(export) == (before, after)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan"), "fast"])
def test_pytest_rejects_invalid_timeout(tmp_path: Path, timeout: object) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        collect_pytest_evidence(
            (_TEST_ID,),
            cwd=tmp_path,
            tool_event_ref="call",
            resource_id="tests",
            phase="before",
            timeout_seconds=timeout,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("test_ids", [(), ("",), ("same", "same"), ("a\x00b",)])
def test_pytest_rejects_invalid_nodeids(tmp_path: Path, test_ids: tuple[str, ...]) -> None:
    # The embedded-NUL case matters beyond well-formedness: subprocess.Popen
    # raises ValueError for it, uncaught by _run_pytest's except tuple, so this
    # must be rejected here rather than crashing the collector.
    with pytest.raises(ValueError, match=r"nodeid|unique"):
        _tests(tmp_path, test_ids=test_ids)


def test_pytest_rejects_option_like_nodeid_without_side_effect(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-be-overwritten"
    marker.write_text("sentinel", encoding="utf-8")
    with pytest.raises(ValueError, match="nodeid"):
        _tests(tmp_path, test_ids=(f"--junitxml={marker}",))
    assert marker.read_text(encoding="utf-8") == "sentinel"


def test_pytest_after_rejects_changed_selection() -> None:
    before = _scenario_records("healthy_effect")[1]
    with pytest.raises(ValueError, match="different pytest selection"):
        _tests(
            _FIXTURES / "healthy_effect" / "after",
            phase="after",
            before=before,
            test_ids=("test_target.py::other",),
        )


def test_fixture_exports_round_trip_and_distinguish_effect(tmp_path: Path) -> None:
    healthy = _scenario_records("healthy_effect")
    faulty = _scenario_records("faulty_no_effect")
    expected_healthy = load_state_evidence_export(
        _FIXTURES / "healthy_effect" / "expected_export.v1.json"
    )
    expected_faulty = load_state_evidence_export(
        _FIXTURES / "faulty_no_effect" / "expected_export.v1.json"
    )
    healthy_path = tmp_path / "healthy.json"
    faulty_path = tmp_path / "faulty.json"

    dump_state_evidence_export(healthy, healthy_path)
    dump_state_evidence_export(faulty, faulty_path)

    assert load_state_evidence_export(healthy_path) == healthy
    assert load_state_evidence_export(faulty_path) == faulty
    assert healthy == expected_healthy
    assert faulty == expected_faulty
    assert {record.tool_event_ref for record in healthy} == {"call-edit-target"}
    assert healthy[2].delta is not None and healthy[2].delta["changed"] is True
    assert healthy[3].delta is not None and healthy[3].delta["newly_passing"] == [_TEST_ID]
    assert faulty[2].delta is not None and faulty[2].delta["changed"] is False
    assert faulty[3].delta is not None and faulty[3].delta["still_failing"] == [_TEST_ID]


def test_export_is_deterministic_and_preserves_record_order(tmp_path: Path) -> None:
    records = _scenario_records("healthy_effect")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    dump_state_evidence_export(records, first)
    dump_state_evidence_export(records, second)
    assert first.read_bytes() == second.read_bytes()
    assert [record.phase for record in load_state_evidence_export(first)] == [
        "before",
        "before",
        "after",
        "after",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": 2, "evidence": []},
        {"schema_version": 1, "evidence": {}},
        {"schema_version": 1, "evidence": [], "extra": True},
        {"schema_version": 1, "evidence": [{"not": "evidence"}]},
    ],
)
def test_export_rejects_invalid_document(tmp_path: Path, payload: object) -> None:
    export = tmp_path / "evidence.json"
    export.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_state_evidence_export(export)


def test_export_rejects_duplicates_incomplete_pairs_and_wrong_order(tmp_path: Path) -> None:
    before, _, after, _ = _scenario_records("healthy_effect")
    with pytest.raises(ValueError, match="duplicate"):
        dump_state_evidence_export((before, before, after), tmp_path / "duplicate.json")
    with pytest.raises(ValueError, match="exactly one"):
        dump_state_evidence_export((before,), tmp_path / "incomplete.json")
    with pytest.raises(ValueError, match="must precede"):
        dump_state_evidence_export((after, before), tmp_path / "reversed.json")

    all_records = _scenario_records("healthy_effect")
    interleaved = (all_records[0], all_records[2], all_records[1], all_records[3])
    with pytest.raises(ValueError, match="all before"):
        dump_state_evidence_export(interleaved, tmp_path / "interleaved.json")


def test_export_rejects_observed_pair_without_delta(tmp_path: Path) -> None:
    before, _, after, _ = _scenario_records("healthy_effect")
    without_delta = replace(after, delta=None)
    with pytest.raises(ValueError, match="delta"):
        dump_state_evidence_export((before, without_delta), tmp_path / "missing-delta.json")


def test_export_rejects_fabricated_delta_and_fingerprint(tmp_path: Path) -> None:
    before, _, after, _ = _scenario_records("healthy_effect")
    fabricated_delta = replace(
        after,
        delta={
            "changed": False,
            "hash_before": before.canonical_fingerprint,
            "hash_after": after.canonical_fingerprint,
        },
    )
    with pytest.raises(ValueError, match="does not match"):
        dump_state_evidence_export((before, fabricated_delta), tmp_path / "bad-delta.json")

    fabricated_fingerprint = replace(before, canonical_fingerprint="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="filesystem evidence"):
        dump_state_evidence_export(
            (fabricated_fingerprint, after), tmp_path / "bad-fingerprint.json"
        )


def test_export_rejects_unknown_collector_and_non_integer_version(tmp_path: Path) -> None:
    before, _, after, _ = _scenario_records("healthy_effect")
    unknown_before = replace(
        before,
        resource_type="database",
        collector_version="database-collector@1",
    )
    unknown_after = replace(
        after,
        resource_type="database",
        collector_version="database-collector@1",
    )
    with pytest.raises(ValueError, match="unsupported resource_type"):
        dump_state_evidence_export((unknown_before, unknown_after), tmp_path / "unknown.json")

    export = tmp_path / "float-version.json"
    export.write_text('{"schema_version": 1.0, "evidence": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_state_evidence_export(export)


def test_export_loader_bounds_bytes_and_json_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * 33)
    monkeypatch.setattr(collectors_module, "_MAX_EXPORT_BYTES", 32)
    with pytest.raises(ValueError, match="exceeds"):
        load_state_evidence_export(oversized)

    monkeypatch.setattr(collectors_module, "_MAX_EXPORT_BYTES", 64 * 1024)
    too_deep = tmp_path / "deep.json"
    too_deep.write_text("{}", encoding="utf-8")

    def recurse(_encoded: bytes) -> object:
        raise RecursionError

    monkeypatch.setattr(collectors_module.json, "loads", recurse)
    with pytest.raises(ValueError, match="bounded JSON"):
        load_state_evidence_export(too_deep)


def test_private_pytest_plugin_collapses_reports_and_writes_stable_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "plugin.json"
    monkeypatch.setenv("PATHLAYER_PYTEST_EVIDENCE_OUTPUT", str(output))
    pytest_plugin.pytest_sessionstart(SimpleNamespace())
    pytest_plugin.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="test_b.py::test_b", when="call", passed=True, failed=False, skipped=False
        )
    )
    pytest_plugin.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="test_a.py::test_a", when="setup", passed=False, failed=False, skipped=True
        )
    )
    pytest_plugin.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="test_b.py::test_b", when="teardown", passed=False, failed=True, skipped=False
        )
    )
    pytest_plugin.pytest_sessionfinish(SimpleNamespace(), 1)
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "exit_code": 1,
        "outcomes": {
            "test_a.py::test_a": "failed",
            "test_b.py::test_b": "failed",
        },
    }


def test_private_pytest_plugin_without_output_is_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PATHLAYER_PYTEST_EVIDENCE_OUTPUT", raising=False)
    pytest_plugin.pytest_sessionstart(SimpleNamespace())
    pytest_plugin.pytest_sessionfinish(SimpleNamespace(), 0)


def test_private_pytest_plugin_bounds_captured_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "plugin.json"
    monkeypatch.setenv("PATHLAYER_PYTEST_EVIDENCE_OUTPUT", str(output))
    monkeypatch.setattr(pytest_plugin, "MAX_OUTCOMES", 1)
    pytest_plugin.pytest_sessionstart(SimpleNamespace())
    for nodeid in ("test_a.py::test_a", "test_b.py::test_b"):
        pytest_plugin.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid=nodeid,
                when="call",
                passed=True,
                failed=False,
                skipped=False,
            )
        )
    pytest_plugin.pytest_sessionfinish(SimpleNamespace(), 0)
    assert json.loads(output.read_text(encoding="utf-8"))["outcomes"] is None


def test_manually_corrupted_prior_is_rejected(tmp_path: Path) -> None:
    valid = _filesystem(tmp_path / "missing")
    corrupted = replace(valid, canonical_fingerprint={"wrong": True})
    with pytest.raises(ValueError, match="filesystem evidence"):
        _filesystem(tmp_path / "missing", phase="after", before=corrupted)


def test_dump_revalidates_manually_constructed_evidence(tmp_path: Path) -> None:
    valid = _filesystem(tmp_path / "missing")
    corrupted = StateEvidence(**{**asdict(valid), "phase": "later"})
    with pytest.raises(ValueError, match="invalid StateEvidence"):
        dump_state_evidence_export((corrupted,), tmp_path / "invalid.json")
