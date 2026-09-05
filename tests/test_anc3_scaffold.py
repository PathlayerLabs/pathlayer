"""ANC-3: deterministic progress-proof scaffolding."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pathlayer.detectors.proof import validate_proof_spec
from pathlayer.detectors.proof_producer import build_producer, load_proof_spec
from pathlayer.proof_scaffold import (
    ScaffoldError,
    main,
    render_progress_proof_scaffold,
    write_progress_proof_scaffold,
)


def _render(**overrides: object):  # type: ignore[no-untyped-def]
    inputs: dict[str, object] = {
        "task_id": "task-1",
        "name": "Accepted deliverables",
        "objective": "Ship a reviewed report.",
        "checkpoints": ["reviewed", "drafted", "published"],
    }
    inputs.update(overrides)
    return render_progress_proof_scaffold(**inputs)  # type: ignore[arg-type]


def test_rendered_manifest_is_canonical_and_valid_without_edits() -> None:
    bundle = _render()
    raw = json.loads(bundle.manifest_json)

    assert validate_proof_spec(raw) == bundle.proof_spec
    assert raw == {
        "kind": "checkpoint_set",
        "proof_id": "Accepted deliverables",
        "required": ["drafted", "published", "reviewed"],
        "schema_version": 1,
        "scope_id": "task-1",
        "task_id": "task-1",
    }
    assert bundle.manifest_json.endswith("\n")


def test_render_is_byte_deterministic_and_checkpoint_order_independent() -> None:
    first = _render()
    second = _render(checkpoints=["published", "reviewed", "drafted"])

    assert first == second


def test_explicit_identity_overrides_defaults() -> None:
    bundle = _render(proof_id="acceptance-v2", scope_id="customer-a")

    assert bundle.proof_spec.proof_id == "acceptance-v2"
    assert bundle.proof_spec.scope_id == "customer-a"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("objective", ""),
        ("task_id", ""),
        ("checkpoints", []),
        ("checkpoints", "abc"),
        ("checkpoints", ["same", "same"]),
        ("checkpoints", [f"checkpoint-{index}" for index in range(257)]),
    ],
)
def test_invalid_inputs_are_rejected_by_scaffolder(field: str, value: object) -> None:
    with pytest.raises(ScaffoldError):
        _render(**{field: value})


def test_special_characters_cannot_break_generated_python() -> None:
    bundle = _render(
        name='Review "quoted" output',
        objective='Line one\nLine two with """ and \\ escapes',
        checkpoints=['schema-"valid"', "accepted/équipe"],
    )

    compile(bundle.probe_python, "probe.py", "exec")
    assert 'schema-\\"valid\\"' in bundle.probe_python
    assert "Line one" in bundle.readme_markdown
    assert "> Line two" in bundle.readme_markdown


def test_write_round_trips_through_anc1_loader(tmp_path: Path) -> None:
    paths = write_progress_proof_scaffold(
        output_dir=tmp_path / "proof",
        task_id="task-1",
        name="Accepted deliverables",
        objective="Ship a reviewed report.",
        checkpoints=["reviewed", "drafted"],
    )

    assert load_proof_spec(paths.manifest).required == ("drafted", "reviewed")
    assert paths.probe.read_text(encoding="utf-8").endswith("\n")
    assert "NotImplementedError" in paths.probe.read_text(encoding="utf-8")
    readme = paths.readme.read_text(encoding="utf-8")
    assert "Ship a reviewed report." in readme
    assert "observed agent's write boundary" in readme


def test_unimplemented_probe_is_fail_soft_for_real_producer(tmp_path: Path) -> None:
    paths = write_progress_proof_scaffold(
        output_dir=tmp_path / "proof",
        task_id="task-1",
        name="Acceptance",
        objective="Verify the deliverable.",
        checkpoints=["done"],
    )
    producer = build_producer(
        str(paths.manifest), (sys.executable, str(paths.probe)), timeout_seconds=5.0
    )

    assert producer.probe() is None


def test_implemented_generated_probe_emits_anc2_wire_shape(tmp_path: Path) -> None:
    paths = write_progress_proof_scaffold(
        output_dir=tmp_path / "proof",
        task_id="task-1",
        name="Acceptance",
        objective="Verify the deliverable.",
        checkpoints=["schema_valid", "accepted"],
    )
    source = paths.probe.read_text(encoding="utf-8")
    start = source.index("def collect_checkpoints()")
    end = source.index("\n\ndef main()", start)
    implementation = """def collect_checkpoints() -> dict[str, bool]:
    return {"schema_valid": True, "accepted": False}
"""
    paths.probe.write_text(source[:start] + implementation + source[end:], encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(paths.probe)], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {
        "checkpoints": {"accepted": False, "schema_valid": True}
    }
    producer = build_producer(
        str(paths.manifest), (sys.executable, str(paths.probe)), timeout_seconds=5.0
    )
    produced = producer.probe()
    assert produced is not None
    _, proof = produced
    assert proof["checkpoints"] == {"accepted": False, "schema_valid": True}


def test_existing_target_refuses_all_writes(tmp_path: Path) -> None:
    output = tmp_path / "proof"
    output.mkdir()
    existing = output / "probe.py"
    existing.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(ScaffoldError, match="refusing to overwrite"):
        write_progress_proof_scaffold(
            output_dir=output,
            task_id="task-1",
            name="Acceptance",
            objective="Verify the deliverable.",
            checkpoints=["done"],
        )

    assert existing.read_text(encoding="utf-8") == "operator-owned\n"
    assert not (output / "proof_spec.json").exists()
    assert not (output / "README.md").exists()


def test_mid_install_failure_rolls_back_whole_new_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "proof"
    original_open = Path.open

    def flaky_open(path: Path, mode: str = "r", *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if path.name == "probe.py" and mode == "xb":
            raise OSError("simulated second-file failure")
        return original_open(path, mode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", flaky_open)
    with pytest.raises(ScaffoldError, match="could not write"):
        write_progress_proof_scaffold(
            output_dir=output,
            task_id="task-1",
            name="Acceptance",
            objective="Verify the deliverable.",
            checkpoints=["done"],
        )

    assert not output.exists()


def test_cli_writes_bundle_and_reports_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "proof"
    code = main(
        [
            "--task-id",
            "task-1",
            "--name",
            "Acceptance",
            "--objective",
            "Verify the deliverable.",
            "--checkpoint",
            "done",
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    assert (output / "proof_spec.json").exists()
    assert (output / "probe.py").exists()
    assert (output / "README.md").exists()
    assert "created" in capsys.readouterr().out
