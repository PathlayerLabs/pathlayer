"""ANC-4: declarative coding contracts compiled to the existing ANC artifacts.

Build-time half only (loading, validation, compilation, rendering, writing and
the CLI). The capture-time evaluator the generated probe calls is covered by
``tests/test_anc4_contract_probe.py``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pathlayer.coding_contracts import (
    ArtifactProduced,
    CodingContract,
    CodingContractError,
    CommandSucceeds,
    FileChanges,
    FileExists,
    TestPasses,
    TestsImprove,
    coding_contract_from_mapping,
    compile_coding_contract,
    load_coding_contract,
    main,
    write_coding_contract,
)
from pathlayer.config import ConfigError, PathlayerConfig, build_pipeline
from pathlayer.detectors.no_progress import NoProgressDetector
from pathlayer.detectors.proof import ProofProgressAnchor, compute_spec_hash
from pathlayer.detectors.proof_producer import build_producer, load_proof_spec
from pathlayer.proof_scaffold import ScaffoldError

_TICKET_YAML = """\
task: fix-login
objective: Make the login flow pass its test.
workspace: ws
progress:
  - type: test_passes
    test: tests/test_login.py::test_login
  - type: file_changes
    path: src/auth.py
"""


def _workspace(root: Path) -> Path:
    workspace = root / "ws"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "src" / "auth.py").write_text("def login():\n    return False\n", "utf-8")
    (workspace / "tests" / "test_login.py").write_text(
        "from src.auth import login\n\ndef test_login():\n    assert login()\n", "utf-8"
    )
    return workspace


def _contract(workspace: Path, *progress: Any, **overrides: Any) -> CodingContract:
    fields: dict[str, Any] = {
        "task": "fix-login",
        "objective": "Make the login flow pass its test.",
        "workspace": workspace,
        "progress": progress or (TestPasses("tests/test_login.py::test_login"),),
    }
    fields.update(overrides)
    return CodingContract(**fields)


def _mapping(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "task": "fix-login",
        "objective": "Make the login flow pass its test.",
        "workspace": "ws",
        "progress": [{"type": "test_passes", "test": "tests/test_login.py::test_login"}],
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- #
# The ticket's own example, three equivalent surfaces
# --------------------------------------------------------------------------- #


def test_ticket_yaml_example_compiles_to_anc_checkpoint_set(tmp_path: Path) -> None:
    _workspace(tmp_path)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_TICKET_YAML, encoding="utf-8")

    compiled = compile_coding_contract(load_coding_contract(contract_path))

    spec = compiled.bundle.proof_spec
    assert spec.kind == "checkpoint_set"
    assert (spec.task_id, spec.proof_id, spec.scope_id) == ("fix-login",) * 3
    assert spec.required == ("file_changes:src/auth.py", "test:tests/test_login.py::test_login")
    manifest = tmp_path / "proof_spec.json"
    manifest.write_text(compiled.bundle.manifest_json, encoding="utf-8")
    assert load_proof_spec(manifest) == spec
    assert compiled.payload["python"] is None  # the interpreter running the probe


def test_python_api_and_yaml_produce_byte_identical_bundles(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_TICKET_YAML, encoding="utf-8")

    from_yaml = compile_coding_contract(load_coding_contract(contract_path))
    from_python = compile_coding_contract(
        _contract(
            workspace,
            TestPasses("tests/test_login.py::test_login"),
            FileChanges("src/auth.py"),
        )
    )

    assert from_python.bundle == from_yaml.bundle
    assert from_python.payload == from_yaml.payload


def test_json_contract_is_equivalent_to_yaml(tmp_path: Path) -> None:
    _workspace(tmp_path)
    yaml_path = tmp_path / "contract.yaml"
    yaml_path.write_text(_TICKET_YAML, encoding="utf-8")
    json_path = tmp_path / "contract.json"
    json_path.write_text(
        json.dumps(
            {
                "task": "fix-login",
                "objective": "Make the login flow pass its test.",
                "workspace": "ws",
                "progress": [
                    {"type": "test_passes", "test": "tests/test_login.py::test_login"},
                    {"type": "file_changes", "path": "src/auth.py"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert compile_coding_contract(load_coding_contract(json_path)).bundle == (
        compile_coding_contract(load_coding_contract(yaml_path)).bundle
    )


def test_compilation_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    primitives = (
        FileExists("docs/NOTES.md"),
        FileChanges("src/auth.py"),
        TestPasses("tests/test_login.py::test_login"),
        CommandSucceeds("typecheck", (sys.executable, "-c", "pass")),
        ArtifactProduced("dist/report.json", min_bytes=2),
    )

    first = compile_coding_contract(_contract(workspace, *primitives))
    second = compile_coding_contract(_contract(workspace, *reversed(primitives)))

    assert first == second


def test_default_workspace_is_the_contract_files_directory(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    contract_path = workspace / "contract.yaml"
    contract_path.write_text(
        textwrap.dedent(
            """\
            task: fix-login
            objective: Make the login flow pass its test.
            progress:
              - type: test_passes
                test: tests/test_login.py::test_login
            """
        ),
        encoding="utf-8",
    )

    compiled = compile_coding_contract(load_coding_contract(contract_path))

    assert compiled.payload["workspace"] == str(workspace.resolve())


def test_identity_overrides_are_honoured(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    compiled = compile_coding_contract(
        _contract(workspace, proof_id="login-acceptance-v2", scope_id="customer-a")
    )

    spec = compiled.bundle.proof_spec
    assert (spec.task_id, spec.proof_id, spec.scope_id) == (
        "fix-login",
        "login-acceptance-v2",
        "customer-a",
    )


# --------------------------------------------------------------------------- #
# Malformed contracts fail at build time, never during a run
# --------------------------------------------------------------------------- #


def test_contract_error_is_a_config_error() -> None:
    assert issubclass(CodingContractError, ConfigError)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"taks": "typo"}, "unknown key"),
        ({"task": ""}, "task"),
        ({"task": 3}, "task"),
        ({"objective": ""}, "objective"),
        ({"progress": []}, "at least one"),
        ({"progress": "test_passes"}, "progress"),
        ({"progress": ["test_passes"]}, "mapping"),
        ({"progress": [{"path": "src/auth.py"}]}, "type"),
        ({"progress": [{"type": "file_is_nice", "path": "x"}]}, "unknown progress type"),
        ({"progress": [{"type": "file_exists", "pth": "x"}]}, "unknown key"),
        ({"progress": [{"type": "file_exists"}]}, "path"),
        ({"progress": [{"type": "test_passes"}]}, "test"),
        ({"progress": [{"type": "tests_improve"}]}, "exactly one of"),
        (
            {
                "progress": [
                    {"type": "tests_improve", "tests": ["tests/test_login.py::t"], "path": "tests"}
                ]
            },
            "exactly one of",
        ),
        ({"progress": [{"type": "tests_improve", "tests": []}]}, "at least one"),
        ({"progress": [{"type": "command_succeeds", "command": ["true"]}]}, "id"),
        ({"progress": [{"type": "command_succeeds", "id": "build"}]}, "command"),
        ({"progress": [{"type": "artifact_produced", "path": "x", "min_bytes": True}]}, "int"),
        ({"timeout_seconds": "30"}, "number"),
        ({"workspace": 7}, "string"),
        ({"proof_id": 3}, "proof_id"),
        ({"python": 3}, "python"),
        ({"progress": [{"type": "tests_improve", "path": 3}]}, "expected string"),
        ({"progress": [{"type": "tests_improve", "tests": "a.py::t"}]}, "list of strings"),
        ({"progress": [{"type": "artifact_produced", "path": "x", "sha256": 12}]}, "sha256"),
        ({"progress": [{"type": "command_succeeds", "id": "b", "command": "make"}]}, "list of"),
    ],
)
def test_malformed_mapping_is_rejected_when_loaded(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    _workspace(tmp_path)
    with pytest.raises(CodingContractError, match=message):
        compile_coding_contract(
            coding_contract_from_mapping(_mapping(**overrides), base_dir=tmp_path)
        )


@pytest.mark.parametrize("root", [[], "task: x", 3, None])
def test_contract_root_must_be_a_mapping(tmp_path: Path, root: object) -> None:
    with pytest.raises(CodingContractError, match="mapping"):
        coding_contract_from_mapping(root, base_dir=tmp_path)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("primitive", "message"),
    [
        (FileExists(""), "empty"),
        (FileExists("/etc/passwd"), "relative"),
        (FileExists("../secret.txt"), "outside the workspace"),
        (FileExists("src/../../secret.txt"), "outside the workspace"),
        (FileExists("src/\x00auth.py"), "NUL"),
        (FileExists(7), "string"),  # type: ignore[arg-type]
        (TestPasses("tests/test_login.py"), "exact pytest nodeid"),
        (TestPasses("-p no:cov::x"), "must not start with '-'"),
        (TestPasses("../tests/test_login.py::test_login"), "outside the workspace"),
        (TestPasses("tests/test_login.py::test\x00login"), "NUL"),
        (TestsImprove(tests=("tests/test_login.py::a", "tests/test_login.py::a")), "duplicate"),
        (TestsImprove(tests=("tests/test_login.py::a",), path="tests"), "exactly one of"),
        (CommandSucceeds("has space", ("true",)), "id"),
        (CommandSucceeds("", ("true",)), "id"),
        (CommandSucceeds("build", ()), "command"),
        (CommandSucceeds("build", ("",)), "command"),
        (CommandSucceeds("build", ("echo", "a\x00b")), "NUL"),
        (ArtifactProduced("dist/x", min_bytes=-1), "min_bytes"),
        (ArtifactProduced("dist/x", min_bytes=1.5), "min_bytes"),  # type: ignore[arg-type]
        (ArtifactProduced("dist/x", sha256="nope"), "sha256"),
        (ArtifactProduced("dist/x", sha256="A" * 63), "sha256"),
    ],
)
def test_malformed_primitive_is_rejected_at_compile_time(
    tmp_path: Path, primitive: Any, message: str
) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(CodingContractError, match=message):
        compile_coding_contract(
            _contract(workspace, TestPasses("tests/test_login.py::test_login"), primitive)
        )


@pytest.mark.parametrize("timeout", [0, -1.0, math.nan, math.inf, True, 3601])
def test_invalid_probe_budget_is_rejected(tmp_path: Path, timeout: Any) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(CodingContractError, match="timeout_seconds"):
        compile_coding_contract(_contract(workspace, timeout_seconds=timeout))


@pytest.mark.parametrize(
    "progress",
    [
        (FileExists("src/auth.py"), FileExists("src/auth.py")),
        (FileExists("src/auth.py"), FileExists("./src//auth.py")),
        (
            TestPasses("tests/test_login.py::test_login"),
            TestsImprove(tests=("tests/test_login.py::test_login",)),
        ),
        (
            CommandSucceeds("build", ("true",)),
            CommandSucceeds("build", ("false",)),
        ),
    ],
)
def test_duplicate_checkpoints_are_rejected(tmp_path: Path, progress: tuple[Any, ...]) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(CodingContractError, match="duplicate"):
        compile_coding_contract(_contract(workspace, *progress))


def test_effect_only_contract_is_rejected(tmp_path: Path) -> None:
    """An observed effect is never the objective.

    Once every checkpoint is true the anchor is satisfied and NoProgressDetector
    goes silent for the rest of the run. A contract made only of effects would
    therefore disable no-progress detection at the agent's first edit.
    """
    workspace = _workspace(tmp_path)
    with pytest.raises(CodingContractError, match="objective"):
        compile_coding_contract(
            _contract(workspace, FileChanges("src/auth.py"), FileChanges("tests/test_login.py"))
        )


def test_checkpoint_cap_is_enforced_with_a_clear_message(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    tests = tuple(f"tests/test_login.py::test_{index}" for index in range(257))
    with pytest.raises(CodingContractError, match="256"):
        compile_coding_contract(_contract(workspace, TestsImprove(tests=tests)))


def test_workspace_must_be_an_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(CodingContractError, match="workspace"):
        compile_coding_contract(_contract(tmp_path / "missing"))
    regular = tmp_path / "file.txt"
    regular.write_text("x", encoding="utf-8")
    with pytest.raises(CodingContractError, match="workspace"):
        compile_coding_contract(_contract(regular))


def test_file_changes_baseline_must_be_a_regular_file_or_missing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(CodingContractError, match="baseline"):
        compile_coding_contract(
            _contract(workspace, TestPasses("tests/test_login.py::test_login"), FileChanges("src"))
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_symlink_escaping_the_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create symlinks here")
    with pytest.raises(CodingContractError, match="outside the workspace"):
        compile_coding_contract(
            _contract(
                workspace, TestPasses("tests/test_login.py::test_login"), FileExists("link/x.txt")
            )
        )


def test_loader_reports_unreadable_or_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(CodingContractError, match="not found"):
        load_coding_contract(tmp_path / "missing.yaml")
    toml = tmp_path / "contract.toml"
    toml.write_text("task = 'x'", encoding="utf-8")
    with pytest.raises(CodingContractError, match="unsupported"):
        load_coding_contract(toml)
    broken = tmp_path / "contract.yaml"
    broken.write_text("task: [unclosed", encoding="utf-8")
    with pytest.raises(CodingContractError, match="YAML"):
        load_coding_contract(broken)
    binary = tmp_path / "contract.json"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(CodingContractError, match="cannot read"):
        load_coding_contract(binary)


# --------------------------------------------------------------------------- #
# Vocabulary semantics that live at compile time
# --------------------------------------------------------------------------- #


def test_primitive_classes_are_not_collected_as_tests() -> None:
    """Importing the vocabulary into a user's test module must stay silent (P4)."""
    assert TestPasses.__test__ is False
    assert TestsImprove.__test__ is False


def test_checkpoint_ids_are_workspace_relative_and_command_ids_explicit(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    compiled = compile_coding_contract(
        _contract(
            workspace,
            TestPasses("tests/test_login.py::test_login"),
            FileExists("./docs//NOTES.md"),
            CommandSucceeds("typecheck", (sys.executable, "-m", "mypy", "src")),
            CommandSucceeds("typecheck-strict", (sys.executable, "-m", "mypy", "--strict")),
        )
    )

    required = compiled.bundle.proof_spec.required
    assert "file_exists:docs/NOTES.md" in required
    assert {"command:typecheck", "command:typecheck-strict"} <= set(required)
    assert not any(str(workspace) in checkpoint for checkpoint in required)
    assert not any(sys.executable in checkpoint for checkpoint in required)


def test_long_identifiers_are_shortened_deterministically(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    prefix = "tests/test_login.py::test_login[" + "x" * 300
    first = f"{prefix}-a]"
    second = f"{prefix}-b]"

    compiled = compile_coding_contract(_contract(workspace, TestsImprove(tests=(first, second))))
    again = compile_coding_contract(_contract(workspace, TestsImprove(tests=(first, second))))

    required = compiled.bundle.proof_spec.required
    assert len(required) == 2
    assert all(len(checkpoint) <= 256 and checkpoint.startswith("test:") for checkpoint in required)
    assert required == again.bundle.proof_spec.required
    # The payload keeps the exact nodeid the probe must run.
    assert sorted(compiled.payload["tests"]) == [first, second]
    assert set(compiled.payload["tests"].values()) == set(required)


def test_file_changes_baseline_is_captured_at_compile_time(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    compiled = compile_coding_contract(
        _contract(
            workspace,
            TestPasses("tests/test_login.py::test_login"),
            FileChanges("src/auth.py"),
            FileChanges("src/new_module.py"),
        )
    )

    baselines = {
        check["path"]: check["baseline"]
        for check in compiled.payload["files"]
        if check["kind"] == "file_changes"
    }
    assert baselines["src/new_module.py"] == "missing"
    assert baselines["src/auth.py"].startswith("sha256:")


def test_tests_improve_path_selection_expands_at_compile_time(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "tests" / "test_calc.py").write_text(
        "def test_add():\n    pass\n\ndef test_sub():\n    pass\n\n"
        "class TestDiv:\n    def test_div(self):\n        pass\n",
        encoding="utf-8",
    )

    compiled = compile_coding_contract(
        _contract(workspace, TestsImprove(path="tests/test_calc.py"))
    )

    assert compiled.bundle.proof_spec.required == (
        "test:tests/test_calc.py::TestDiv::test_div",
        "test:tests/test_calc.py::test_add",
        "test:tests/test_calc.py::test_sub",
    )
    by_directory = compile_coding_contract(_contract(workspace, TestsImprove(path="tests")))
    assert len(by_directory.bundle.proof_spec.required) == 4


def test_tests_improve_path_selection_fails_fast_when_uncollectable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "tests" / "test_broken.py").write_text("import not_a_module\n", "utf-8")
    (workspace / "empty").mkdir()

    with pytest.raises(CodingContractError, match="collect"):
        compile_coding_contract(_contract(workspace, TestsImprove(path="tests/test_broken.py")))
    with pytest.raises(CodingContractError, match="no test"):
        compile_coding_contract(_contract(workspace, TestsImprove(path="empty")))
    with pytest.raises(CodingContractError, match="does not exist"):
        compile_coding_contract(_contract(workspace, TestsImprove(path="tests/missing.py")))


def test_compiled_contract_carries_a_safe_producer_timeout(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    compiled = compile_coding_contract(_contract(workspace, timeout_seconds=12))

    assert compiled.payload["timeout_seconds"] == 12.0
    assert compiled.producer_timeout_seconds > 12.0


# --------------------------------------------------------------------------- #
# The project's own interpreter for test primitives
# --------------------------------------------------------------------------- #

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX wrapper interpreters")


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _project_python(path: Path, *, with_pytest: bool = True) -> Path:
    """An interpreter like a project venv's: it can run pytest, not import pathlayer."""
    site = Path(pytest.__file__).resolve().parent.parent
    pythonpath = f'PYTHONPATH="{site}${{PYTHONPATH:+:$PYTHONPATH}}" ' if with_pytest else ""
    return _executable(path, f'#!/bin/sh\n{pythonpath}exec "{sys.executable}" -S "$@"\n')


@posix_only
def test_python_is_resolved_in_the_workspace_without_following_symlinks(tmp_path: Path) -> None:
    """A venv's bin/python is usually a symlink: invoking its target bypasses the venv."""
    workspace = _workspace(tmp_path)
    _project_python(tmp_path / "real-env" / "bin" / "python")
    (workspace / ".venv").symlink_to(tmp_path / "real-env", target_is_directory=True)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_TICKET_YAML + "python: .venv/bin/python\n", encoding="utf-8")

    compiled = compile_coding_contract(load_coding_contract(contract_path))

    assert compiled.payload["python"] == str(workspace.resolve() / ".venv" / "bin" / "python")
    assert compiled.payload["python"] in compiled.bundle.readme_markdown


@posix_only
def test_absolute_python_outside_the_workspace_is_accepted(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    python = _project_python(tmp_path / "envs" / "project" / "bin" / "python")

    compiled = compile_coding_contract(_contract(workspace, python=str(python)))

    assert compiled.payload["python"] == str(python)


def test_missing_python_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(CodingContractError, match=r"python.*not an executable file"):
        compile_coding_contract(_contract(workspace, python=".venv/bin/python"))


@posix_only
def test_python_that_cannot_run_pytest_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    no_pytest = _project_python(tmp_path / "bare" / "python", with_pytest=False)
    not_python = _executable(tmp_path / "fake" / "python", "#!/bin/sh\nexit 1\n")
    too_old = _executable(tmp_path / "old" / "python", "#!/bin/sh\nexit 3\n")

    with pytest.raises(CodingContractError, match="cannot run pytest"):
        compile_coding_contract(_contract(workspace, python=str(no_pytest)))
    with pytest.raises(CodingContractError, match="cannot run Python"):
        compile_coding_contract(_contract(workspace, python=str(not_python)))
    with pytest.raises(CodingContractError, match=r"older than Python 3\.8"):
        compile_coding_contract(_contract(workspace, python=str(too_old)))


@posix_only
def test_python_is_checked_before_expanding_a_test_path(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    no_pytest = _project_python(tmp_path / "bare" / "python", with_pytest=False)

    with pytest.raises(CodingContractError, match="cannot run pytest"):
        compile_coding_contract(
            _contract(workspace, TestsImprove(path="tests"), python=str(no_pytest))
        )


@posix_only
def test_python_without_test_primitives_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    python = _project_python(tmp_path / "env" / "bin" / "python")

    with pytest.raises(CodingContractError, match="only"):
        compile_coding_contract(_contract(workspace, FileExists("out.txt"), python=str(python)))


def test_readme_names_the_default_test_interpreter(tmp_path: Path) -> None:
    compiled = compile_coding_contract(_contract(_workspace(tmp_path)))

    readme = " ".join(compiled.bundle.readme_markdown.split())
    assert "the interpreter running this probe" in readme
    assert "`python`" in readme


# --------------------------------------------------------------------------- #
# Rendered artifacts
# --------------------------------------------------------------------------- #


def test_generated_probe_is_valid_python_whatever_the_text(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / 'odd "dir"').mkdir()

    compiled = compile_coding_contract(
        _contract(
            workspace,
            TestPasses("tests/test_login.py::test_login"),
            FileExists('odd "dir"/é x.txt'),
            task='fix "login" \\ now',
            objective='Line one\nLine two with """ and \\ escapes, \u2028 and \u2029',
        )
    )

    code = compile(compiled.bundle.probe_python, "probe.py", "exec")
    namespace: dict[str, Any] = {"__name__": "not_main"}
    exec(code, namespace)
    assert namespace["COMPILED_CONTRACT"] == compiled.payload


def test_readme_distinguishes_effect_progress_and_objective(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    compiled = compile_coding_contract(
        _contract(
            workspace,
            TestPasses("tests/test_login.py::test_login"),
            FileChanges("src/auth.py"),
        )
    )

    readme = " ".join(compiled.bundle.readme_markdown.split())
    lowered = readme.lower()
    assert "observed effect" in lowered
    assert "progress toward the objective" in lowered
    assert "objective reached" in lowered
    assert "never reaches the objective on its own" in lowered
    for checkpoint in compiled.bundle.proof_spec.required:
        assert checkpoint in readme


def test_readme_states_the_trust_boundary_honestly(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    compiled = compile_coding_contract(_contract(workspace))

    readme = " ".join(compiled.bundle.readme_markdown.split())
    assert "outside the observed agent's write boundary" in readme
    # P2: the judge executes agent-writable test code.
    assert "conftest.py" in readme
    assert "can forge" in readme
    # P5: no atomicity claim for a three-file install.
    assert "atomic" not in readme.lower()
    assert "every tool call" in readme


def test_write_installs_a_bundle_the_anc2_producer_accepts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    contract = _contract(workspace)

    installed = write_coding_contract(contract, output_dir=tmp_path / "proof")

    assert installed.paths.manifest.is_file()
    assert installed.paths.probe.is_file()
    assert installed.paths.readme.is_file()
    producer = build_producer(
        str(installed.paths.manifest),
        (sys.executable, str(installed.paths.probe)),
        installed.compiled.producer_timeout_seconds,
    )
    assert producer.spec == installed.compiled.bundle.proof_spec
    assert producer.spec_hash == compute_spec_hash(producer.spec)
    snippet = installed.progress_proof_config()
    assert snippet == {
        "enabled": True,
        "manifest_path": str(installed.paths.manifest.resolve()),
        "probe_command": [sys.executable, str(installed.paths.probe.resolve())],
        "timeout_seconds": installed.compiled.producer_timeout_seconds,
    }
    pipeline = build_pipeline(
        PathlayerConfig.from_mapping({"detectors": {"progress_proof": snippet}})
    )
    no_progress = next(d for d in pipeline._detectors if isinstance(d, NoProgressDetector))
    assert any(isinstance(anchor, ProofProgressAnchor) for anchor in no_progress._anchors)


def test_write_refuses_to_overwrite_any_existing_file(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    output = tmp_path / "proof"
    output.mkdir()
    (output / "README.md").write_text("operator notes\n", encoding="utf-8")

    with pytest.raises(ScaffoldError, match="refusing to overwrite"):
        write_coding_contract(_contract(workspace), output_dir=output)

    assert (output / "README.md").read_text(encoding="utf-8") == "operator notes\n"
    assert not (output / "probe.py").exists()


def test_zero_config_pipeline_is_unchanged() -> None:
    pipeline = build_pipeline(PathlayerConfig())
    no_progress = next(d for d in pipeline._detectors if isinstance(d, NoProgressDetector))
    assert not any(isinstance(anchor, ProofProgressAnchor) for anchor in no_progress._anchors)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_writes_bundle_and_prints_ready_to_paste_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _workspace(tmp_path)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_TICKET_YAML, encoding="utf-8")
    output = tmp_path / "proof"

    code = main([str(contract_path), "--output-dir", str(output), "--no-check"])

    assert code == 0
    out = capsys.readouterr().out
    assert f"created {output / 'probe.py'}" in out
    snippet = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert snippet["detectors"]["progress_proof"]["manifest_path"] == str(
        (output / "proof_spec.json").resolve()
    )
    assert snippet["detectors"]["progress_proof"]["probe_command"][0] == sys.executable


def test_cli_check_runs_the_probe_once_and_prints_the_current_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _workspace(tmp_path)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_TICKET_YAML, encoding="utf-8")

    code = main([str(contract_path), "--output-dir", str(tmp_path / "proof")])

    assert code == 0
    out = capsys.readouterr().out
    assert "file_changes:src/auth.py: false" in out
    assert "test:tests/test_login.py::test_login: false" in out


def test_cli_check_names_what_it_could_not_observe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _workspace(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "task": "fix-login",
                "objective": "Make the login flow pass its test.",
                "workspace": "ws",
                "progress": [
                    {"type": "command_succeeds", "id": "lint", "command": ["no-such-linter-xyz"]}
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main([str(contract_path), "--output-dir", str(tmp_path / "proof")]) == 0

    out = capsys.readouterr().out
    assert "command:lint: unobserved (command could not be launched)" in out
    assert "exited with status 4" in out


def test_cli_rejects_a_malformed_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("task: x\nobjective: y\nprogress: []\n", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        main([str(contract_path), "--output-dir", str(tmp_path / "proof")])

    assert raised.value.code == 2
    assert "at least one" in capsys.readouterr().err
    assert not (tmp_path / "proof").exists()
