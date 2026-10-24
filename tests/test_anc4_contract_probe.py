"""ANC-4: the capture-time evaluator behind every generated coding-contract probe.

Each check reports a fact (``true``/``false``) or nothing: a resource the probe
cannot observe is omitted, with a fixed-vocabulary reason on stderr, so it can
neither create nor erase progress. Most tests call the evaluator in-process
(so coverage sees it); a few run the generated ``probe.py`` exactly as the
ANC-2 producer does.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from pathlayer import coding_contract_probe
from pathlayer.coding_contract_probe import (
    EXIT_INVALID_CONTRACT,
    EXIT_NOTHING_OBSERVED,
    EXIT_OBSERVED,
    EXIT_PATHLAYER_UNAVAILABLE,
    REASON_BUDGET,
    REASON_COMMAND_LAUNCH,
    REASON_COMMAND_TIMEOUT,
    REASON_NOT_REGULAR,
    REASON_OUTSIDE,
    REASON_PYTEST_TIMEOUT,
    REASON_PYTEST_UNREADABLE,
    REASON_TEST_INTERPRETER,
    REASON_TEST_UNCOLLECTABLE,
    REASON_UNEXPECTED,
    REASON_WORKSPACE,
    check_test_interpreter,
    run_compiled_contract,
)
from pathlayer.coding_contracts import (
    ArtifactProduced,
    CodingContract,
    CommandSucceeds,
    FileChanges,
    FileExists,
    TestPasses,
    TestsImprove,
    compile_coding_contract,
    write_coding_contract,
)
from pathlayer.detectors.proof import validate_progress_proof
from pathlayer.detectors.proof_producer import build_producer

_ANCHOR = TestPasses("tests/test_anchor.py::test_anchor")


def _workspace(root: Path) -> Path:
    workspace = root / "ws"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "tests" / "test_anchor.py").write_text(
        "def test_anchor():\n    pass\n", encoding="utf-8"
    )
    return workspace


def _payload(workspace: Path, *progress: Any, timeout: float = 30.0) -> dict[str, Any]:
    contract = CodingContract(
        task="probe-test",
        objective="Exercise the evaluator.",
        workspace=workspace,
        progress=progress,
        timeout_seconds=timeout,
    )
    return dict(compile_coding_contract(contract).payload)


def _run(
    payload: object, *, clock: Callable[[], float] = time.monotonic
) -> tuple[int, dict[str, bool] | None, dict[str, str]]:
    """Run in-process; return (status, checkpoints or None, unobserved reasons)."""
    out, err = io.StringIO(), io.StringIO()
    status = run_compiled_contract(payload, stdout=out, stderr=err, clock=clock)
    checkpoints = json.loads(out.getvalue())["checkpoints"] if out.getvalue() else None
    reasons: dict[str, str] = {}
    for line in err.getvalue().splitlines():
        if line.startswith("unobserved "):
            checkpoint, _, reason = line[len("unobserved ") :].rpartition(": ")
            reasons[checkpoint] = reason
    return status, checkpoints, reasons


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def test_file_exists_reports_missing_then_created(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(workspace, FileExists("out/report.txt"), FileChanges("tests/test_anchor.py"))

    assert _run(payload)[1] == {
        "file_changes:tests/test_anchor.py": False,
        "file_exists:out/report.txt": False,
    }
    (workspace / "out").mkdir()
    (workspace / "out" / "report.txt").write_text("", encoding="utf-8")
    assert _run(payload)[1] == {
        "file_changes:tests/test_anchor.py": False,
        "file_exists:out/report.txt": True,
    }


def test_directory_where_a_file_is_expected_is_omitted_not_false(tmp_path: Path) -> None:
    """P1: one unobservable path costs its own checkpoint, never the whole tick."""
    workspace = _workspace(tmp_path)
    (workspace / "notes.md").write_text("draft", encoding="utf-8")
    payload = _payload(
        workspace, FileExists("out"), FileChanges("notes.md"), ArtifactProduced("dist")
    )
    (workspace / "out").mkdir()
    (workspace / "dist").mkdir()
    (workspace / "notes.md").write_text("edited", encoding="utf-8")

    status, checkpoints, reasons = _run(payload)

    assert status == EXIT_OBSERVED
    assert checkpoints == {"file_changes:notes.md": True}
    assert reasons == {
        "artifact_produced:dist": REASON_NOT_REGULAR,
        "file_exists:out": REASON_NOT_REGULAR,
    }


def test_file_changes_is_an_effect_relative_to_the_compile_time_baseline(tmp_path: Path) -> None:
    """P6: restoring the baseline makes the effect false again (never a new record)."""
    workspace = _workspace(tmp_path)
    source = workspace / "auth.py"
    source.write_text("before\n", encoding="utf-8")
    payload = _payload(workspace, _ANCHOR, FileChanges("auth.py"), FileChanges("new.py"))

    def effects() -> dict[str, bool]:
        checkpoints = _run(payload)[1]
        assert checkpoints is not None
        return {key: value for key, value in checkpoints.items() if key.startswith("file_")}

    assert effects() == {"file_changes:auth.py": False, "file_changes:new.py": False}
    source.write_text("after\n", encoding="utf-8")
    (workspace / "new.py").write_text("", encoding="utf-8")
    assert effects() == {"file_changes:auth.py": True, "file_changes:new.py": True}
    source.write_text("before\n", encoding="utf-8")
    assert effects()["file_changes:auth.py"] is False
    source.unlink()
    assert effects()["file_changes:auth.py"] is True


def test_file_changes_on_a_path_turned_directory_is_omitted(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "auth.py").write_text("before\n", encoding="utf-8")
    payload = _payload(workspace, FileExists("README.md"), FileChanges("auth.py"))
    (workspace / "auth.py").unlink()
    (workspace / "auth.py").mkdir()

    status, checkpoints, reasons = _run(payload)

    assert status == EXIT_OBSERVED
    assert checkpoints == {"file_exists:README.md": False}
    assert reasons == {"file_changes:auth.py": REASON_NOT_REGULAR}


def test_artifact_produced_applies_its_deterministic_validation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    good = "4" * 64
    digest = hashlib.sha256(b"expected\n").hexdigest()
    payload = _payload(
        workspace,
        ArtifactProduced("dist/a.bin"),
        ArtifactProduced("dist/b.bin", min_bytes=4),
        ArtifactProduced("dist/c.txt", sha256=digest.upper()),
        ArtifactProduced("dist/d.txt", sha256=good),
        ArtifactProduced("dist/e.txt", min_bytes=0),
    )
    (workspace / "dist").mkdir()
    assert _run(payload)[1] == dict.fromkeys(
        [
            f"artifact_produced:dist/{name}"
            for name in ("a.bin", "b.bin", "c.txt", "d.txt", "e.txt")
        ],
        False,
    )

    (workspace / "dist" / "a.bin").write_bytes(b"")
    (workspace / "dist" / "b.bin").write_bytes(b"abc")
    (workspace / "dist" / "c.txt").write_bytes(b"expected\n")
    (workspace / "dist" / "d.txt").write_bytes(b"something else\n")
    (workspace / "dist" / "e.txt").write_bytes(b"")
    assert _run(payload)[1] == {
        "artifact_produced:dist/a.bin": False,  # empty: below the default min_bytes=1
        "artifact_produced:dist/b.bin": False,  # 3 < 4 bytes
        "artifact_produced:dist/c.txt": True,  # SHA-256 matches (case-insensitive input)
        "artifact_produced:dist/d.txt": False,  # SHA-256 differs
        "artifact_produced:dist/e.txt": True,  # min_bytes=0 accepts an empty file
    }


@pytest.fixture
def symlinks_supported(tmp_path: Path) -> Iterator[None]:
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    probe.unlink()
    yield


@pytest.mark.usefixtures("symlinks_supported")
def test_symlinks_are_never_followed_to_a_verdict(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x", encoding="utf-8")
    (workspace / "real.txt").write_text("x", encoding="utf-8")
    payload = _payload(workspace, FileExists("link.txt"), FileExists("escape/secret.txt"), _ANCHOR)
    # Planted by the agent after the contract was compiled.
    (workspace / "link.txt").symlink_to(workspace / "real.txt")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)

    _, checkpoints, reasons = _run(payload)

    assert checkpoints == {"test:tests/test_anchor.py::test_anchor": True}
    assert reasons == {
        "file_exists:escape/secret.txt": REASON_OUTSIDE,
        "file_exists:link.txt": REASON_NOT_REGULAR,
    }


@pytest.mark.usefixtures("symlinks_supported")
def test_a_symlink_loop_is_unobservable_on_every_python(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    (workspace / "loop").symlink_to(workspace / "loop")

    with pytest.raises(coding_contract_probe.Unobservable) as raised:
        coding_contract_probe.contained_path(workspace, "loop/x.py")
    assert raised.value.reason == REASON_NOT_REGULAR


def test_missing_workspace_observes_nothing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(
        workspace, FileExists("x"), _ANCHOR, CommandSucceeds("ok", (sys.executable, "-V"))
    )
    shutil.rmtree(workspace)

    status, checkpoints, reasons = _run(payload)

    assert status == EXIT_NOTHING_OBSERVED
    assert checkpoints is None
    assert set(reasons.values()) == {REASON_WORKSPACE}
    assert len(reasons) == 3


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def test_command_exit_status_is_a_fact_and_runs_in_the_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    check_marker = "import os, sys; sys.exit(0 if os.path.exists('marker') else 3)"
    payload = _payload(
        workspace,
        CommandSucceeds("marker", (sys.executable, "-c", check_marker)),
        CommandSucceeds("fails", (sys.executable, "-c", "raise SystemExit(1)")),
    )

    assert _run(payload)[1] == {"command:fails": False, "command:marker": False}
    (workspace / "marker").write_text("", encoding="utf-8")
    assert _run(payload)[1] == {"command:fails": False, "command:marker": True}


def test_command_is_never_run_through_a_shell(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(
        workspace,
        CommandSucceeds("literal", (sys.executable, "-c", "pass", "; touch hacked", "$(touch x)")),
    )

    assert _run(payload)[1] == {"command:literal": True}
    assert not (workspace / "hacked").exists()
    assert not (workspace / "x").exists()


def test_unlaunchable_command_is_omitted(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(
        workspace, _ANCHOR, CommandSucceeds("missing", ("pathlayer-no-such-binary-xyz",))
    )

    _, checkpoints, reasons = _run(payload)

    assert checkpoints == {"test:tests/test_anchor.py::test_anchor": True}
    assert reasons == {"command:missing": REASON_COMMAND_LAUNCH}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
def test_timed_out_command_is_omitted_and_its_process_tree_killed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    grandchild = "import time; time.sleep(1.5); open('late', 'w').write('x')"
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
    )
    payload = _payload(
        workspace,
        FileExists("x"),
        CommandSucceeds("hangs", (sys.executable, "-c", parent)),
        timeout=0.8,
    )

    started = time.monotonic()
    _, checkpoints, reasons = _run(payload)
    elapsed = time.monotonic() - started

    assert checkpoints == {"file_exists:x": False}
    assert reasons == {"command:hangs": REASON_COMMAND_TIMEOUT}
    assert elapsed < 10
    time.sleep(2.0)
    assert not (workspace / "late").exists()


def test_command_output_never_leaks_into_probe_diagnostics(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    noisy = "import sys; sys.stderr.write('TOKEN=hunter2'); print('TOKEN=hunter2'); sys.exit(1)"
    payload = _payload(workspace, CommandSucceeds("noisy", (sys.executable, "-c", noisy)))
    out, err = io.StringIO(), io.StringIO()

    assert run_compiled_contract(payload, stdout=out, stderr=err) == EXIT_OBSERVED

    assert "hunter2" not in out.getvalue() + err.getvalue()


# --------------------------------------------------------------------------- #
# Tests: one pytest session per tick, facts vs omissions
# --------------------------------------------------------------------------- #


def _test_workspace(root: Path) -> Path:
    workspace = _workspace(root)
    (workspace / "tests" / "test_calc.py").write_text(
        textwrap.dedent(
            """\
            import pytest

            def test_pass():
                pass

            def test_fail():
                assert False

            @pytest.mark.skip(reason="not yet")
            def test_skip():
                pass

            def test_not_required():
                open("NOT_REQUIRED_RAN", "w").write("x")
            """
        ),
        encoding="utf-8",
    )
    (workspace / "tests" / "test_broken.py").write_text(
        "import module_that_does_not_exist\n\ndef test_x():\n    pass\n", encoding="utf-8"
    )
    return workspace


def test_required_tests_report_facts_from_a_single_session(tmp_path: Path) -> None:
    workspace = _test_workspace(tmp_path)
    payload = _payload(
        workspace,
        TestsImprove(
            tests=(
                "tests/test_calc.py::test_pass",
                "tests/test_calc.py::test_fail",
                "tests/test_calc.py::test_skip",
                "tests/test_calc.py::test_deleted",
                "tests/test_broken.py::test_x",
                "tests/test_not_written_yet.py::test_new",
            )
        ),
    )

    status, checkpoints, reasons = _run(payload)

    assert status == EXIT_OBSERVED
    # An import error may be the environment's (a missing dependency), not the
    # agent's: never a negative fact that could stall a healthy run.
    assert reasons == {"test:tests/test_broken.py::test_x": REASON_TEST_UNCOLLECTABLE}
    assert checkpoints == {
        "test:tests/test_calc.py::test_deleted": False,  # no such test
        "test:tests/test_calc.py::test_fail": False,
        "test:tests/test_calc.py::test_pass": True,
        "test:tests/test_calc.py::test_skip": False,  # skipped is not passing
        "test:tests/test_not_written_yet.py::test_new": False,  # file absent
    }
    # Unrelated tests in the same file are deselected, and the probe leaves no
    # cache or bytecode behind in the observed workspace.
    assert not (workspace / "NOT_REQUIRED_RAN").exists()
    assert not (workspace / ".pytest_cache").exists()
    assert not list(workspace.rglob("__pycache__"))


def test_pytest_timeout_omits_every_test_checkpoint_but_keeps_files(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "tests" / "test_slow.py").write_text(
        "import time\n\ndef test_slow():\n    time.sleep(60)\n", encoding="utf-8"
    )
    payload = _payload(
        workspace,
        FileExists("out.txt"),
        TestsImprove(tests=("tests/test_slow.py::test_slow", "tests/test_anchor.py::test_anchor")),
        timeout=1.5,
    )

    started = time.monotonic()
    _, checkpoints, reasons = _run(payload)

    assert time.monotonic() - started < 15
    assert checkpoints == {"file_exists:out.txt": False}
    assert reasons == {
        "test:tests/test_anchor.py::test_anchor": REASON_PYTEST_TIMEOUT,
        "test:tests/test_slow.py::test_slow": REASON_PYTEST_TIMEOUT,
    }


def test_broken_conftest_is_not_mistaken_for_failing_tests(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(workspace, FileExists("out.txt"), _ANCHOR)
    (workspace / "conftest.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")

    _, checkpoints, reasons = _run(payload)

    assert checkpoints == {"file_exists:out.txt": False}
    assert reasons == {"test:tests/test_anchor.py::test_anchor": REASON_PYTEST_UNREADABLE}


def test_broken_test_package_omits_only_the_tests_below_it(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    package = workspace / "tests" / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("raise ImportError('missing dependency')\n", "utf-8")
    (package / "test_inner.py").write_text("def test_inner():\n    pass\n", "utf-8")
    payload = _payload(
        workspace,
        TestsImprove(
            tests=("tests/pkg/test_inner.py::test_inner", "tests/test_anchor.py::test_anchor")
        ),
    )

    _, checkpoints, reasons = _run(payload)

    assert checkpoints == {"test:tests/test_anchor.py::test_anchor": True}
    assert reasons == {"test:tests/pkg/test_inner.py::test_inner": REASON_TEST_UNCOLLECTABLE}


def test_installed_plugins_are_available_to_the_tests(tmp_path: Path) -> None:
    """A suite that relies on a plugin fixture must not read as failing."""
    pytest.importorskip("pytest_cov")
    workspace = _workspace(tmp_path)
    (workspace / "tests" / "test_plugin.py").write_text(
        "def test_uses_plugin_fixture(no_cover):\n    pass\n", encoding="utf-8"
    )
    payload = _payload(workspace, TestPasses("tests/test_plugin.py::test_uses_plugin_fixture"))

    assert _run(payload)[1] == {"test:tests/test_plugin.py::test_uses_plugin_fixture": True}


def test_addopts_cannot_hide_a_required_test(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "pytest.ini").write_text("[pytest]\naddopts = -x -k test_fail\n", "utf-8")
    (workspace / "tests" / "test_two.py").write_text(
        "def test_fail():\n    assert False\n\ndef test_pass():\n    pass\n", "utf-8"
    )
    payload = _payload(
        workspace,
        TestsImprove(tests=("tests/test_two.py::test_fail", "tests/test_two.py::test_pass")),
    )

    assert _run(payload)[1] == {
        "test:tests/test_two.py::test_fail": False,
        "test:tests/test_two.py::test_pass": True,
    }


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX wrapper interpreters")


def _project_python(path: Path) -> Path:
    """An interpreter like a project venv's: it can run pytest, not import pathlayer."""
    site = Path(pytest.__file__).resolve().parent.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    pythonpath = f'PYTHONPATH="{site}${{PYTHONPATH:+:$PYTHONPATH}}"'
    path.write_text(f'#!/bin/sh\n{pythonpath} exec "{sys.executable}" -S "$@"\n', "utf-8")
    path.chmod(0o755)
    return path


@posix_only
def test_tests_run_under_the_project_interpreter_without_pathlayer(tmp_path: Path) -> None:
    python = _project_python(tmp_path / "project-env" / "bin" / "python")
    isolated = subprocess.run(
        [str(python), "-c", "import pathlayer"], cwd=tmp_path, capture_output=True, check=False
    )
    if isolated.returncode == 0:
        pytest.skip("pathlayer is installed in site-packages; cannot build an isolated interpreter")
    workspace = _workspace(tmp_path)
    (workspace / "tests" / "test_env.py").write_text(
        "import importlib.util\n\n"
        "def test_runs_without_pathlayer():\n"
        "    assert importlib.util.find_spec('pathlayer') is None\n\n"
        "def test_fails():\n"
        "    assert False\n",
        encoding="utf-8",
    )
    contract = CodingContract(
        task="probe-test",
        objective="Run the project's tests with the project's interpreter.",
        workspace=workspace,
        python=str(python),
        progress=(
            TestsImprove(
                tests=(
                    "tests/test_env.py::test_runs_without_pathlayer",
                    "tests/test_env.py::test_fails",
                )
            ),
        ),
    )
    payload = dict(compile_coding_contract(contract).payload)

    status, checkpoints, reasons = _run(payload)

    assert status == EXIT_OBSERVED, reasons
    assert checkpoints == {
        "test:tests/test_env.py::test_fails": False,
        "test:tests/test_env.py::test_runs_without_pathlayer": True,
    }


@posix_only
def test_test_interpreter_removed_after_compile_is_unobserved(tmp_path: Path) -> None:
    python = _project_python(tmp_path / "project-env" / "bin" / "python")
    workspace = _workspace(tmp_path)
    contract = CodingContract(
        task="probe-test",
        objective="Lose the interpreter.",
        workspace=workspace,
        python=str(python),
        progress=(_ANCHOR, FileExists("out.txt")),
    )
    payload = dict(compile_coding_contract(contract).payload)
    python.unlink()

    _, checkpoints, reasons = _run(payload)

    assert checkpoints == {"file_exists:out.txt": False}
    assert reasons == {"test:tests/test_anchor.py::test_anchor": REASON_TEST_INTERPRETER}


def test_plugin_directory_goes_first_on_pythonpath_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/project/src", "/project/lib"]))

    env = coding_contract_probe._pytest_env(tmp_path, {})

    assert env["PYTHONPATH"].split(os.pathsep) == [str(tmp_path), "/project/src", "/project/lib"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_interpreter_check_is_cached_per_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    check_test_interpreter(sys.executable, 30.0)

    def unreachable(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("a verified interpreter is not checked again")

    monkeypatch.setattr(coding_contract_probe, "_launch", unreachable)
    check_test_interpreter(sys.executable, 30.0)


def test_missing_pytest_is_omitted_not_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``python -m <missing module>`` exits 1 -- a "fact" status -- but writes no
    report, so a core install without pytest never turns tests into ``false``."""
    workspace = _workspace(tmp_path)
    payload = _payload(workspace, FileExists("out.txt"), _ANCHOR)
    monkeypatch.setattr(coding_contract_probe, "_PYTEST_MODULE", "pathlayer_missing_runner_xyz")

    _, checkpoints, reasons = _run(payload)

    assert checkpoints == {"file_exists:out.txt": False}
    assert reasons == {"test:tests/test_anchor.py::test_anchor": REASON_PYTEST_UNREADABLE}


def test_interrupted_session_is_omitted_even_with_a_report(tmp_path: Path) -> None:
    """An incomplete session says nothing about the tests it never ran."""
    workspace = _workspace(tmp_path)
    (workspace / "tests" / "test_exit.py").write_text(
        "import pytest\n\ndef test_exit():\n    pytest.exit('suite aborted', returncode=2)\n",
        encoding="utf-8",
    )
    payload = _payload(
        workspace,
        FileExists("out.txt"),
        TestsImprove(tests=("tests/test_exit.py::test_exit", "tests/test_anchor.py::test_anchor")),
    )

    _, checkpoints, reasons = _run(payload)

    assert checkpoints == {"file_exists:out.txt": False}
    assert reasons == {
        "test:tests/test_anchor.py::test_anchor": REASON_PYTEST_UNREADABLE,
        "test:tests/test_exit.py::test_exit": REASON_PYTEST_UNREADABLE,
    }


def test_test_file_turned_directory_is_omitted(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(workspace, _ANCHOR, FileExists("out.txt"))
    (workspace / "tests" / "test_anchor.py").unlink()
    (workspace / "tests" / "test_anchor.py").mkdir()

    _, checkpoints, reasons = _run(payload)

    assert checkpoints == {"file_exists:out.txt": False}
    assert reasons == {"test:tests/test_anchor.py::test_anchor": REASON_NOT_REGULAR}


def test_exhausted_budget_omits_the_remaining_subprocess_checks(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(
        workspace,
        FileExists("out.txt"),
        _ANCHOR,
        CommandSucceeds("ok", (sys.executable, "-V")),
        timeout=5,
    )
    readings = iter([0.0, 100.0, 100.0, 100.0])

    _, checkpoints, reasons = _run(payload, clock=lambda: next(readings))

    assert checkpoints == {"file_exists:out.txt": False}
    assert reasons == {
        "command:ok": REASON_BUDGET,
        "test:tests/test_anchor.py::test_anchor": REASON_BUDGET,
    }


def test_an_unexpected_error_in_one_check_costs_only_that_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(workspace, FileExists("a.txt"), FileExists("b.txt"))
    original = coding_contract_probe._evaluate_file

    def flaky(workspace: Path, check: Any) -> bool:
        if check.path == "a.txt":
            raise KeyError("boom")
        return original(workspace, check)

    monkeypatch.setattr(coding_contract_probe, "_evaluate_file", flaky)

    status, checkpoints, reasons = _run(payload)

    assert status == EXIT_OBSERVED
    assert checkpoints == {"file_exists:b.txt": False}
    assert reasons == {"file_exists:a.txt": REASON_UNEXPECTED}


# --------------------------------------------------------------------------- #
# Payload robustness (it is read back from an operator-owned file)
# --------------------------------------------------------------------------- #


def _mutations(payload: dict[str, Any]) -> list[object]:
    def changed(**updates: Any) -> dict[str, Any]:
        return {**payload, **updates}

    files = payload["files"]
    return [
        None,
        [],
        "payload",
        changed(schema_version=2),
        changed(schema_version=True),
        changed(format="something-else"),
        changed(extra=1),
        {key: value for key, value in payload.items() if key != "commands"},
        changed(timeout_seconds=float("nan")),
        changed(timeout_seconds=0),
        changed(workspace=""),
        changed(files=[{**files[0], "path": "../escape.txt"}]),
        changed(files=[{**files[0], "path": "/abs.txt"}]),
        changed(files=[{**files[0], "kind": "file_teleports"}]),
        changed(files=[{**files[0], "unexpected": True}]),
        changed(files=[files[0], files[0]]),
        changed(tests={"-p evil::x": "test:x"}),
        changed(tests={"tests/test_anchor.py": "test:x"}),
        changed(commands=[{"checkpoint": "command:x", "argv": ["a\x00b"]}]),
        changed(commands=[{"checkpoint": "command:x", "argv": []}]),
        changed(python="relative/python"),
        changed(python=3),
        {key: value for key, value in payload.items() if key != "python"},
        changed(files=[], tests={}, commands=[]),
    ]


def test_unreadable_payload_exits_3_without_observing_anything(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    payload = _payload(workspace, FileExists("a.txt"), _ANCHOR)

    for mutated in _mutations(payload):
        status, checkpoints, _ = _run(mutated)
        assert status == EXIT_INVALID_CONTRACT, mutated
        assert checkpoints is None


# --------------------------------------------------------------------------- #
# The generated probe, exactly as the ANC-2 producer runs it
# --------------------------------------------------------------------------- #


def _install(tmp_path: Path, *progress: Any) -> tuple[Path, Any]:
    workspace = _workspace(tmp_path)
    installed = write_coding_contract(
        CodingContract(
            task="probe-test",
            objective="Exercise the generated probe.",
            workspace=workspace,
            progress=progress or (_ANCHOR, FileChanges("tests/test_anchor.py")),
        ),
        output_dir=tmp_path / "bundle",
    )
    return workspace, installed


def test_generated_probe_prints_the_anc2_wire_shape(tmp_path: Path) -> None:
    workspace, installed = _install(tmp_path)
    in_process = _run(installed.compiled.payload)[1]

    completed = subprocess.run(
        [sys.executable, str(installed.paths.probe)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=tmp_path,
    )

    assert completed.returncode == EXIT_OBSERVED, completed.stderr
    assert json.loads(completed.stdout) == {"checkpoints": in_process}
    assert in_process == {
        "file_changes:tests/test_anchor.py": False,
        "test:tests/test_anchor.py::test_anchor": True,
    }
    assert workspace.is_dir()


def test_generated_probe_without_pathlayer_exits_2_explicitly(tmp_path: Path) -> None:
    _, installed = _install(tmp_path)
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    completed = subprocess.run(
        [sys.executable, "-S", str(installed.paths.probe)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=neutral,
        env=env,
    )

    assert completed.returncode == EXIT_PATHLAYER_UNAVAILABLE
    assert completed.stdout == ""
    assert "pathlayer is not importable" in completed.stderr


def proof_spec_required(installed: Any) -> tuple[str, ...]:
    required: tuple[str, ...] = installed.compiled.bundle.proof_spec.required
    return required


def test_real_producer_attaches_a_valid_proof_from_the_generated_probe(tmp_path: Path) -> None:
    workspace, installed = _install(tmp_path)
    producer = build_producer(
        str(installed.paths.manifest),
        (sys.executable, str(installed.paths.probe)),
        installed.compiled.producer_timeout_seconds,
    )

    result = producer.probe()
    assert set(result) == {"proof_spec", "progress_proof"}
    spec_payload, proof_payload = result["proof_spec"], result["progress_proof"]
    proof = validate_progress_proof(proof_payload)
    assert proof is not None
    assert proof.spec_hash == producer.spec_hash
    assert dict(proof.checkpoints) == {
        "file_changes:tests/test_anchor.py": False,
        "test:tests/test_anchor.py::test_anchor": True,
    }
    assert spec_payload["required"] == list(installed.compiled.bundle.proof_spec.required)

    # Nothing observable -> exit 4 -> ANC-2 attaches a marker for every
    # required checkpoint instead of a proof: a gap, not a freeze.
    shutil.rmtree(workspace)
    result = producer.probe()
    assert "progress_proof" not in result
    assert result["progress_proof_unobserved"]["unobserved"] == list(proof_spec_required(installed))
    assert result["progress_proof_unobserved"]["reason"] == "exited with status 4"
