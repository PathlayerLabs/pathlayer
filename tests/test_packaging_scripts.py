"""Tests for release artifact validation helpers."""

import tomllib
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from scripts import verify_index_hashes
from scripts.check_dist import (
    _assert_metadata,
    _assert_safe_contents,
    _assert_safe_paths,
    _check_wheel,
)

# check_dist compares the requested version with pyproject, so the tests use
# the project's own version instead of a literal that a release bump breaks.
_VERSION = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]


def test_public_contact_and_documentation_placeholder_are_allowed() -> None:
    _assert_safe_contents(
        [
            (
                PurePosixPath("README.md"),
                b'Contact security@pathlayer.dev; token = "example-value".',
            )
        ]
    )


@pytest.mark.parametrize(
    "content",
    [
        b"-----BEGIN " + b"PRIVATE KEY-----",
        b"-----BEGIN " + b"ENCRYPTED PRIVATE KEY-----",
        b"ghp_" + b"x" * 36,
        b"AKIA" + b"A" * 16,
    ],
)
def test_known_secret_formats_are_rejected(content: bytes) -> None:
    with pytest.raises(SystemExit, match="possible"):
        _assert_safe_contents([(PurePosixPath("pathlayer/config.py"), content)])


@pytest.mark.parametrize(
    "path",
    [
        PurePosixPath("/absolute/pathlayer.py"),
        PurePosixPath("pathlayer/../outside.py"),
    ],
)
def test_unsafe_artifact_paths_are_rejected(path: PurePosixPath) -> None:
    with pytest.raises(SystemExit, match="unsafe path"):
        _assert_safe_paths([path])


def test_wrong_distribution_name_is_rejected() -> None:
    with pytest.raises(SystemExit, match="metadata Name differs"):
        _assert_metadata(
            f"Name: unrelated-project\nVersion: {_VERSION}\n".encode(),
            version=_VERSION,
            artifact="wheel",
        )


def test_duplicate_wheel_entries_are_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / f"pathlayer-{_VERSION}-py3-none-any.whl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("pathlayer/py.typed", b"first")
            archive.writestr("pathlayer/py.typed", b"second")

    with pytest.raises(SystemExit, match="duplicate entries"):
        _check_wheel(wheel, version=_VERSION)


def test_a_wheel_without_the_openclaw_plugin_is_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / f"pathlayer-{_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pathlayer/__init__.py", b"")
        archive.writestr("pathlayer/py.typed", b"")
        archive.writestr("pathlayer/adapters/openclaw_plugin/index.js", b"")
        archive.writestr(f"pathlayer-{_VERSION}.dist-info/METADATA", b"Name: pathlayer\n")
        archive.writestr(f"pathlayer-{_VERSION}.dist-info/licenses/LICENSE", b"")

    with pytest.raises(SystemExit, match="wheel missing required files") as info:
        _check_wheel(wheel, version=_VERSION)
    assert "openclaw_plugin/openclaw.plugin.json" in str(info.value)
    assert "openclaw_plugin/package.json" in str(info.value)
    assert "openclaw_plugin/index.js" not in str(info.value)


def test_index_hash_check_retries_partial_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases = iter(
        [
            {"urls": [{"filename": "pathlayer.whl", "digests": {"sha256": "wheel"}}]},
            {
                "urls": [
                    {"filename": "pathlayer.whl", "digests": {"sha256": "wheel"}},
                    {"filename": "pathlayer.tar.gz", "digests": {"sha256": "sdist"}},
                ]
            },
        ]
    )
    monkeypatch.setattr(verify_index_hashes, "_fetch_release", lambda url: next(releases))

    verify_index_hashes._wait_for_hashes(
        "https://example.invalid/pathlayer/0.1.0/json",
        expected={"pathlayer.whl": "wheel", "pathlayer.tar.gz": "sdist"},
        attempts=2,
        delay=0,
    )
