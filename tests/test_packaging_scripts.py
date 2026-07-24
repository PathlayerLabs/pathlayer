"""Tests for release artifact validation helpers."""

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
            b"Name: unrelated-project\nVersion: 0.1.0\n",
            version="0.1.0",
            artifact="wheel",
        )


def test_duplicate_wheel_entries_are_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "pathlayer-0.1.0-py3-none-any.whl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("pathlayer/py.typed", b"first")
            archive.writestr("pathlayer/py.typed", b"second")

    with pytest.raises(SystemExit, match="duplicate entries"):
        _check_wheel(wheel, version="0.1.0")


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
