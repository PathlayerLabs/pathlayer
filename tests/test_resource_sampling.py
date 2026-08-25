"""RES-3: tests for the RSS sampling primitive."""

from __future__ import annotations

import io
from typing import Any

import pytest

from pathlayer import resource_sampling


def _fake_open(content: str) -> Any:
    def _open(path: str, encoding: str | None = None) -> io.StringIO:
        return io.StringIO(content)

    return _open


def test_sample_rss_bytes_reads_vmrss_and_converts_kb_to_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "Name:\tpython3\nVmRSS:\t   12345 kB\nVmSize:\t 99999 kB\n"
    monkeypatch.setattr(resource_sampling, "open", _fake_open(content), raising=False)

    assert resource_sampling.sample_rss_bytes() == 12345 * 1024


def test_sample_rss_bytes_returns_none_when_vmrss_line_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "Name:\tpython3\nVmSize:\t 99999 kB\n"
    monkeypatch.setattr(resource_sampling, "open", _fake_open(content), raising=False)

    assert resource_sampling.sample_rss_bytes() is None


def test_sample_rss_bytes_returns_none_when_proc_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_open(path: str, encoding: str | None = None) -> io.StringIO:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(resource_sampling, "open", _raise_open, raising=False)

    assert resource_sampling.sample_rss_bytes() is None


def test_sample_rss_bytes_returns_none_on_malformed_vmrss_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "VmRSS:\tnot-a-number kB\n"
    monkeypatch.setattr(resource_sampling, "open", _fake_open(content), raising=False)

    assert resource_sampling.sample_rss_bytes() is None


def test_sample_rss_bytes_returns_none_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_open(path: str, encoding: str | None = None) -> io.StringIO:
        raise OSError("simulated failure")

    monkeypatch.setattr(resource_sampling, "open", _raise_open, raising=False)

    assert resource_sampling.sample_rss_bytes() is None


def test_sample_rss_bytes_against_the_real_filesystem() -> None:
    # No mocking: on Linux this resolves to a real, positive int (this
    # process definitely has some RSS); on any platform without /proc
    # (e.g. macOS, the dev/CI box this may run on) it's gracefully None.
    result = resource_sampling.sample_rss_bytes()
    assert result is None or (isinstance(result, int) and result > 0)
