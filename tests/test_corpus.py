"""VAL-5: tests for the reproducible corpus builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathlayer.validation import VALID_LABELS, build_corpus, load_trace
from pathlayer.validation.corpus import main
from pathlayer.validation.replay import replay_trace


def _all_traces(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def test_build_corpus_writes_balanced_split(tmp_path: Path) -> None:
    summary = build_corpus(tmp_path, n=10, seed=0)
    assert summary.healthy + summary.pathological == 10
    # Alternating even/odd -> 5 healthy, 5 pathological.
    assert summary.healthy == 5
    assert summary.pathological == 5
    assert summary.total_events > 0
    assert len(_all_traces(tmp_path)) == 10


def test_generated_traces_are_well_formed(tmp_path: Path) -> None:
    build_corpus(tmp_path, n=12, seed=3)
    for path in _all_traces(tmp_path):
        trace = load_trace(path)
        assert trace.label in VALID_LABELS
        assert trace.events
        assert all(event.tool for event in trace.events)


def test_build_corpus_is_reproducible(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_corpus(a, n=8, seed=42)
    build_corpus(b, n=8, seed=42)
    files_a = [p.relative_to(a) for p in _all_traces(a)]
    files_b = [p.relative_to(b) for p in _all_traces(b)]
    assert files_a == files_b
    for rel in files_a:
        assert (a / rel).read_text() == (b / rel).read_text()


def test_different_seed_changes_corpus(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_corpus(a, n=8, seed=1)
    build_corpus(b, n=8, seed=2)
    texts_a = {p.read_text() for p in _all_traces(a)}
    texts_b = {p.read_text() for p in _all_traces(b)}
    assert texts_a != texts_b


def test_build_corpus_is_idempotent(tmp_path: Path) -> None:
    build_corpus(tmp_path, n=6, seed=0)
    first = _all_traces(tmp_path)
    build_corpus(tmp_path, n=6, seed=0)
    second = _all_traces(tmp_path)
    # Clean-before-write: no duplication, identical file set.
    assert first == second


def test_generated_healthy_never_trips_the_loop_detector(tmp_path: Path) -> None:
    build_corpus(tmp_path, n=20, seed=7)
    for path in _all_traces(tmp_path / "healthy"):
        assert not replay_trace(load_trace(path)).intervened


def test_build_corpus_rejects_negative_n(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_corpus(tmp_path, n=-1)


def test_build_corpus_zero_is_empty(tmp_path: Path) -> None:
    summary = build_corpus(tmp_path, n=0)
    assert summary.healthy == 0
    assert summary.pathological == 0
    assert _all_traces(tmp_path) == []


def test_summary_format_lists_categories(tmp_path: Path) -> None:
    text = build_corpus(tmp_path, n=10, seed=0).format()
    assert "corpus:" in text
    assert "per category:" in text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_main_builds_into_fresh_dir(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "corpus"
    code = main(["--out", str(out), "--n", "6", "--seed", "1"])
    assert code == 0
    assert "corpus:" in capsys.readouterr().out
    assert len(_all_traces(out)) == 6


def test_main_refuses_existing_dir_without_yes(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "corpus"
    out.mkdir()
    code = main(["--out", str(out), "--n", "4"])
    assert code == 1
    assert "refusing to overwrite" in capsys.readouterr().out


def test_main_overwrites_with_yes(tmp_path: Path) -> None:
    out = tmp_path / "corpus"
    out.mkdir()
    code = main(["--out", str(out), "--n", "4", "--yes"])
    assert code == 0
    assert len(_all_traces(out)) == 4
