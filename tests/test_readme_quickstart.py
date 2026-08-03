"""Keep the README core quickstart executable and self-contained."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.readme_quickstart import END_MARKER, START_MARKER, extract_quickstart

REPO_ROOT = Path(__file__).parents[1]
README = REPO_ROOT / "README.md"


def test_marked_quickstart_runs_from_an_empty_directory(tmp_path: Path) -> None:
    script = tmp_path / "quickstart.py"
    script.write_text(extract_quickstart(README), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Pathlayer stopped the loop after" in completed.stdout
    assert not (tmp_path / ".pathlayer").exists()


@pytest.mark.parametrize(
    "markdown",
    [
        "",
        f"{START_MARKER}\n```python\nprint('missing end')\n```",
        f"{START_MARKER}\n```text\nnot Python\n```\n{END_MARKER}",
        f"{END_MARKER}\n{START_MARKER}\n```python\nprint('reversed')\n```",
        (
            f"{START_MARKER}\n```python\nprint('one')\n```\n{END_MARKER}\n"
            f"{START_MARKER}\n```python\nprint('two')\n```\n{END_MARKER}"
        ),
    ],
)
def test_quickstart_extraction_rejects_ambiguous_markdown(tmp_path: Path, markdown: str) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(markdown, encoding="utf-8")

    with pytest.raises(ValueError):
        extract_quickstart(readme)
