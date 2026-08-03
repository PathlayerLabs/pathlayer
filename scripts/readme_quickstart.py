"""Extract the executable core quickstart from README.md."""

from __future__ import annotations

from pathlib import Path

START_MARKER = "<!-- pathlayer-quickstart:start -->"
END_MARKER = "<!-- pathlayer-quickstart:end -->"


def extract_quickstart(readme: Path) -> str:
    """Return the single marked Python block from ``readme``.

    The strict shape keeps documentation tests honest: duplicate markers,
    another fence inside the marked section, or a non-Python block fail instead
    of silently executing a different snippet.
    """
    markdown = readme.read_text(encoding="utf-8")
    if markdown.count(START_MARKER) != 1 or markdown.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one quickstart marker pair")

    start = markdown.index(START_MARKER)
    end = markdown.index(END_MARKER)
    if end <= start:
        raise ValueError("README quickstart end marker must follow its start marker")

    marked = markdown[start + len(START_MARKER) : end].strip()
    lines = marked.splitlines()
    if len(lines) < 3 or lines[0] != "```python" or lines[-1] != "```":
        raise ValueError("marked quickstart must contain exactly one Python code fence")
    if any(line.startswith("```") for line in lines[1:-1]):
        raise ValueError("marked quickstart contains an unexpected nested code fence")
    return "\n".join(lines[1:-1]) + "\n"
