#!/usr/bin/env bash
set -euo pipefail

version="$(
  python - <<'PY'
import pathlib
import tomllib

print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])
PY
)"

python -c 'import shutil; shutil.rmtree("dist", ignore_errors=True)'
uv build
twine check dist/*
python scripts/check_dist.py dist --version "$version"
