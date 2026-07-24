"""Validate that release artifacts contain only the intended public package."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from collections import Counter
from collections.abc import Iterable
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - covered by the Python 3.10 CI job
    import tomli as tomllib

_SDIST_ALLOWED_ROOTS = {
    ".gitignore",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "pathlayer",
    "pyproject.toml",
}
_FORBIDDEN_PARTS = {
    ".cursor",
    ".env",
    ".github",
    "__pycache__",
    "report_assets",
    "tests",
    "traces",
}
_FORBIDDEN_SUFFIXES = {".docx", ".html", ".pyc", ".pyo", ".sqlite", ".xlsx"}
_SENSITIVE_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:[A-Z0-9][A-Z0-9-]* )*PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
}


def _fail(message: str) -> None:
    raise SystemExit(message)


def _assert_safe_paths(paths: Iterable[PurePosixPath]) -> None:
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            _fail(f"unsafe path in artifact: {path}")
        if any(part in _FORBIDDEN_PARTS for part in path.parts):
            _fail(f"forbidden path in artifact: {path}")
        if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
            _fail(f"forbidden file type in artifact: {path}")


def _assert_safe_contents(files: Iterable[tuple[PurePosixPath, bytes]]) -> None:
    for path, content in files:
        for label, pattern in _SENSITIVE_PATTERNS.items():
            if pattern.search(content):
                _fail(f"possible {label} in artifact file: {path}")


def _project_metadata() -> dict[str, Any]:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def _requirements_for_extra(
    extra: str,
    *,
    project_name: str,
    extras: dict[str, list[str]],
    source_extra: str | None = None,
) -> list[str]:
    source = source_extra or extra
    requirements: list[str] = []
    for raw_requirement in extras[source]:
        requirement = Requirement(raw_requirement)
        if canonicalize_name(requirement.name) == canonicalize_name(project_name):
            for nested_extra in sorted(requirement.extras):
                requirements.extend(
                    _requirements_for_extra(
                        extra,
                        project_name=project_name,
                        extras=extras,
                        source_extra=nested_extra,
                    )
                )
            continue
        rendered = str(requirement)
        if requirement.marker is not None:
            rendered = rendered.split(";", 1)[0]
        normalized = f"{canonicalize_name(requirement.name)}{rendered[len(requirement.name) :]}"
        marker = f"extra == '{extra}'"
        if requirement.marker is not None:
            marker = f"({requirement.marker}) and {marker}"
        requirements.append(f"{normalized}; {marker}")
    return requirements


def _assert_metadata(content: bytes, *, version: str, artifact: str) -> None:
    metadata = BytesParser().parsebytes(content)
    project = _project_metadata()
    if project["version"] != version:
        _fail(f"requested version {version!r} differs from pyproject: {project['version']!r}")
    actual_name = metadata.get("Name")
    if actual_name is None or canonicalize_name(actual_name) != canonicalize_name(project["name"]):
        _fail(f"{artifact} metadata Name differs: {actual_name!r}")

    required_single_values = {
        "Version": version,
        "License-Expression": project["license"],
        "Requires-Python": project["requires-python"],
    }
    wrong_single_values = {
        f"{key}: {metadata.get(key)!r}"
        for key, expected in required_single_values.items()
        if metadata.get(key) != expected
    }
    if wrong_single_values:
        _fail(f"{artifact} metadata differs: {sorted(wrong_single_values)}")

    extras = project["optional-dependencies"]
    expected_requirements = {
        requirement
        for extra in extras
        for requirement in _requirements_for_extra(
            extra,
            project_name=project["name"],
            extras=extras,
        )
    }
    required_multi_values = {
        "Classifier": set(project["classifiers"]),
        "License-File": set(project["license-files"]),
        "Project-URL": {f"{name}, {url}" for name, url in project["urls"].items()},
        "Provides-Extra": set(extras),
        "Requires-Dist": expected_requirements,
    }
    for key, expected_values in required_multi_values.items():
        actual_values = metadata.get_all(key, [])
        if key == "Requires-Dist":
            expected_values = {str(Requirement(value)) for value in expected_values}
            actual_values = [str(Requirement(value)) for value in actual_values]
        if Counter(actual_values) != Counter(expected_values):
            _fail(
                f"{artifact} metadata {key} differs: "
                f"expected={sorted(expected_values)}, actual={sorted(actual_values)}"
            )


def _check_sdist(archive: Path, *, version: str) -> None:
    expected_root = f"pathlayer-{version}"
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if any(member.issym() or member.islnk() for member in members):
            _fail(f"sdist contains a link: {archive}")
        paths = [PurePosixPath(member.name) for member in members if member.name]
        files = []
        for member in members:
            if member.isfile():
                source_file = source.extractfile(member)
                if source_file is not None:
                    files.append((PurePosixPath(member.name), source_file.read()))
        _assert_safe_contents(files)

    roots = {path.parts[0] for path in paths}
    if roots != {expected_root}:
        _fail(f"unexpected sdist root(s): {sorted(roots)}")

    relative = [PurePosixPath(*path.parts[1:]) for path in paths if len(path.parts) > 1]
    _assert_safe_paths(relative)
    unexpected = {
        path.parts[0]
        for path in relative
        if path.parts and path.parts[0] not in _SDIST_ALLOWED_ROOTS
    }
    if unexpected:
        _fail(f"unexpected top-level sdist content: {sorted(unexpected)}")

    required = {
        PurePosixPath("LICENSE"),
        PurePosixPath("PKG-INFO"),
        PurePosixPath("README.md"),
        PurePosixPath("pathlayer/__init__.py"),
        PurePosixPath("pathlayer/py.typed"),
        PurePosixPath("pyproject.toml"),
    }
    missing = required.difference(relative)
    if missing:
        _fail(f"sdist missing required files: {sorted(map(str, missing))}")
    pkg_info = next(
        content for path, content in files if path == PurePosixPath(f"{expected_root}/PKG-INFO")
    )
    _assert_metadata(pkg_info, version=version, artifact="sdist")


def _check_wheel(archive: Path, *, version: str) -> None:
    dist_info = f"pathlayer-{version}.dist-info"
    with zipfile.ZipFile(archive) as wheel:
        infos = wheel.infolist()
        names = [info.filename for info in infos if info.filename]
        if len(names) != len(set(names)):
            _fail(f"wheel contains duplicate entries: {archive}")
        paths = [PurePosixPath(name) for name in names]
        _assert_safe_paths(paths)
        _assert_safe_contents(
            (PurePosixPath(info.filename), wheel.read(info))
            for info in infos
            if info.filename and not info.is_dir()
        )
        unexpected = [
            path for path in paths if path.parts and path.parts[0] not in {"pathlayer", dist_info}
        ]
        if unexpected:
            _fail(f"unexpected wheel content: {unexpected[0]}")

        required = {
            PurePosixPath("pathlayer/__init__.py"),
            PurePosixPath("pathlayer/py.typed"),
            PurePosixPath(f"{dist_info}/METADATA"),
            PurePosixPath(f"{dist_info}/licenses/LICENSE"),
        }
        missing = required.difference(paths)
        if missing:
            _fail(f"wheel missing required files: {sorted(map(str, missing))}")

        _assert_metadata(
            wheel.read(f"{dist_info}/METADATA"),
            version=version,
            artifact="wheel",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    wheels = sorted(args.dist.glob("*.whl"))
    sdists = sorted(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        _fail(
            f"expected one wheel and one sdist, got {len(wheels)} wheel(s), {len(sdists)} sdist(s)"
        )

    _check_wheel(wheels[0], version=args.version)
    _check_sdist(sdists[0], version=args.version)
    print(f"artifacts-ok version={args.version}")


if __name__ == "__main__":
    main()
