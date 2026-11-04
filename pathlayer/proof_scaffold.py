"""Deterministic scaffolding for ANC-1/ANC-2 progress-proof producers.

Pathlayer owns the wire format; the operator owns the business truth.  This
module therefore renders only a validated ``ProofSpec`` manifest, an inert
probe skeleton, and local setup instructions.  It never derives checkpoint
values from the task name, objective, event text, or an LLM.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pathlayer.detectors.proof import ProofSpec, validate_proof_spec

_MANIFEST_NAME = "proof_spec.json"
_PROBE_NAME = "probe.py"
_README_NAME = "README.md"


class ScaffoldError(ValueError):
    """Raised when scaffold inputs or output targets are unsafe."""


@dataclass(frozen=True, slots=True)
class ScaffoldBundle:
    """The three deterministic text artifacts produced by the pure renderer."""

    proof_spec: ProofSpec
    manifest_json: str
    probe_python: str
    readme_markdown: str


@dataclass(frozen=True, slots=True)
class ScaffoldPaths:
    """Filesystem locations written by :func:`write_progress_proof_scaffold`."""

    manifest: Path
    probe: Path
    readme: Path


def _spec_payload(spec: ProofSpec) -> dict[str, object]:
    return {
        "schema_version": spec.schema_version,
        "task_id": spec.task_id,
        "proof_id": spec.proof_id,
        "scope_id": spec.scope_id,
        "kind": spec.kind,
        "required": list(spec.required),
    }


def _validated_spec(
    *, task_id: str, proof_id: str, scope_id: str, checkpoints: Sequence[str]
) -> ProofSpec:
    """Build the V1 ``checkpoint_set`` spec, or raise :class:`ScaffoldError`."""
    if isinstance(checkpoints, (str, bytes)):
        raise ScaffoldError("checkpoints must be a sequence of checkpoint ids, not text")
    raw_spec = {
        "schema_version": 1,
        "task_id": task_id,
        "proof_id": proof_id,
        "scope_id": scope_id,
        "kind": "checkpoint_set",
        "required": list(checkpoints),
    }
    spec = validate_proof_spec(raw_spec)
    if spec is None:
        raise ScaffoldError("generated proof_spec failed schema validation")
    return spec


def _manifest_json(spec: ProofSpec) -> str:
    return json.dumps(_spec_payload(spec), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _quoted(value: object) -> str:
    """Return a deterministic string/list literal valid in JSON and Python."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _blockquote(value: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in value.splitlines())


def _render_probe(*, name: str, objective: str, required: tuple[str, ...]) -> str:
    return f'''#!/usr/bin/env python3
"""External progress probe for {_quoted(name)}.

Objective (descriptive only): {_quoted(objective)}
"""

from __future__ import annotations

import json
import sys

REQUIRED_CHECKPOINTS = {_quoted(list(required))}


def collect_checkpoints() -> dict[str, bool]:
    """Return one boolean fact for every required checkpoint.

    Replace this exception with checks against state outside the observed
    agent's control.  Do not infer success from agent-authored text.
    """
    raise NotImplementedError(
        "implement external checks for: " + ", ".join(REQUIRED_CHECKPOINTS)
    )


def main() -> int:
    json.dump({{"checkpoints": collect_checkpoints()}}, sys.stdout, sort_keys=True)
    sys.stdout.write("\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _render_readme(
    *, name: str, objective: str, spec: ProofSpec, output_names: tuple[str, str]
) -> str:
    manifest_name, probe_name = output_names
    checkpoints = "\n".join(f"- {_quoted(checkpoint)}" for checkpoint in spec.required)
    return f"""# Pathlayer progress-proof scaffold

This directory is a deterministic Pathlayer progress-proof scaffold.

## Name

{_blockquote(name)}

## Objective

{_blockquote(objective)}

The name and objective are descriptive only. They never determine progress.
The operator remains responsible for the business meaning of every checkpoint.

## Required checkpoints

{checkpoints}

## Implement and test the probe

Edit `{probe_name}` so `collect_checkpoints()` reads state outside the observed
agent's control and returns one boolean per checkpoint. Until then the probe
raises `NotImplementedError`, so Pathlayer receives no proof and stays inert.

The manifest, probe code, and state it reads must all remain outside the
observed agent's write boundary (or be protected by equivalent integration
controls). If the agent can edit its own probe, the proof is self-reported and
must not be trusted.

```bash
python {probe_name}
```

Successful stdout must have this shape:

```json
{{"checkpoints": {{"checkpoint-id": true}}}}
```

## Configure Pathlayer

Run Pathlayer from this directory or replace the relative paths below:

```json
{{
  "detectors": {{
    "progress_proof": {{
      "enabled": true,
      "manifest_path": "./{manifest_name}",
      "probe_command": ["python", "./{probe_name}"]
    }}
  }}
}}
```

The manifest was validated against the `ProofSpec` schema when this
scaffold was generated. Pathlayer validates it again before a run starts.
"""


def render_progress_proof_scaffold(
    *,
    task_id: str,
    name: str,
    objective: str,
    checkpoints: Sequence[str],
    proof_id: str | None = None,
    scope_id: str | None = None,
) -> ScaffoldBundle:
    """Render a progress-proof scaffold without reading or writing anything.

    ``proof_id`` defaults to the descriptive ``name`` and ``scope_id`` to
    ``task_id``.  Those defaults identify the contract; neither value is used
    to infer checkpoint state.
    """
    if not isinstance(name, str) or not name:
        raise ScaffoldError("name must be a non-empty string")
    if not isinstance(objective, str) or not objective:
        raise ScaffoldError("objective must be a non-empty string")
    spec = _validated_spec(
        task_id=task_id,
        proof_id=name if proof_id is None else proof_id,
        scope_id=task_id if scope_id is None else scope_id,
        checkpoints=checkpoints,
    )
    return ScaffoldBundle(
        proof_spec=spec,
        manifest_json=_manifest_json(spec),
        probe_python=_render_probe(name=name, objective=objective, required=spec.required),
        readme_markdown=_render_readme(
            name=name,
            objective=objective,
            spec=spec,
            output_names=(_MANIFEST_NAME, _PROBE_NAME),
        ),
    )


def write_progress_proof_scaffold(
    *,
    output_dir: str | Path,
    task_id: str,
    name: str,
    objective: str,
    checkpoints: Sequence[str],
    proof_id: str | None = None,
    scope_id: str | None = None,
) -> ScaffoldPaths:
    """Render and write all three files, refusing to overwrite any target.

    All content is rendered and staged before installation.  Exclusive file
    creation protects against a target appearing after the initial preflight;
    any files installed by a failed call are removed before the error escapes.
    """
    bundle = render_progress_proof_scaffold(
        task_id=task_id,
        name=name,
        objective=objective,
        checkpoints=checkpoints,
        proof_id=proof_id,
        scope_id=scope_id,
    )
    return _install_bundle(output_dir, bundle)


def _install_bundle(output_dir: str | Path, bundle: ScaffoldBundle) -> ScaffoldPaths:
    """Write an already-rendered bundle's three files, refusing any overwrite.

    Shared by the inert ANC-3 scaffold and the ANC-4 coding-contract compiler
    (``pathlayer.coding_contracts``), so both get the same preflight,
    exclusive-create and rollback behaviour from one implementation.
    """
    destination = Path(output_dir)
    targets = ScaffoldPaths(
        manifest=destination / _MANIFEST_NAME,
        probe=destination / _PROBE_NAME,
        readme=destination / _README_NAME,
    )
    target_content = (
        (targets.manifest, bundle.manifest_json),
        (targets.probe, bundle.probe_python),
        (targets.readme, bundle.readme_markdown),
    )
    existing = [path for path, _ in target_content if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise ScaffoldError(f"refusing to overwrite existing scaffold file(s): {joined}")
    if destination.exists() and not destination.is_dir():
        raise ScaffoldError(f"output directory is not a directory: {destination}")

    destination_was_created = not destination.exists()
    created: list[Path] = []
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".pathlayer-proof-scaffold-", dir=destination.parent
        ) as staging_name:
            staging = Path(staging_name)
            staged = (
                (staging / _MANIFEST_NAME, targets.manifest, bundle.manifest_json),
                (staging / _PROBE_NAME, targets.probe, bundle.probe_python),
                (staging / _README_NAME, targets.readme, bundle.readme_markdown),
            )
            for staged_path, _, content in staged:
                staged_path.write_text(content, encoding="utf-8")
            for staged_path, target, _ in staged:
                with target.open("xb") as output:
                    output.write(staged_path.read_bytes())
                created.append(target)
    except (OSError, UnicodeError) as error:
        for path in reversed(created):
            with contextlib.suppress(OSError):
                path.unlink()
        if destination_was_created:
            with contextlib.suppress(OSError):
                destination.rmdir()
        raise ScaffoldError(f"could not write progress-proof scaffold: {error}") from error
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pathlayer-scaffold-proof",
        description="Generate a deterministic Pathlayer progress-proof scaffold.",
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="required checkpoint id; repeat for each checkpoint",
    )
    parser.add_argument("--proof-id", default=None)
    parser.add_argument("--scope-id", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    try:
        paths = write_progress_proof_scaffold(
            output_dir=args.output_dir,
            task_id=args.task_id,
            name=args.name,
            objective=args.objective,
            checkpoints=args.checkpoints,
            proof_id=args.proof_id,
            scope_id=args.scope_id,
        )
    except ScaffoldError as error:
        parser.error(str(error))
    print(f"created {paths.manifest}")
    print(f"created {paths.probe}")
    print(f"created {paths.readme}")
    return 0


__all__ = [
    "ScaffoldBundle",
    "ScaffoldError",
    "ScaffoldPaths",
    "render_progress_proof_scaffold",
    "write_progress_proof_scaffold",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
