"""ANC-2: concrete, out-of-agent-control producer for the ANC-1 proof contract.

ANC-1 (``pathlayer.detectors.proof``) defines the ``ProofSpec``/``ProgressProof``
schema and the pure ``ProofProgressAnchor``, but wires nothing: without a
producer, the anchor never observes a ``progress_proof`` payload and stays
inert. This module is the trust boundary ANC-1 deliberately left open --
loading and validating an operator-supplied manifest (fail-fast, at build
time) and running an executable probe (fail-*soft*, at capture time, never
inside the pure detection loop) to populate ``ToolEvent.metadata``.

Two failure philosophies, matching two different moments:

- Build time (:func:`load_proof_spec`, :func:`build_producer`): a malformed
  manifest is an operator mistake caught before a run starts, so it raises
  :class:`ManifestError`, exactly like a malformed detector section raises
  :class:`~pathlayer.config.ConfigError` in ``build_pipeline``.
- Capture time (:meth:`ProgressProofProducer.probe`): a probe hiccup (missing
  binary, timeout, garbled stdout) is not a reason to interrupt a live agent
  run that the HTTP ingest path (SEC-3) would otherwise swallow anyway -- it
  degrades to "no proof this tick", which the ANC-1 anchor already treats as
  inert by construction.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathlayer.config import ConfigError
from pathlayer.detectors.proof import (
    _PROGRESS_PROOF_KEY,
    _PROOF_SPEC_KEY,
    ProofSpec,
    compute_spec_hash,
    validate_proof_spec,
)
from pathlayer.events import ToolEvent


class ManifestError(ConfigError):
    """Raised when an ANC-2 ``ProofSpec`` manifest is missing or malformed."""


def load_proof_spec(path: Path) -> ProofSpec:
    """Load and validate a ``ProofSpec`` manifest from disk. Never returns ``None``.

    Called only at build time (never per-event): a missing file, invalid
    JSON, or a payload :func:`~pathlayer.detectors.proof.validate_proof_spec`
    rejects all raise :class:`ManifestError` naming ``path`` and the reason,
    so a bad manifest fails the build, not the run.
    """
    try:
        # UnicodeDecodeError (a non-UTF-8 manifest) is not an OSError -- must
        # be caught explicitly or it escapes as a raw exception instead of
        # ManifestError/ConfigError, breaking build-time fail-fast.
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ManifestError(f"cannot read proof_spec manifest {path}: {error}") from error
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ManifestError(f"proof_spec manifest {path} is not valid JSON: {error}") from error
    spec = validate_proof_spec(raw)
    if spec is None:
        raise ManifestError(f"proof_spec manifest {path} failed schema validation")
    return spec


@dataclass(slots=True)
class ProgressProofProducer:
    """Runs an operator-supplied probe at capture time to populate a proof.

    Constructed once at build time from an already-validated :class:`ProofSpec`
    and its ``spec_hash``. ``_sequence`` is a monotonically increasing counter,
    incremented once per successful probe invocation, shared by every run this
    producer is attached to -- the ANC-1 anchor only requires ``sequence`` to
    strictly increase within one active epoch, not that it start at any
    particular value, so one producer instance safely serves multiple
    concurrent runs. ``_lock`` guards that counter and the probe invocation
    itself, since concurrent runs each hold their own per-run lock elsewhere
    and could otherwise call :meth:`attach` from different threads at once.
    """

    spec: ProofSpec
    spec_hash: str
    probe_command: tuple[str, ...]
    timeout_seconds: float
    _sequence: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def probe(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Run the probe once; ``None`` on any failure (never raises).

        Contract: exit code 0 and stdout ``{"checkpoints": {"<id>": bool, ...}}``.
        Any other exit code, a timeout, unreadable output, or output that is
        not that shape is treated as "no proof this tick" -- the same
        no-file/no-command/no-clock/no-LLM discipline ANC-1 enforces inside the
        detection loop simply means a flaky *producer* must not propagate into
        one, not that failures are hidden from operators (they are still
        loggable by the caller if desired). The exception tuple is
        deliberately wide (``ValueError`` covers ``json.JSONDecodeError`` and a
        non-UTF-8-decodable ``UnicodeDecodeError`` from ``text=True``, and also
        the ``timeout=nan`` case ``subprocess.run`` itself raises on): a flaky
        or misbehaving probe must never propagate past this method.

        Runs the subprocess itself outside :attr:`_lock` -- only bumping
        ``_sequence`` needs to be atomic -- so one slow probe cannot serialize
        every concurrent run sharing this producer behind it.
        """
        try:
            completed = subprocess.run(
                self.probe_command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        checkpoints = payload.get("checkpoints") if isinstance(payload, dict) else None
        if not isinstance(checkpoints, dict):
            return None
        # All-or-nothing, matching ANC-1's own validate_progress_proof: a probe
        # bug corrupting *one* entry must reject the whole tick as "no proof",
        # not silently drop just the bad entry while crediting the rest. Once
        # every entry is confirmed well-formed, dropping ids outside
        # spec.required is a separate, legitimate filter (ANC-1's validator is
        # spec-agnostic by design -- a checkpoints dict may name ids the spec
        # never required).
        if not all(
            isinstance(checkpoint_id, str) and isinstance(state, bool)
            for checkpoint_id, state in checkpoints.items()
        ):
            return None
        filtered = {
            checkpoint_id: state
            for checkpoint_id, state in checkpoints.items()
            if checkpoint_id in self.spec.required
        }
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        proof_spec_payload = {
            "schema_version": self.spec.schema_version,
            "task_id": self.spec.task_id,
            "proof_id": self.spec.proof_id,
            "scope_id": self.spec.scope_id,
            "kind": self.spec.kind,
            "required": list(self.spec.required),
        }
        progress_proof_payload = {
            "spec_hash": self.spec_hash,
            "sequence": sequence,
            "checkpoints": filtered,
        }
        return proof_spec_payload, progress_proof_payload

    def attach(self, event: ToolEvent) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Probe and, on success, populate ``event.metadata`` with both payloads.

        Writes to ``event.metadata``, not ``event.result``: ``event.result`` is
        the real tool-call result and must not be overwritten by an unrelated
        producer, matching ANC-1's ``_find`` lookup
        (``pathlayer/detectors/proof.py``), which already anticipates an
        adapter populating these two keys on ``event.metadata``. ``proof_spec``
        is (re)written on every call, not just the first, so the anchor's
        spec cache never depends on event ordering.

        Returns the applied ``(proof_spec, progress_proof)`` pair (or ``None``
        on probe failure) so a durable caller can journal exactly what was
        attached -- replaying a raw hook payload alone can never reproduce a
        live probe result, so a caller with a durability contract to uphold
        (``OpenClawRunRecorder``) must persist this return value itself rather
        than re-deriving it from the event later.
        """
        result = self.probe()
        if result is None:
            return None
        proof_spec_payload, progress_proof_payload = result
        event.metadata[_PROOF_SPEC_KEY] = proof_spec_payload
        event.metadata[_PROGRESS_PROOF_KEY] = progress_proof_payload
        return result


def build_producer(
    manifest_path: str, probe_command: tuple[str, ...], timeout_seconds: float
) -> ProgressProofProducer:
    """Load, validate and hash the manifest, then construct a producer.

    Called only when the ANC-2 config section is enabled. Raises
    :class:`ManifestError` (a :class:`~pathlayer.config.ConfigError`) on any
    manifest problem, so a caller wrapping detector construction in a
    ``ConfigError``-normalizing ``try``/``except`` (as ``build_pipeline``
    already does) gets consistent fail-fast behaviour for free.
    """
    spec = load_proof_spec(Path(manifest_path))
    spec_hash = compute_spec_hash(spec)
    return ProgressProofProducer(
        spec=spec,
        spec_hash=spec_hash,
        probe_command=probe_command,
        timeout_seconds=timeout_seconds,
    )
