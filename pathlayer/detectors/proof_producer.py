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
  degrades to "nothing observed this tick".

ANC-4 makes that degradation explicit, in two places. Every tick that yields
no proof, and every proof that omits a required checkpoint, is logged as a
warning with a fixed category -- never the probe's own output, which can carry
secrets. And the same fact is written into the event, as a
``progress_proof_unobserved`` marker naming the required checkpoints the probe
could not observe and that category: the ANC-1 anchor turns it into a *gap*
(``level=None`` for that tick) instead of silently keeping its last known
level, so a probe outage in the middle of a productive run no longer ends in a
``no_progress`` STOP (see ``pathlayer.detectors.proof``). The producer
therefore always writes *something* after a probe: a full proof, a partial
proof plus a marker, or a marker alone.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathlayer.config import ConfigError, PathlayerConfig
from pathlayer.detectors.proof import (
    _PROGRESS_PROOF_KEY,
    _PROGRESS_PROOF_UNOBSERVED_KEY,
    _PROOF_SPEC_KEY,
    PROOF_METADATA_KEYS,
    ProofSpec,
    compute_spec_hash,
    validate_proof_spec,
)
from pathlayer.events import ToolEvent

logger = logging.getLogger(__name__)

# Bounds the "missing checkpoints" warning on a large contract.
_MAX_LOGGED_CHECKPOINTS = 5

# What one probe run attaches to an event: ``ToolEvent.metadata`` key -> payload.
ProofPayloads = dict[str, dict[str, Any]]


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

    def probe(self) -> ProofPayloads:
        """Run the probe once and return the metadata payloads to attach (never raises).

        Contract: exit code 0 and stdout ``{"checkpoints": {"<id>": bool, ...}}``.
        The result always carries ``proof_spec``, then:

        - every required checkpoint observed: a ``progress_proof``;
        - some observed: a ``progress_proof`` with those, plus a
          ``progress_proof_unobserved`` marker naming the rest;
        - none observed -- any other exit code, a timeout, unreadable output,
          output that is not that shape, or an empty report: the marker alone,
          naming every required checkpoint.

        The marker's ``reason`` is a fixed category (ANC-4), and the same
        category is logged as a warning; the probe's own output is never
        echoed. The exception handling is deliberately wide (``ValueError``
        covers ``json.JSONDecodeError`` and a non-UTF-8-decodable
        ``UnicodeDecodeError`` from ``text=True``, and also the ``timeout=nan``
        case ``subprocess.run`` itself raises on; ``RecursionError`` covers a
        deeply nested stdout): a flaky or misbehaving probe must never
        propagate past this method. Only a successful proof consumes a
        ``sequence`` number.

        Runs the subprocess itself outside :attr:`_lock` -- only bumping
        ``_sequence`` needs to be atomic -- so one slow probe cannot serialize
        every concurrent run sharing this producer behind it.
        """
        observed = self._observe()
        payloads: ProofPayloads = {_PROOF_SPEC_KEY: self._proof_spec_payload()}
        if isinstance(observed, str):
            _log_no_proof(observed)
            payloads[_PROGRESS_PROOF_UNOBSERVED_KEY] = self._marker(self.spec.required, observed)
            return payloads
        checkpoints = observed
        missing = tuple(
            checkpoint for checkpoint in self.spec.required if checkpoint not in checkpoints
        )
        if not checkpoints:
            category = "reported none of the required checkpoints"
            _log_no_proof(category)
            payloads[_PROGRESS_PROOF_UNOBSERVED_KEY] = self._marker(missing, category)
            return payloads
        if missing:
            shown = ", ".join(missing[:_MAX_LOGGED_CHECKPOINTS])
            if len(missing) > _MAX_LOGGED_CHECKPOINTS:
                shown += ", ..."
            category = (
                f"did not report {len(missing)} of {len(self.spec.required)} required checkpoint(s)"
            )
            logger.warning(
                "progress_proof probe %s (%s); no level this tick, their last known values "
                "are kept",
                category,
                shown,
            )
            payloads[_PROGRESS_PROOF_UNOBSERVED_KEY] = self._marker(missing, category)
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        payloads[_PROGRESS_PROOF_KEY] = {
            "spec_hash": self.spec_hash,
            "sequence": sequence,
            "checkpoints": checkpoints,
        }
        return payloads

    def _observe(self) -> dict[str, bool] | str:
        """The required checkpoints the probe reported, or the failure category.

        Returning the category (a ``str``) instead of storing it on ``self``
        keeps the probe re-entrant: concurrent runs share one producer and run
        their probes outside :attr:`_lock`.
        """
        try:
            completed = subprocess.run(
                self.probe_command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {self.timeout_seconds:g}s"
        except subprocess.CalledProcessError as error:
            return f"exited with status {error.returncode}"
        except UnicodeDecodeError:
            return "wrote output that is not UTF-8 text"
        except OSError:
            return "could not be launched"
        except (subprocess.SubprocessError, ValueError):
            return "could not be run"
        try:
            payload = json.loads(completed.stdout)
        except (ValueError, RecursionError):
            return "output is not JSON"
        checkpoints = payload.get("checkpoints") if isinstance(payload, dict) else None
        if not isinstance(checkpoints, dict):
            return "output is not the checkpoints JSON contract"
        # All-or-nothing, matching ANC-1's own validate_progress_proof: a probe
        # bug corrupting *one* entry must reject the whole tick as "nothing
        # observed", not silently drop just the bad entry while crediting the
        # rest. Once every entry is confirmed well-formed, dropping ids outside
        # spec.required is a separate, legitimate filter (ANC-1's validator is
        # spec-agnostic by design -- a checkpoints dict may name ids the spec
        # never required).
        if not all(
            isinstance(checkpoint_id, str) and isinstance(state, bool)
            for checkpoint_id, state in checkpoints.items()
        ):
            return "output is not the checkpoints JSON contract"
        return {
            checkpoint_id: state
            for checkpoint_id, state in checkpoints.items()
            if checkpoint_id in self.spec.required
        }

    def _proof_spec_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.spec.schema_version,
            "task_id": self.spec.task_id,
            "proof_id": self.spec.proof_id,
            "scope_id": self.spec.scope_id,
            "kind": self.spec.kind,
            "required": list(self.spec.required),
        }

    def _marker(self, unobserved: tuple[str, ...], reason: str) -> dict[str, Any]:
        return {"spec_hash": self.spec_hash, "unobserved": list(unobserved), "reason": reason}

    def attach(self, event: ToolEvent) -> ProofPayloads:
        """Probe and write the resulting payloads into ``event.metadata``.

        Writes to ``event.metadata``, not ``event.result``: ``event.result`` is
        the real tool-call result and the only side ANC-1's ``_find`` never
        reads (``pathlayer/detectors/proof.py``) -- a proof there would be the
        agent's own claim. ``proof_spec`` is (re)written on every call, not
        just the first, so the anchor's spec cache never depends on event
        ordering. A probe failure writes the ``progress_proof_unobserved``
        marker instead of nothing, so the anchor sees a gap, not a freeze.

        Returns exactly the ``{metadata key: payload}`` mapping applied so a
        durable caller can journal it -- replaying a raw hook payload alone can
        never reproduce a live probe result, so a caller with a durability
        contract to uphold (``OpenClawRunRecorder``) must persist this return
        value itself rather than re-deriving it from the event later.
        """
        payloads = self.probe()
        for key in PROOF_METADATA_KEYS:
            event.metadata.pop(key, None)
        event.metadata.update(payloads)
        return payloads


def _log_no_proof(category: str) -> None:
    logger.warning("progress_proof probe %s; no proof this tick", category)


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


def producer_from_config(config: PathlayerConfig) -> ProgressProofProducer | None:
    """The producer ``config.detectors.progress_proof`` describes, or ``None`` if disabled.

    For integrations that run the probe themselves -- ``LangGraphAdapter``'s
    ``progress_proof`` argument (ANC-4) -- so they build the exact producer
    ``build_pipeline`` validated, from the same section. Raises
    :class:`ManifestError` like :func:`build_producer`.
    """
    section = config.detectors.progress_proof
    if not section.enabled:
        return None
    return build_producer(section.manifest_path, section.probe_command, section.timeout_seconds)
