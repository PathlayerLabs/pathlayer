"""VAL-7: evidence-gated repetition context, independent of run lifecycle.

Capture journals context; this module only validates recorded facts. No new
threshold or signal is introduced: the wrapped detector keeps its name and
the stronger of its local and cross-run readings (never additive fusion).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.canonical import ToolMasks, canonicalize
from pathlayer.events import EventKind, ToolEvent

SESSION_METADATA_KEY = "pathlayer_session_context"
SESSION_VERSION = 1
MAX_SESSION_EVENTS = 64
MAX_CONTEXT_BYTES = 131072


def continuity_key(event: ToolEvent) -> str | None:
    """Comparable observed state, or abstention. Tool output is never a proof.

    Only a failed call can continue a window, and only into the same error: a
    successful call repeated across turns is usually the user asking again
    (polling, a re-asked question), not the agent stuck retrying.

    Complete unsatisfied checkpoint proofs can establish a baseline. State
    evidence must be unchanged and pass the collector-specific validators.
    When both are supplied both must be usable. Any changed observation breaks
    continuity, conservatively including regressions and irrelevant changes.
    The error joins the key only alongside that evidence, never on its own.
    """
    # Local imports avoid config/store/pipeline import cycles.
    from pathlayer.detectors.progress import _has_test_evidence
    from pathlayer.detectors.proof import (
        PROOF_METADATA_KEYS,
        compute_spec_hash,
        validate_progress_proof,
        validate_proof_spec,
    )
    from pathlayer.detectors.retry import _error_of
    from pathlayer.detectors.state_progress import _resource_keys
    from pathlayer.state_evidence import (
        STATE_EVIDENCE_METADATA_KEY,
        StateOutcome,
        state_evidence_from_event,
    )
    from pathlayer.store import _canonical_sha256

    if event.kind is not EventKind.TOOL_CALL or _has_test_evidence(event):
        return None
    error = _error_of(event)
    if error is None:
        return None
    measured: dict[str, Any] = {}
    if any(key in event.metadata for key in PROOF_METADATA_KEYS):
        spec = validate_proof_spec(event.metadata.get("proof_spec"))
        proof = validate_progress_proof(event.metadata.get("progress_proof"))
        if (
            "progress_proof_unobserved" in event.metadata
            or spec is None
            or proof is None
            or proof.spec_hash != compute_spec_hash(spec)
        ):
            return None
        checkpoints = dict(proof.checkpoints)
        if not all(key in checkpoints for key in spec.required):
            return None
        if all(checkpoints[key] for key in spec.required):
            return None
        measured["proof"] = [proof.spec_hash, [(k, checkpoints[k]) for k in spec.required]]
    if STATE_EVIDENCE_METADATA_KEY in event.metadata:
        envelope = state_evidence_from_event(event)
        if envelope is None or envelope.outcome is not StateOutcome.UNCHANGED:
            return None
        keys = []
        for before, after in zip(envelope.records[::2], envelope.records[1::2], strict=True):
            pair = _resource_keys(before, after)
            if pair is None or pair[0] != pair[1]:
                return None
            keys.append([before.collector_version, pair[0]])
        measured["state"] = sorted(keys)
    if not measured:
        return None
    measured["error"] = error
    return _canonical_sha256(measured)


def without_context(event: ToolEvent) -> ToolEvent:
    return replace(
        event, metadata={k: v for k, v in event.metadata.items() if k != SESSION_METADATA_KEY}
    )


def session_history(event: ToolEvent, masks: ToolMasks | None = None) -> list[ToolEvent]:
    """Validate bounded prior observations before letting them affect a signal."""
    from pathlayer.store import StateStoreError, decode_event

    raw = event.metadata.get(SESSION_METADATA_KEY)
    if (
        not isinstance(raw, dict)
        or type(raw.get("version")) is not int
        or raw.get("version") != SESSION_VERSION
        or raw.get("unavailable")
    ):
        return []
    rows = raw.get("observations")
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_SESSION_EVENTS:
        return []
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return []
    if any(
        not isinstance(raw.get(k), str) or not raw[k].strip() or len(raw[k]) > 512
        for k in ("source", "session_key")
    ):
        return []
    key = continuity_key(event)
    if key is None:
        return []
    history: list[ToolEvent] = []
    runs: set[str] = set()
    identities: set[tuple[str, str]] = set()
    sequences: dict[str, int] = {}
    total = 0
    previous_index = -1
    try:
        for row in rows:
            if not isinstance(row, dict):
                return []
            payload, row_run, index = row["event"], row["run_id"], row["index"]
            event_id = row["event_id"]
            if (
                not isinstance(payload, str)
                or not isinstance(row_run, str)
                or not row_run
                or not isinstance(event_id, str)
                or not event_id
                or (row_run, event_id) in identities
                or type(index) is not int
                or index <= previous_index
            ):
                return []
            total += len(payload.encode("utf-8"))
            if total > MAX_CONTEXT_BYTES:
                return []
            prior = decode_event(payload)
            if SESSION_METADATA_KEY in prior.metadata:
                return []
            if continuity_key(prior) != key or canonicalize(prior, masks) != canonicalize(
                event, masks
            ):
                return []
            proof = prior.metadata.get("progress_proof")
            if isinstance(proof, dict):
                sequence = proof["sequence"]
                if sequence <= sequences.get(row_run, -1):
                    return []
                sequences[row_run] = sequence
            history.append(prior)
            identities.add((row_run, event_id))
            runs.add(row_run)
            previous_index = index
        if rows[-1]["run_id"] != run_id or len(runs) < 2:
            return []
        if history[-1] != without_context(event):
            return []
    except (KeyError, TypeError, ValueError, StateStoreError):
        return []
    return history


class SessionAwareDetector(Detector):
    """Extend a repetition detector only with verified cross-run observations."""

    def __init__(self, detector: Detector, masks: ToolMasks | None = None) -> None:
        self.detector = detector
        self.name = detector.name
        self.event_kinds = detector.event_kinds
        self.masks = masks

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        local = self.detector.evaluate(history)
        relevant = self._relevant(history)
        if not relevant:
            return local
        context = session_history(relevant[-1], self.masks)
        if not context:
            return local
        from pathlayer.policies.base import _normalized_confidence

        stitched = self.detector.evaluate(context)
        # Same normalization as the policy use sites: a NaN reading must never
        # displace a valid local signal.
        if _normalized_confidence(stitched.confidence) <= _normalized_confidence(local.confidence):
            return local
        return replace(stitched, reason=f"session continuity: {stitched.reason}")
