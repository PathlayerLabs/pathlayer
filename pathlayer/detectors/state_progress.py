"""Opt-in novelty anchor for one explicitly selected output artifact (VAL-5).

A new filesystem fingerprint is evidence of a new state, not proof of task
success. Use only where artifact novelty is an appropriate progress proxy.
This anchor is deliberately absent from the default coding/suppression sets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pathlayer.detectors.progress import AnchorSnapshot, ProgressAnchor
from pathlayer.events import EventKind, ToolEvent
from pathlayer.state_collectors import (
    FILESYSTEM_COLLECTOR_VERSION,
    _filesystem_canonical_fingerprint,
)
from pathlayer.state_evidence import StateOutcome, state_evidence_from_event


@dataclass(frozen=True, slots=True)
class _StateProgress:
    seen: frozenset[str] = frozenset()
    level: int = 0
    exhausted: bool = False
    snapshot: AnchorSnapshot = field(default_factory=lambda: AnchorSnapshot(None, False))


class StateEvidenceProgressAnchor(ProgressAnchor):
    """Count novel after-states on a known filesystem resource, without I/O.

    Feed the same explicit anchor to NoProgressDetector and, when desired,
    Pipeline.progress_anchors with the existing loop suppression rule. Missing,
    invalid, failed, overlapping or differently scoped evidence is a gap, never
    stagnation. Revisited states do not advance. The bounded seen-set fails open
    permanently at capacity rather than evicting old states and crediting cycles.
    No success verdict can be inferred from a file fingerprint.
    """

    name = "state_evidence"

    def __init__(self, resource_id: str, *, max_states: int = 4096) -> None:
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ValueError("resource_id must name the known output artifact")
        if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 2:
            raise ValueError("max_states must be an integer >= 2")
        self.resource_id = resource_id
        self.max_states = max_states

    def create_state(self) -> _StateProgress:
        return _StateProgress()

    def step(self, state: Any, event: ToolEvent) -> tuple[_StateProgress, AnchorSnapshot]:
        if not isinstance(state, _StateProgress):
            raise TypeError("invalid StateEvidenceProgressAnchor state")
        if event.kind is not EventKind.TOOL_CALL:
            return state, state.snapshot
        if state.exhausted:
            return state, state.snapshot
        envelope = state_evidence_from_event(event)
        if envelope is None or envelope.outcome not in {
            StateOutcome.CHANGED,
            StateOutcome.UNCHANGED,
        }:
            return self._gap(state, "state evidence unavailable or invalid")
        # V1 intentionally supports exactly one filesystem sensor. Do not use
        # an aggregate outcome to credit some other resource's changes.
        if len(envelope.records) != 2:
            return self._gap(state, "expected one filesystem state sensor")
        before, after = envelope.records
        # The envelope reader already validates pair identity on replay as well
        # as live input. Keep this consumer's supported scope explicit for both.
        if any(
            record.resource_type != "filesystem"
            or record.resource_id != self.resource_id
            or record.collector_version != FILESYSTEM_COLLECTOR_VERSION
            for record in (before, after)
        ):
            return self._gap(state, "state evidence scope mismatch")
        # The general StateEvidence schema permits bounded JSON fingerprints.
        # This consumer requires the filesystem collector's stronger contract,
        # including agreement with the raw fingerprint, before hashing values.
        try:
            prior = _filesystem_canonical_fingerprint(before)
            current = _filesystem_canonical_fingerprint(after)
        except ValueError:
            return self._gap(state, "invalid filesystem fingerprint")
        seen = state.seen | {prior}
        novel = current not in seen
        seen = seen | {current}
        if len(seen) > self.max_states:
            return self._gap(replace(state, exhausted=True), "state evidence capacity reached")
        level = state.level + int(novel)
        snapshot = AnchorSnapshot(float(level), False)
        return _StateProgress(frozenset(seen), level, False, snapshot), snapshot

    @staticmethod
    def _gap(state: _StateProgress, reason: str) -> tuple[_StateProgress, AnchorSnapshot]:
        snapshot = AnchorSnapshot(None, False, (reason,))
        return replace(state, snapshot=snapshot), snapshot

    def _snapshot(self, history: Sequence[ToolEvent]) -> AnchorSnapshot:
        state = self.create_state()
        for event in history:
            state, _ = self.step(state, event)
        return state.snapshot

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        return self._snapshot(history).level

    def diagnose(self, history: Sequence[ToolEvent]) -> tuple[str, ...]:
        return self._snapshot(history).diagnostics
