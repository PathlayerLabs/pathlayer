"""Opt-in novelty anchors over recorded ``StateEvidence``.

A new fingerprint is evidence of a new state, not proof of task success. Use
these anchors only where state novelty is an appropriate progress proxy. They
are deliberately absent from the default coding/suppression sets.

:class:`StateEvidenceProgressAnchor` watches one explicitly selected
filesystem artifact observed around every tool call.
:class:`CodingStateProgressAnchor` watches whatever the coding preset
planned for each call -- a set of resources that changes from call to call --
and so has to answer two questions VAL-5 never faced:

- **a call that claims no state.** The preset abstains on reads and unknown
  tools, so those events carry no envelope. Treating them as gaps would reset
  the detector's patience *and* deny loop suppression its baseline; treating
  them as flat progress would count a healthy exploration phase as stagnation.
  The event is therefore neither, and which one it leans to is the caller's
  choice: ``unclaimed="gap"`` (never invent stagnation from a read) for
  ``NoProgressDetector``, ``unclaimed="hold"`` (keep the last level so a later
  novel state can still register as advancing) for loop suppression. An
  ``unavailable`` or failed observation stays a gap under both.
- **objective evidence.** Novelty is an *effect*. Under
  ``NoProgressDetector``'s optimistic aggregation, an agent producing a fresh
  wrong state on every call would otherwise keep setting records and silence a
  stalled ANC-4 contract. With ``defer_to_objective``, the anchor stops
  measuring for the rest of the run as soon as test results or progress proofs
  appear, exactly as ``GitProgressAnchor`` defers to tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pathlayer.coding_state_collectors import (
    COMMAND_COLLECTOR_VERSION,
    GIT_COLLECTOR_VERSION,
    command_outcome_of,
    git_tree_of,
)
from pathlayer.detectors.progress import AnchorSnapshot, ProgressAnchor, _has_test_evidence
from pathlayer.detectors.proof import PROOF_METADATA_KEYS
from pathlayer.detectors.state_evidence import StateEvidence
from pathlayer.events import EventKind, ToolEvent
from pathlayer.state_collectors import (
    FILESYSTEM_COLLECTOR_VERSION,
    PYTEST_COLLECTOR_VERSION,
    _filesystem_canonical_fingerprint,
    _test_outcomes,
)
from pathlayer.state_evidence import StateOutcome, state_evidence_from_event
from pathlayer.store import _canonical_sha256

UNCLAIMED_GAP = "gap"
UNCLAIMED_HOLD = "hold"
_UNCLAIMED_MODES = (UNCLAIMED_GAP, UNCLAIMED_HOLD)


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


@dataclass(frozen=True, slots=True)
class _CodingStateProgress:
    seen: frozenset[str] = frozenset()
    level: int = 0
    observed: bool = False
    exhausted: bool = False
    deferred: bool = False
    snapshot: AnchorSnapshot = field(default_factory=lambda: AnchorSnapshot(None, False))


class CodingStateProgressAnchor(ProgressAnchor):
    """Count novel states across whatever resources each call was observed on.

    Reads only recorded evidence, so it does the same thing live and in replay.
    Every resource the coding preset supports is honoured, and each is tracked
    separately: a file returning to a content already seen never earns progress
    even if another resource is at a fresh state. One capture that failed, was
    unavailable, or carries a fingerprint its collector's own contract rejects
    turns the whole event into a gap -- crediting the observed half of a
    partial observation would attribute progress no one measured.

    Never satisfied: a new state is not a reached objective, and this anchor
    has no way to know one. Never ``is_unavailable``: it holds no other anchor
    silent, unlike an ANC-4 proof that lost its measurement.
    """

    name = "coding_state"

    def __init__(
        self,
        *,
        unclaimed: str = UNCLAIMED_GAP,
        defer_to_objective: bool = False,
        max_states: int = 4096,
    ) -> None:
        if unclaimed not in _UNCLAIMED_MODES:
            raise ValueError(f"unclaimed must be one of {', '.join(_UNCLAIMED_MODES)}")
        if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 2:
            raise ValueError("max_states must be an integer >= 2")
        self.unclaimed = unclaimed
        self.defer_to_objective = defer_to_objective
        self.max_states = max_states

    def create_state(self) -> _CodingStateProgress:
        return _CodingStateProgress()

    def step(self, state: Any, event: ToolEvent) -> tuple[_CodingStateProgress, AnchorSnapshot]:
        if not isinstance(state, _CodingStateProgress):
            raise TypeError("invalid CodingStateProgressAnchor state")
        if event.kind is not EventKind.TOOL_CALL:
            return state, state.snapshot
        if state.deferred or state.exhausted:
            # Both are permanent, and both were already explained on the event
            # that caused them: keep measuring nothing, quietly.
            return state, state.snapshot
        if self.defer_to_objective and _carries_objective_evidence(event):
            # Sticky, like GitProgressAnchor's deference to tests: an effect
            # must never outvote objective evidence, and must not flip back to
            # outvoting it on the next call that carries none.
            return replace(state, deferred=True, snapshot=AnchorSnapshot(None, False)), (
                AnchorSnapshot(None, False, ("deferring to objective evidence",))
            )
        envelope = state_evidence_from_event(event)
        if envelope is None:
            return self._unclaimed(state)
        if envelope.outcome not in {StateOutcome.CHANGED, StateOutcome.UNCHANGED}:
            return self._gap(state, f"state evidence {envelope.outcome.value}")
        records = envelope.records
        seen = set(state.seen)
        novel = 0
        for prior, current in zip(records[::2], records[1::2], strict=True):
            keys = _resource_keys(prior, current)
            if keys is None:
                return self._gap(state, "unsupported or invalid state evidence")
            before_key, after_key = keys
            seen.add(before_key)
            if after_key not in seen:
                novel += 1
            seen.add(after_key)
        if len(seen) > self.max_states:
            # Fail open permanently rather than evict: forgetting an old state
            # would let a cycle through it look like new progress.
            return replace(state, exhausted=True, snapshot=AnchorSnapshot(None, False)), (
                AnchorSnapshot(None, False, ("state evidence capacity reached",))
            )
        level = state.level + novel
        snapshot = AnchorSnapshot(float(level), False)
        return self._settle(
            replace(state, seen=frozenset(seen), level=level, observed=True), snapshot
        )

    def _unclaimed(
        self, state: _CodingStateProgress
    ) -> tuple[_CodingStateProgress, AnchorSnapshot]:
        if self.unclaimed == UNCLAIMED_HOLD and state.observed:
            return self._settle(state, AnchorSnapshot(float(state.level), False))
        return self._settle(state, AnchorSnapshot(None, False))

    def _gap(
        self, state: _CodingStateProgress, reason: str
    ) -> tuple[_CodingStateProgress, AnchorSnapshot]:
        return self._settle(state, AnchorSnapshot(None, False, (reason,)))

    @staticmethod
    def _settle(
        state: _CodingStateProgress, snapshot: AnchorSnapshot
    ) -> tuple[_CodingStateProgress, AnchorSnapshot]:
        return replace(state, snapshot=snapshot), snapshot

    def _snapshot(self, history: Sequence[ToolEvent]) -> AnchorSnapshot:
        """Fold the whole prefix and keep the snapshot the last event produced.

        The *produced* snapshot, not the stored one: a permanent condition is
        explained on the event that caused it and stays quiet afterwards, so
        ``measure``/``diagnose`` must report what ``step`` reported for the
        same prefix -- the PERF-2 equivalence this anchor is tested against.
        """
        state = self.create_state()
        snapshot = AnchorSnapshot(None, False)
        for event in history:
            state, snapshot = self.step(state, event)
        return snapshot

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        return self._snapshot(history).level

    def diagnose(self, history: Sequence[ToolEvent]) -> tuple[str, ...]:
        return self._snapshot(history).diagnostics


def _carries_objective_evidence(event: ToolEvent) -> bool:
    """Whether this event carries evidence about the objective, not just an effect."""
    return _has_test_evidence(event) or any(key in event.metadata for key in PROOF_METADATA_KEYS)


def _resource_keys(prior: StateEvidence, current: StateEvidence) -> tuple[str, str] | None:
    """Comparable identities for one before/after pair, or ``None`` if unusable.

    The envelope reader already checked that the pair describes one resource in
    the right order. What is checked here is the collector's own stronger
    contract: the generic schema allows any bounded JSON fingerprint, and a
    fingerprint that does not agree with its raw observation must never be
    compared as if it did.
    """
    validator = _COLLECTOR_CONTRACTS.get((prior.resource_type, prior.collector_version))
    if validator is None:
        return None
    try:
        validator(prior)
        validator(current)
    except ValueError:
        return None
    identity = f"{prior.resource_type}\x00{prior.resource_id}\x00"
    return (
        identity + _canonical_sha256(prior.canonical_fingerprint),
        identity + _canonical_sha256(current.canonical_fingerprint),
    )


# Every collector the preset can plan, with the check that its recorded
# fingerprint is internally consistent. An unlisted collector abstains.
_COLLECTOR_CONTRACTS = {
    ("filesystem", FILESYSTEM_COLLECTOR_VERSION): _filesystem_canonical_fingerprint,
    ("tests", PYTEST_COLLECTOR_VERSION): _test_outcomes,
    ("git", GIT_COLLECTOR_VERSION): git_tree_of,
    ("command", COMMAND_COLLECTOR_VERSION): command_outcome_of,
}
