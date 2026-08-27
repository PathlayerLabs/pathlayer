"""Deterministic ``checkpoint_set`` progress contract (ANC-1).

Four ``no_progress`` traces (``no_progress_trivial``, ``replan_churn``,
``research_rabbit_hole``, ``slow_burn_47k`` -- issue #132) carry no native ground
truth: no test results, no exit code, no git diff. The coding anchors in
``pathlayer.detectors.progress`` have nothing to measure there, so
``NoProgressDetector`` stays silent -- correctly, since it never stops an agent
it cannot measure, but there is no signal at all on that class of trace either.

This module gives a domain a deterministic, framework-agnostic way to *define*
what "progress" means for one task (a :class:`ProofSpec`: a named set of
required checkpoints) and *emit* verifiable advancement against it (a
:class:`ProgressProof`), without any text similarity, LLM judgement, or
semantic scoring. Pathlayer standardizes the protocol; the domain remains
responsible for business truth.

Trust boundary: the proof this anchor consumes must originate outside the
observed agent's control -- a self-reported claim is falsifiable. This module
only poses that constraint in the contract and does *structural* integrity
(recomputing ``spec_hash`` from a co-located ``proof_spec`` and cross-checking
it against what ``progress_proof`` claims); it proves nothing about
*provenance*. The hash describes the identity of the rule, not the veracity of
the source. Wiring a concrete producer that genuinely sits outside agent
control -- and validating a spec fail-fast before a real run -- is ANC-2's job,
not this one's. ``ProofProgressAnchor`` is therefore **not** wired into
:func:`~pathlayer.detectors.progress.coding_anchors` or
:func:`~pathlayer.detectors.progress.loop_suppression_anchors`: it changes the
behaviour of zero existing traces until something explicitly constructs and
uses it.

Zero-argument anchor, by design: exactly like ``TestProgressAnchor()`` /
``GitProgressAnchor()`` / ``ExitCodeProgressAnchor()``, a :class:`ProofSpec` is
never passed at construction. It is discovered dynamically from the event
stream and cached by its own recomputed ``spec_hash`` -- the same shape as
``TestProgressAnchor`` discovering per-event ``target``s rather than being
pre-configured with them. This is what makes "a change of ``scope_id`` or
``spec_hash``" a real, live mechanism rather than a permanent rejection: the
anchor tracks one *active* epoch (the most recently advanced ``(scope_id,
spec_hash)`` pair) and can move on to a new one mid-run, e.g. an agent handed a
sequence of related subtasks each with their own checkpoint set.

``checkpoint_set`` semantics are **last-known-value-per-checkpoint**, not a
monotone debt/credit ratchet like NOP-5's ``_RichTargetState``
(``pathlayer.detectors.progress``): a later proof reporting a previously-true
required checkpoint as ``false`` legitimately *lowers* the level. This is a
deliberate divergence from the NOP-5 test-outcome anchor -- a required
checkpoint is a live fact about the task's state (e.g. "the deliverable is
currently accepted"), not a test that, once green, stays credited forever.

A malformed or out-of-order proof is *inert*, never sticky: one degraded event
never erases or resets an established level, mirroring
``_rich_test_result``'s "a malformed rich claim is inert" convention. The one
case that *does* reset (to ``None``, a gap) is a genuine epoch transition -- a
proof for a different ``spec_hash`` than the currently active one -- so
``NoProgressDetector``'s CUSUM re-baselines instead of comparing two unrelated
scales. No code change to ``no_progress.py`` is needed for this: its
``_push_level``/``_cusum`` already treat any anchor's ``None`` as "lost
baseline, reset" (see their docstrings), and this anchor never emits
``NaN``/``inf``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pathlayer.detectors.progress import AnchorSnapshot, AnchorState, ProgressAnchor
from pathlayer.events import ToolEvent
from pathlayer.store import _canonical_sha256

_PROOF_SPEC_KEY = "proof_spec"
_PROGRESS_PROOF_KEY = "progress_proof"

_CHECKPOINT_SET_KIND = "checkpoint_set"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

_SPEC_HASH_ALGORITHM_PREFIX = "sha256:"
_SPEC_HASH_HEX_LENGTH = 64
_SPEC_HASH_HEX_DIGITS = frozenset("0123456789abcdef")

# Untrusted, producer-controlled wire payload -- bounding every axis prevents an
# adversarial or malformed stream from turning immutable per-tick copies into
# unbounded memory growth, mirroring progress.py's own _MAX_* constants for the
# NOP-5 rich-test contract.
_MAX_TASK_ID_LENGTH = 512
_MAX_PROOF_ID_LENGTH = 512
_MAX_SCOPE_ID_LENGTH = 512
_MAX_CHECKPOINT_ID_LENGTH = 256
_MAX_REQUIRED_CHECKPOINTS = 256
_MAX_CHECKPOINTS_PER_PROOF = 256
_MAX_TRACKED_SPEC_HASHES = 256


@dataclass(slots=True, frozen=True)
class ProofSpec:
    """Validated V1 ``proof_spec`` contract (ANC-1). See module docstring."""

    schema_version: int
    task_id: str
    proof_id: str
    scope_id: str
    kind: str
    required: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ProgressProof:
    """Validated V1 ``progress_proof`` contract (ANC-1). See module docstring."""

    spec_hash: str
    sequence: int
    checkpoints: tuple[tuple[str, bool], ...]


def validate_proof_spec(raw: Any) -> ProofSpec | None:
    """Structurally validate a raw ``proof_spec`` mapping; ``None`` on any violation.

    Closed schema: exactly ``{schema_version, task_id, proof_id, scope_id, kind,
    required}`` -- V1 has no optional field. Never raises: malformed,
    producer-controlled input is rejected, not an error. ``required`` is
    returned sorted and deduplicated so :func:`compute_spec_hash` is
    independent of the author's original ordering.
    """
    if not isinstance(raw, Mapping):
        return None
    allowed = {"schema_version", "task_id", "proof_id", "scope_id", "kind", "required"}
    if set(raw) != allowed:
        return None
    schema_version = raw["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)  # bool is an int subclass
        or schema_version not in _SUPPORTED_SCHEMA_VERSIONS
    ):
        return None
    task_id = raw["task_id"]
    proof_id = raw["proof_id"]
    scope_id = raw["scope_id"]
    kind = raw["kind"]
    for value, cap in (
        (task_id, _MAX_TASK_ID_LENGTH),
        (proof_id, _MAX_PROOF_ID_LENGTH),
        (scope_id, _MAX_SCOPE_ID_LENGTH),
    ):
        if not isinstance(value, str) or not value or len(value) > cap:
            return None
    if kind != _CHECKPOINT_SET_KIND:
        return None
    required = raw["required"]
    # A raw string IS a Sequence of characters -- must be excluded explicitly,
    # else "abc" would silently validate as three checkpoints "a", "b", "c".
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        return None
    if not required or len(required) > _MAX_REQUIRED_CHECKPOINTS:
        return None
    seen: set[str] = set()
    normalized: list[str] = []
    for checkpoint_id in required:
        if (
            not isinstance(checkpoint_id, str)
            or not checkpoint_id
            or len(checkpoint_id) > _MAX_CHECKPOINT_ID_LENGTH
            or checkpoint_id in seen
        ):
            return None
        seen.add(checkpoint_id)
        normalized.append(checkpoint_id)
    return ProofSpec(
        schema_version=schema_version,
        task_id=task_id,
        proof_id=proof_id,
        scope_id=scope_id,
        kind=kind,
        required=tuple(sorted(normalized)),
    )


def validate_progress_proof(raw: Any) -> ProgressProof | None:
    """Structurally validate a raw ``progress_proof`` mapping; ``None`` on any violation.

    Closed schema: exactly ``{spec_hash, sequence, checkpoints}``. Checkpoint
    values must be ``bool`` -- never coerced from ``int``/``str``/``float`` --
    so a proof can never smuggle a numeric score in under a checkpoint id: the
    level is always *derived* (count of required checkpoints verified), per the
    contract, not supplied. ``checkpoints`` may legitimately name ids outside
    any ``ProofSpec.required`` -- this validator is spec-agnostic by design
    (structure/type only); filtering to the required subset happens later, in
    :class:`_ProofAnchorState`'s fold.
    """
    if not isinstance(raw, Mapping):
        return None
    allowed = {"spec_hash", "sequence", "checkpoints"}
    if set(raw) != allowed:
        return None
    spec_hash = raw["spec_hash"]
    if not isinstance(spec_hash, str) or not _is_well_formed_spec_hash(spec_hash):
        return None
    sequence = raw["sequence"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        return None
    checkpoints = raw["checkpoints"]
    if (
        not isinstance(checkpoints, Mapping)
        or not checkpoints
        or len(checkpoints) > _MAX_CHECKPOINTS_PER_PROOF
    ):
        return None
    normalized: list[tuple[str, bool]] = []
    for checkpoint_id, state in checkpoints.items():
        if (
            not isinstance(checkpoint_id, str)
            or not checkpoint_id
            or len(checkpoint_id) > _MAX_CHECKPOINT_ID_LENGTH
            or not isinstance(state, bool)
        ):
            return None
        normalized.append((checkpoint_id, state))
    return ProgressProof(
        spec_hash=spec_hash, sequence=sequence, checkpoints=tuple(sorted(normalized))
    )


def _is_well_formed_spec_hash(value: str) -> bool:
    if not value.startswith(_SPEC_HASH_ALGORITHM_PREFIX):
        return False
    digest = value[len(_SPEC_HASH_ALGORITHM_PREFIX) :]
    return len(digest) == _SPEC_HASH_HEX_LENGTH and set(digest) <= _SPEC_HASH_HEX_DIGITS


def compute_spec_hash(spec: ProofSpec) -> str:
    """Canonical content hash of a validated :class:`ProofSpec`.

    Mirrors ``pathlayer.store.configuration_fingerprint``'s canonical-JSON
    approach (sorted keys, compact separators, no ``NaN``). Order-independent
    in ``required`` since it is already sorted on the validated ``ProofSpec``.
    Describes *what the rule is*, never *where the proof came from* -- see the
    module docstring's trust-boundary note.
    """
    digest = _canonical_sha256(
        {
            "schema_version": spec.schema_version,
            "task_id": spec.task_id,
            "proof_id": spec.proof_id,
            "scope_id": spec.scope_id,
            "kind": spec.kind,
            "required": list(spec.required),
        }
    )
    return _SPEC_HASH_ALGORITHM_PREFIX + digest


def _find(event: ToolEvent, key: str) -> Any:
    """Look up ``key`` in ``event.result``, then ``event.metadata``.

    Deliberately resolves each key independently rather than picking one
    whole-mapping "source" the way ``pathlayer.detectors.progress``'s private
    ``_evidence`` does for a single payload (``test_outcomes``, ``exit_code``,
    ...): ``proof_spec`` and ``progress_proof`` are two independent keys that
    may legitimately be attached to different sources by a producer -- e.g. an
    adapter writing ``progress_proof`` into ``event.metadata`` while the
    agent's own tool ``result`` carries an unrelated (or non-``Mapping``, e.g.
    multimodal-list) payload. Resolving per-key means neither half is ever
    silently dropped just because it landed on the other side of one event.
    """
    result = event.result
    if isinstance(result, Mapping) and key in result:
        return result[key]
    if key in event.metadata:
        return event.metadata[key]
    return None


@dataclass(slots=True, frozen=True)
class _ProofEpoch:
    """Last-known-value-per-checkpoint ledger for the one *active* epoch.

    Unlike NOP-5's debt/credit ledger (``_RichTargetState``), there is no
    ratchet: ``states`` always reflects the latest reported boolean per
    required checkpoint id that has ever been observed, so the derived level
    can legitimately fall as well as rise -- see the module docstring.
    """

    spec_hash: str
    required: tuple[str, ...]
    states: tuple[tuple[str, bool], ...] = ()
    last_sequence: int = -1


def _merge_checkpoints(
    states: tuple[tuple[str, bool], ...],
    required: tuple[str, ...],
    checkpoints: tuple[tuple[str, bool], ...],
) -> tuple[tuple[str, bool], ...]:
    """Fold ``checkpoints`` into ``states``, keeping only ids in ``required``."""
    merged = dict(states)
    required_set = set(required)
    for checkpoint_id, value in checkpoints:
        if checkpoint_id in required_set:
            merged[checkpoint_id] = value
    return tuple(sorted(merged.items()))


def _apply_proof(epoch: _ProofEpoch, proof: ProgressProof) -> _ProofEpoch:
    return _ProofEpoch(
        spec_hash=epoch.spec_hash,
        required=epoch.required,
        states=_merge_checkpoints(epoch.states, epoch.required, proof.checkpoints),
        last_sequence=proof.sequence,
    )


def _lookup_spec(
    known_specs: tuple[tuple[str, ProofSpec], ...], spec_hash: str
) -> ProofSpec | None:
    """Find a cached spec by hash. Linear scan is fine: bounded by
    ``_MAX_TRACKED_SPEC_HASHES``, and avoids rebuilding a dict on every event
    just to serve one lookup or one conditional insert."""
    for cached_hash, spec in known_specs:
        if cached_hash == spec_hash:
            return spec
    return None


@dataclass(slots=True, frozen=True)
class _ProofAnchorState(AnchorState):
    """Immutable state for :class:`ProofProgressAnchor` (PERF-2).

    ``known_specs`` is a bounded ``spec_hash -> ProofSpec`` cache populated by
    any event carrying a valid ``proof_spec``, independent of which epoch is
    currently active -- this is what lets a spec be declared once and
    referenced by many later ``progress_proof``-only events. ``active`` is the
    single currently-credited epoch; switching to a different ``spec_hash``
    always discards it and starts a fresh one -- a deliberate simplification
    that can only under-count, never over-count, a producer that oscillates
    between epochs (an earlier epoch's ledger is never resurrected even if its
    ``spec_hash`` reappears later).

    At a given ``sequence`` number for the active epoch, only an *exact*
    repeat is a no-op: any other checkpoint state at that same sequence --
    even one that is purely additive (a previously-unseen checkpoint id, no
    value actually contradicted) -- is rejected wholesale, never merged. A
    revision number is meant to identify one coherent snapshot; a producer
    that has new facts to report must bump ``sequence``, not resend an
    already-used one. Silently merging would let two different reports both
    claim to be "the state at revision N", which is exactly the kind of
    ambiguity ``forward-only abstention, never false progress`` rules out.
    """

    known_specs: tuple[tuple[str, ProofSpec], ...] = ()
    active: _ProofEpoch | None = None

    def _snapshot(self, *, diagnostics: tuple[str, ...] = ()) -> AnchorSnapshot:
        if self.active is None:
            return AnchorSnapshot(level=None, satisfied=False, diagnostics=diagnostics)
        level = float(sum(1 for _, value in self.active.states if value))
        satisfied = level == len(self.active.required)
        return AnchorSnapshot(level=level, satisfied=satisfied, diagnostics=diagnostics)

    def advance(self, event: ToolEvent) -> tuple[_ProofAnchorState, AnchorSnapshot]:
        raw_spec = _find(event, _PROOF_SPEC_KEY)
        raw_proof = _find(event, _PROGRESS_PROOF_KEY)
        if raw_spec is None and raw_proof is None:
            return self, self._snapshot()

        diagnostics: list[str] = []
        known_specs = self.known_specs
        fresh_spec_hash: str | None = None

        if raw_spec is not None:
            spec = validate_proof_spec(raw_spec)
            if spec is None:
                diagnostics.append("malformed proof_spec ignored")
            else:
                fresh_spec_hash = compute_spec_hash(spec)
                if _lookup_spec(known_specs, fresh_spec_hash) is None:
                    if len(known_specs) >= _MAX_TRACKED_SPEC_HASHES:
                        diagnostics.append(
                            f"spec capacity ({_MAX_TRACKED_SPEC_HASHES}) reached; "
                            f"spec_hash {fresh_spec_hash!r} ignored"
                        )
                    else:
                        known_specs = (*known_specs, (fresh_spec_hash, spec))

        def _settle(
            active: _ProofEpoch | None, *, gap: bool = False
        ) -> tuple[_ProofAnchorState, AnchorSnapshot]:
            new_state = _ProofAnchorState(known_specs=known_specs, active=active)
            if gap:
                return new_state, AnchorSnapshot(
                    level=None, satisfied=False, diagnostics=tuple(diagnostics)
                )
            return new_state, new_state._snapshot(diagnostics=tuple(diagnostics))

        if raw_proof is None:
            return _settle(self.active)

        proof = validate_progress_proof(raw_proof)
        if proof is None:
            diagnostics.append("malformed progress_proof ignored")
            return _settle(self.active)

        if fresh_spec_hash is not None and fresh_spec_hash != proof.spec_hash:
            diagnostics.append(
                "proof_spec/progress_proof spec_hash mismatch ignored "
                f"(spec hashes to {fresh_spec_hash!r}, proof claims {proof.spec_hash!r})"
            )
            return _settle(self.active)

        spec = _lookup_spec(known_specs, proof.spec_hash)
        if spec is None:
            diagnostics.append(
                f"spec_hash {proof.spec_hash!r}: no known proof_spec; progress_proof ignored"
            )
            return _settle(self.active)

        active = self.active
        if active is None:
            # True first-ever epoch: report its real baseline immediately, no
            # gap -- there is no prior epoch to lose.
            new_active = _apply_proof(
                _ProofEpoch(spec_hash=proof.spec_hash, required=spec.required), proof
            )
            return _settle(new_active)

        if active.spec_hash != proof.spec_hash:
            # Epoch transition: gap this tick unconditionally so
            # NoProgressDetector's CUSUM resets instead of comparing this
            # epoch's level against the abandoned one's unrelated scale. The
            # new epoch is already active for the *next* tick's baseline.
            new_active = _apply_proof(
                _ProofEpoch(spec_hash=proof.spec_hash, required=spec.required), proof
            )
            return _settle(new_active, gap=True)

        if proof.sequence < active.last_sequence:
            diagnostics.append(
                f"spec_hash {proof.spec_hash!r}: out-of-order sequence {proof.sequence} "
                f"(last accepted {active.last_sequence}) ignored"
            )
            return _settle(active)

        if proof.sequence == active.last_sequence:
            merged = _merge_checkpoints(active.states, active.required, proof.checkpoints)
            if merged != active.states:
                diagnostics.append(
                    f"spec_hash {proof.spec_hash!r}: sequence {proof.sequence} already accepted "
                    "with a different checkpoint state; restatement ignored (bump sequence to "
                    "add new facts)"
                )
            return _settle(active)

        # proof.sequence > active.last_sequence: genuine advancement.
        return _settle(_apply_proof(active, proof))


def _proof_snapshot(history: Sequence[ToolEvent]) -> AnchorSnapshot:
    state = _ProofAnchorState()
    snapshot = AnchorSnapshot(level=None, satisfied=False)
    for event in history:
        state, snapshot = state.advance(event)
    return snapshot


class ProofProgressAnchor(ProgressAnchor):
    """Deterministic, framework-agnostic ``checkpoint_set`` progress anchor (ANC-1).

    Zero-argument constructor: see the module docstring for why ``ProofSpec``
    is never pre-configured but discovered dynamically from the event stream.
    Deliberately **not** wired into
    :func:`~pathlayer.detectors.progress.coding_anchors` or
    :func:`~pathlayer.detectors.progress.loop_suppression_anchors` here --
    wiring a concrete, out-of-agent-control producer is ANC-2; this anchor
    changes the behaviour of zero existing traces until something constructs
    and uses it explicitly.
    """

    name = "progress_proof"

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        return _proof_snapshot(history).level

    def is_satisfied(self, history: Sequence[ToolEvent]) -> bool:
        return _proof_snapshot(history).satisfied

    def diagnose(self, history: Sequence[ToolEvent]) -> tuple[str, ...]:
        return _proof_snapshot(history).diagnostics

    def create_state(self) -> _ProofAnchorState:
        return _ProofAnchorState()

    def step(self, state: Any, event: ToolEvent) -> tuple[Any, AnchorSnapshot]:
        assert isinstance(state, _ProofAnchorState)
        return state.advance(event)
