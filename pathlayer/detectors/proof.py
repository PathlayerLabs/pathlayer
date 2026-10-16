"""Deterministic ``checkpoint_set`` progress contract (ANC-1).

Four ``no_progress`` traces (``no_progress_trivial``, ``replan_churn``,
``research_rabbit_hole``, ``slow_burn_47k``) carry no native ground
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
enforces one structural half of that boundary: both keys are read from
``ToolEvent.metadata`` only, never from ``ToolEvent.result`` (the agent's own
tool output, see :func:`_find`), and it does *structural* integrity
(recomputing ``spec_hash`` from a co-located ``proof_spec`` and cross-checking
it against what ``progress_proof`` claims); it proves nothing about
*provenance* beyond that. The hash describes the identity of the rule, not
the veracity of the source. Wiring a concrete producer that genuinely sits
outside agent control -- and validating a spec fail-fast before a real run --
is ANC-2's job, not this one's. ``ProofProgressAnchor`` is therefore **not** wired into
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
``_rich_test_result``'s "a malformed rich claim is inert" convention. Two
cases *do* emit a gap (a ``None`` level for that one tick), so
``NoProgressDetector``'s CUSUM re-baselines instead of counting the tick as
one more step without progress:

- a genuine epoch transition -- a proof for a different ``spec_hash`` than the
  currently active one -- where the two levels live on unrelated scales;
- an explicit ``progress_proof_unobserved`` marker (ANC-4): the producer ran
  its probe and could *not* observe one or more required checkpoints (probe
  timed out, test module no longer importable, interpreter gone...). Without
  the marker the anchor kept its last known level and a probe outage of ~10
  tool calls in the middle of a productive run ended in a STOP -- the
  "freeze" ANC-4 initially documented as a known limit. A failed measurement
  must never, on its own, stop an agent: this is the same rule VAL-5's
  ``StateEvidenceProgressAnchor`` applies to unavailable evidence. The marker
  carries no ``sequence`` and never switches epoch: the checkpoint ledger is
  kept intact, a marker for another ``spec_hash`` is inert, and a proof that
  arrives in the same event (a *partial* observation) is folded into the
  ledger before the gap is emitted.

A marker for the epoch being measured also flags the snapshot
``unavailable`` (:meth:`ProofProgressAnchor.is_unavailable`), and
``NoProgressDetector`` stays silent on that tick whatever its other anchors
say. A plain gap would drop this anchor from the optimistic aggregation, so
with ``build_pipeline``'s coding anchors alongside, a long-flat exit-code or
test level would decide alone the moment the probe failed -- stopping, on
the first unobserved call, an agent whose progress or reached objective only
the proof could see. A marker before any epoch (a probe that never worked)
leaves the anchor merely inapplicable: the other anchors decide as without a
contract. The price is recall during the outage: an agent that stays stuck
while its probe keeps failing is left to the other detectors.

A gap changes nothing in ``no_progress.py``'s patience bookkeeping: its
``_push_level``/``_cusum`` already treat any anchor's ``None`` as "lost
baseline, reset" (see their docstrings), and this anchor never emits
``NaN``/``inf``. A tick with no producer payload at all (no proof, no marker)
is unchanged: the last known level is reported, as for any anchor between two
observations.
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
_PROGRESS_PROOF_UNOBSERVED_KEY = "progress_proof_unobserved"
# Every metadata key the ANC-2 producer may write; adapters strip exactly these.
PROOF_METADATA_KEYS = (_PROOF_SPEC_KEY, _PROGRESS_PROOF_KEY, _PROGRESS_PROOF_UNOBSERVED_KEY)

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
_MAX_UNOBSERVED_REASON_LENGTH = 256
# Bounds the checkpoint ids echoed in one gap diagnostic (OBS-1 logs them).
_MAX_DIAGNOSED_CHECKPOINTS = 5


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


@dataclass(slots=True, frozen=True)
class UnobservedMarker:
    """Validated ``progress_proof_unobserved`` marker (ANC-4). See module docstring.

    ``unobserved`` names the required checkpoints the probe could not observe
    this tick; ``reason`` is the producer's fixed-vocabulary category, never
    the probe's own output.
    """

    spec_hash: str
    unobserved: tuple[str, ...]
    reason: str


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


def validate_unobserved_marker(raw: Any) -> UnobservedMarker | None:
    """Structurally validate a raw ``progress_proof_unobserved`` mapping; ``None`` if invalid.

    Closed schema: exactly ``{spec_hash, unobserved, reason}``. Deliberately
    carries no ``sequence``: a marker is not a revision of the checkpoint
    ledger, only a statement that this tick measured nothing for the listed
    ids. Never raises.
    """
    if not isinstance(raw, Mapping):
        return None
    if set(raw) != {"spec_hash", "unobserved", "reason"}:
        return None
    spec_hash = raw["spec_hash"]
    if not isinstance(spec_hash, str) or not _is_well_formed_spec_hash(spec_hash):
        return None
    unobserved = raw["unobserved"]
    if not isinstance(unobserved, Sequence) or isinstance(unobserved, (str, bytes)):
        return None
    if not unobserved or len(unobserved) > _MAX_REQUIRED_CHECKPOINTS:
        return None
    seen: set[str] = set()
    for checkpoint_id in unobserved:
        if (
            not isinstance(checkpoint_id, str)
            or not checkpoint_id
            or len(checkpoint_id) > _MAX_CHECKPOINT_ID_LENGTH
            or checkpoint_id in seen
        ):
            return None
        seen.add(checkpoint_id)
    reason = raw["reason"]
    if not isinstance(reason, str) or not reason or len(reason) > _MAX_UNOBSERVED_REASON_LENGTH:
        return None
    return UnobservedMarker(spec_hash=spec_hash, unobserved=tuple(sorted(seen)), reason=reason)


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
    """Look up ``key`` in ``event.metadata`` -- and only there.

    ``event.result`` is the observed agent's own tool output; ``event.metadata``
    is what an adapter or the ANC-2 producer writes from *outside* the agent's
    control. Reading the proof from ``result`` (as this helper once did, and
    before ``metadata``) let a tool that merely returned
    ``{"progress_proof": {...}}`` replace the producer's observation with a
    satisfied one and silence ``no_progress`` for the rest of the run -- the
    ``spec_hash`` it needs is a deterministic function of the manifest. The
    same boundary ARCH-2 applies to ``pathlayer_state_evidence`` applies here:
    a self-reported claim is not a proof.

    Each key is still resolved independently (``proof_spec`` and
    ``progress_proof`` are two independent keys), just never across sources.
    """
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
        raw_marker = _find(event, _PROGRESS_PROOF_UNOBSERVED_KEY)
        if raw_spec is None and raw_proof is None and raw_marker is None:
            return self, self._snapshot()

        diagnostics: list[str] = []
        known_specs, fresh_spec_hash = self._register_spec(raw_spec, diagnostics)
        active, gap = self._fold_proof(raw_proof, known_specs, fresh_spec_hash, diagnostics)
        marker_applies = _apply_marker(raw_marker, active, diagnostics)
        # Only an epoch that was being measured can lose its measurement; a
        # marker before any epoch (a probe that never worked) leaves the anchor
        # merely inapplicable, so the other anchors decide as without it.
        unavailable = marker_applies and active is not None

        new_state = _ProofAnchorState(known_specs=known_specs, active=active)
        if gap or marker_applies:
            return new_state, AnchorSnapshot(
                level=None,
                satisfied=False,
                diagnostics=tuple(diagnostics),
                unavailable=unavailable,
            )
        return new_state, new_state._snapshot(diagnostics=tuple(diagnostics))

    def _register_spec(
        self, raw_spec: Any, diagnostics: list[str]
    ) -> tuple[tuple[tuple[str, ProofSpec], ...], str | None]:
        """Cache a valid co-located ``proof_spec``: ``(known_specs, its hash or None)``.

        The hash is returned even when the cache is full or already holds it,
        so a co-located ``progress_proof`` can still be cross-checked against it.
        """
        known_specs = self.known_specs
        if raw_spec is None:
            return known_specs, None
        spec = validate_proof_spec(raw_spec)
        if spec is None:
            diagnostics.append("malformed proof_spec ignored")
            return known_specs, None
        spec_hash = compute_spec_hash(spec)
        if _lookup_spec(known_specs, spec_hash) is not None:
            return known_specs, spec_hash
        if len(known_specs) >= _MAX_TRACKED_SPEC_HASHES:
            diagnostics.append(
                f"spec capacity ({_MAX_TRACKED_SPEC_HASHES}) reached; "
                f"spec_hash {spec_hash!r} ignored"
            )
            return known_specs, spec_hash
        return (*known_specs, (spec_hash, spec)), spec_hash

    def _fold_proof(
        self,
        raw_proof: Any,
        known_specs: tuple[tuple[str, ProofSpec], ...],
        fresh_spec_hash: str | None,
        diagnostics: list[str],
    ) -> tuple[_ProofEpoch | None, bool]:
        """Fold one ``progress_proof`` into the active epoch: ``(active, gap)``."""
        active = self.active
        if raw_proof is None:
            return active, False

        proof = validate_progress_proof(raw_proof)
        if proof is None:
            diagnostics.append("malformed progress_proof ignored")
            return active, False

        if fresh_spec_hash is not None and fresh_spec_hash != proof.spec_hash:
            diagnostics.append(
                "proof_spec/progress_proof spec_hash mismatch ignored "
                f"(spec hashes to {fresh_spec_hash!r}, proof claims {proof.spec_hash!r})"
            )
            return active, False

        spec = _lookup_spec(known_specs, proof.spec_hash)
        if spec is None:
            diagnostics.append(
                f"spec_hash {proof.spec_hash!r}: no known proof_spec; progress_proof ignored"
            )
            return active, False

        if active is None:
            # True first-ever epoch: report its real baseline immediately, no
            # gap -- there is no prior epoch to lose.
            fresh = _ProofEpoch(spec_hash=proof.spec_hash, required=spec.required)
            return _apply_proof(fresh, proof), False

        if active.spec_hash != proof.spec_hash:
            # Epoch transition: gap this tick unconditionally so
            # NoProgressDetector's CUSUM resets instead of comparing this
            # epoch's level against the abandoned one's unrelated scale. The
            # new epoch is already active for the *next* tick's baseline.
            fresh = _ProofEpoch(spec_hash=proof.spec_hash, required=spec.required)
            return _apply_proof(fresh, proof), True

        if proof.sequence < active.last_sequence:
            diagnostics.append(
                f"spec_hash {proof.spec_hash!r}: out-of-order sequence {proof.sequence} "
                f"(last accepted {active.last_sequence}) ignored"
            )
            return active, False

        if proof.sequence == active.last_sequence:
            merged = _merge_checkpoints(active.states, active.required, proof.checkpoints)
            if merged != active.states:
                diagnostics.append(
                    f"spec_hash {proof.spec_hash!r}: sequence {proof.sequence} already accepted "
                    "with a different checkpoint state; restatement ignored (bump sequence to "
                    "add new facts)"
                )
            return active, False

        # proof.sequence > active.last_sequence: genuine advancement.
        return _apply_proof(active, proof), False


def _apply_marker(raw_marker: Any, active: _ProofEpoch | None, diagnostics: list[str]) -> bool:
    """Whether a ``progress_proof_unobserved`` marker applies to this tick (a gap).

    Evaluated against the epoch that is active *after* any co-located proof
    was folded, so a partial observation (proof + marker in one event) first
    updates the ledger, then gaps. Never touches the ledger or the epoch: a
    marker for another ``spec_hash`` is inert, with a diagnostic. With no
    active epoch the level is ``None`` regardless; the diagnostic still says
    why nothing was measured.
    """
    if raw_marker is None:
        return False
    marker = validate_unobserved_marker(raw_marker)
    if marker is None:
        diagnostics.append("malformed progress_proof_unobserved ignored")
        return False
    if active is not None and active.spec_hash != marker.spec_hash:
        diagnostics.append(
            f"spec_hash {marker.spec_hash!r}: progress_proof_unobserved for another epoch ignored"
        )
        return False
    shown = ", ".join(marker.unobserved[:_MAX_DIAGNOSED_CHECKPOINTS])
    if len(marker.unobserved) > _MAX_DIAGNOSED_CHECKPOINTS:
        shown += ", ..."
    diagnostics.append(
        f"spec_hash {marker.spec_hash!r}: {len(marker.unobserved)} required checkpoint(s) "
        f"unobserved ({marker.reason}): {shown}; gap this tick"
    )
    return True


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

    def is_unavailable(self, history: Sequence[ToolEvent]) -> bool:
        return _proof_snapshot(history).unavailable

    def create_state(self) -> _ProofAnchorState:
        return _ProofAnchorState()

    def step(self, state: Any, event: ToolEvent) -> tuple[Any, AnchorSnapshot]:
        assert isinstance(state, _ProofAnchorState)
        return state.advance(event)
