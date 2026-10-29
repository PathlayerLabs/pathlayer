"""Minimal ``StateEvidence`` contract.

COMP-1 asked whether state-guided parameterized subtraces reuse better
than full replay without raising false positives on healthy runs. Answering
that requires *some* way to observe, before and after a tool boundary,
whether the state a contract's pre/postconditions talk about actually
changed -- without first designing Pathlayer's general collection
architecture (storage v2, a causal graph, a generic collector registry, ...).
The ARCH-1 spike scoped that down to the smallest thing COMP-1 needed; this
module is that smallest thing: a single, versioned, hand-validated record
type, ``StateEvidence``, plus a validator and a canonical hash. It defines
nothing about *how* a collector captures a resource -- that is ARCH-1B's job
(filesystem existence/hash, test pass/fail sets) -- only the shape every
collector's output must fit.

The sealed COMP-1A benchmark matrix
fixes exactly what vocabulary this shape needs to carry: every
precondition/postcondition across its three contracts is phrased either as a
filesystem canonical-hash comparison ("differs from its initial hash" /
"unchanged") or a named test-id-set pass/fail transition ("selected failure
set becomes empty", "baseline passing tests unchanged") -- never a bare
boolean. ``raw_fingerprint``/``canonical_fingerprint``/``delta`` are
therefore generic, collector-defined JSON here, not fixed to either shape:
closing them to ``{"filesystem", "tests"}``-specific structure would mean
this module hardcoding ARCH-1B's collector set, which ARCH-1A explicitly
reserves for ARCH-1B. Fixtures under ``tests/fixtures/state_evidence/`` demonstrate both
shapes without the schema enforcing either.

Field semantics:

- ``tool_event_ref``: an opaque, caller-minted correlation id for "the
  concerned ``ToolEvent``". ``pathlayer.events.ToolEvent`` carries no id
  field, and OpenClaw's adapter wires only ``after_tool_call`` (no live
  "before" hook exists to correlate against today) -- adding either is an
  event-schema/adapter change the ARCH-1 go/no-go explicitly forbids unless the
  benchmark needs it. It doesn't: whatever brackets a tool call with
  ``capture(phase="before")``/``capture(phase="after")`` (almost certainly
  ARCH-1B's benchmark-harness script) mints this id itself and owns the
  correlation table out of band. This module makes no assumption about its
  format beyond "non-empty bounded string".
- ``phase``: ``"before"`` or ``"after"`` -- which side of the tool boundary
  this observation was taken on.
- ``resource_type`` / ``resource_id``: what was observed (e.g.
  ``"filesystem"`` / a path, or ``"tests"`` / a named test selection).
  ``resource_type`` is a bounded non-empty string, not a closed enum -- see
  above.
- ``status``: ``"observed"`` (a fingerprint was successfully captured),
  ``"unavailable"`` (collection was never attempted or possible at this
  phase -- e.g. no live before-hook), or ``"collection_failed"`` (collection
  was attempted and errored). This tri-state is the *entire* mechanism by
  which "no evidence" stays distinguishable from "no change": a consumer
  must never infer "unchanged" from a missing fingerprint, only from an
  ``observed`` fingerprint that compares equal. COMP-1B's abstention logic reads
  this field directly, so the discriminator is ``status`` alone --
  :func:`validate_state_evidence` enforces that a non-``observed`` record
  can never carry a fingerprint or delta, so this invariant cannot be
  violated by construction once a record has validated.
- ``raw_fingerprint`` / ``canonical_fingerprint``: the collector's captured
  and canonicalized observation, generic JSON, present only when
  ``status == "observed"``.
- ``delta``: the observed change, generic JSON, present only on an
  ``"after"``, ``"observed"`` record (there is nothing to diff against on a
  first, ``"before"`` observation).
- ``collector_version``: identifies which collector/canonicalizer produced
  this record, so a later consumer can check a record was produced by the
  version a contract expects.

Go/no-go (per ARCH-1): **Go to ARCH-1B.** This schema unambiguously represents a
before/after observation and its absences via ``status`` alone, and required
no storage migration, no ``events.py`` change, and no adapter change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from pathlayer.store import _canonical_sha256

_SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
_PHASES = frozenset({"before", "after"})
_STATUSES = frozenset({"observed", "unavailable", "collection_failed"})

_EVIDENCE_HASH_ALGORITHM_PREFIX = "sha256:"

# Untrusted, producer-controlled wire payload -- bounding every axis prevents an
# adversarial or malformed stream from turning a captured record into unbounded
# memory growth, mirroring proof.py's own _MAX_* constants for the ANC-1
# proof_spec/progress_proof contract.
_MAX_TOOL_EVENT_REF_LENGTH = 512
_MAX_RESOURCE_TYPE_LENGTH = 128
_MAX_RESOURCE_ID_LENGTH = 1024
_MAX_COLLECTOR_VERSION_LENGTH = 128

# raw_fingerprint/canonical_fingerprint/delta are recursive JSON trees, not
# proof.py's flat string/bool fields -- bounding only per-container size would
# not stop a combinatorial blow-up from many nested bounded containers, so
# _MAX_JSON_TOTAL_NODES threads one running counter through the whole walk.
# _MAX_JSON_INT_BITS bounds the one JSON leaf type that has no other size axis:
# an unbounded integer costs _is_bounded_json_value only one node to accept, but
# json.dumps must later stringify it to hash it (_canonical_sha256), and CPython
# both caps (ValueError past ~4300 decimal digits) and slows quadratically well
# before that -- so magnitude needs its own bound, checked via the cheap
# int.bit_length() rather than by stringifying the value just to measure it.
_MAX_JSON_DEPTH = 6
_MAX_JSON_STRING_LENGTH = 4096
_MAX_JSON_TOTAL_NODES = 5000
_MAX_JSON_INT_BITS = 256


@dataclass(slots=True, frozen=True)
class StateEvidence:
    """Validated V1 ``StateEvidence`` record. See module docstring.

    Not guaranteed hashable, unlike ANC-1's ``ProofSpec``/``ProgressProof``:
    ``raw_fingerprint``/``canonical_fingerprint``/``delta`` stay generic JSON
    (dicts/lists are unhashable) rather than being frozen into tuples, since
    :func:`compute_evidence_hash` already gets order-independence for free
    from ``_canonical_sha256``'s recursive ``sort_keys`` -- no bespoke frozen
    representation is needed just to make hashing deterministic. Nothing in
    ARCH-1A's scope needs a ``StateEvidence`` as a dict key or set member.
    """

    schema_version: int
    tool_event_ref: str
    phase: str
    resource_type: str
    resource_id: str
    status: str
    raw_fingerprint: Any
    canonical_fingerprint: Any
    delta: Any
    collector_version: str


# Derived from the dataclass itself (not hand-duplicated) so a field added to
# StateEvidence can never silently drift out of sync with the closed-schema
# check below.
_ALLOWED_FIELDS = frozenset(f.name for f in fields(StateEvidence))


def _is_bounded_json_value(value: Any) -> bool:
    """Whether ``value`` is JSON-shaped and within the module's size/depth caps.

    A single running node counter (threaded through the recursive walk via a
    closure, not exposed as a parameter, and never reset per level) closes the
    combinatorial gap a per-container-only bound would leave open: many small,
    individually-bounded nested containers could otherwise still add up to
    unbounded total memory.
    """
    node_count = 0

    def _walk(node: Any, depth: int) -> bool:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_JSON_TOTAL_NODES or depth > _MAX_JSON_DEPTH:
            return False
        if node is None or isinstance(node, bool):
            return True
        if isinstance(node, int):
            # Checked via bit length, not by stringifying -- see _MAX_JSON_INT_BITS.
            return node.bit_length() <= _MAX_JSON_INT_BITS
        if isinstance(node, float):
            return math.isfinite(node)
        if isinstance(node, str):
            return len(node) <= _MAX_JSON_STRING_LENGTH
        if isinstance(node, list):
            return all(_walk(item, depth + 1) for item in node)
        if isinstance(node, dict):
            for key, item in node.items():
                if not isinstance(key, str) or len(key) > _MAX_JSON_STRING_LENGTH:
                    return False
                if not _walk(item, depth + 1):
                    return False
            return True
        return False

    return _walk(value, 0)


def validate_state_evidence(raw: Any) -> StateEvidence | None:
    """Structurally validate a raw ``StateEvidence`` mapping; ``None`` on any violation.

    Closed schema: exactly ``{schema_version, tool_event_ref, phase,
    resource_type, resource_id, status, raw_fingerprint,
    canonical_fingerprint, delta, collector_version}`` -- V1 has no optional
    field. Never raises: malformed, producer-controlled input is rejected,
    not an error.

    The tri-state rule below is the load-bearing check: it is what makes
    ``status`` the *only* discriminator a consumer needs, so "absence of
    proof" can never be smuggled in as, or confused with, "no change".

    - ``status == "observed"``: ``raw_fingerprint`` and
      ``canonical_fingerprint`` must each be present, non-``None``, and
      bounded JSON. ``delta`` must be ``None`` on ``phase == "before"``
      (nothing to diff against yet); on ``phase == "after"`` it may be
      ``None`` or bounded JSON.
    - ``status in {"unavailable", "collection_failed"}``: ``raw_fingerprint``,
      ``canonical_fingerprint``, and ``delta`` must all be exactly ``None`` --
      a producer cannot smuggle a partial fingerprint under a non-``observed``
      status.
    """
    if not isinstance(raw, Mapping):
        return None
    if raw.keys() != _ALLOWED_FIELDS:
        return None
    schema_version = raw["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)  # bool is an int subclass
        or schema_version not in _SUPPORTED_SCHEMA_VERSIONS
    ):
        return None
    tool_event_ref = raw["tool_event_ref"]
    resource_type = raw["resource_type"]
    resource_id = raw["resource_id"]
    collector_version = raw["collector_version"]
    for value, cap in (
        (tool_event_ref, _MAX_TOOL_EVENT_REF_LENGTH),
        (resource_type, _MAX_RESOURCE_TYPE_LENGTH),
        (resource_id, _MAX_RESOURCE_ID_LENGTH),
        (collector_version, _MAX_COLLECTOR_VERSION_LENGTH),
    ):
        if not isinstance(value, str) or not value or len(value) > cap:
            return None
    phase = raw["phase"]
    if phase not in _PHASES:
        return None
    status = raw["status"]
    if status not in _STATUSES:
        return None
    raw_fingerprint = raw["raw_fingerprint"]
    canonical_fingerprint = raw["canonical_fingerprint"]
    delta = raw["delta"]
    if status == "observed":
        if raw_fingerprint is None or not _is_bounded_json_value(raw_fingerprint):
            return None
        if canonical_fingerprint is None or not _is_bounded_json_value(canonical_fingerprint):
            return None
        if phase == "before":
            if delta is not None:
                return None
        elif delta is not None and not _is_bounded_json_value(delta):
            return None
    elif raw_fingerprint is not None or canonical_fingerprint is not None or delta is not None:
        return None
    return StateEvidence(
        schema_version=schema_version,
        tool_event_ref=tool_event_ref,
        phase=phase,
        resource_type=resource_type,
        resource_id=resource_id,
        status=status,
        raw_fingerprint=raw_fingerprint,
        canonical_fingerprint=canonical_fingerprint,
        delta=delta,
        collector_version=collector_version,
    )


def compute_evidence_hash(evidence: StateEvidence) -> str:
    """Canonical content hash of a validated :class:`StateEvidence`.

    Mirrors ``pathlayer.detectors.proof.compute_spec_hash``'s canonical-JSON
    approach (sorted keys, compact separators, no ``NaN``), reusing the same
    shared :func:`pathlayer.store._canonical_sha256`. Because ``sort_keys``
    sorts recursively at every nesting depth, this hash is already
    independent of how a producer ordered keys inside a nested
    ``raw_fingerprint``/``canonical_fingerprint``/``delta`` -- no bespoke
    frozen representation is needed to get that property. Uses
    :func:`dataclasses.asdict` rather than a hand-listed field dict, so a
    field added to :class:`StateEvidence` is hashed automatically instead of
    silently missing from the digest.
    """
    digest = _canonical_sha256(asdict(evidence))
    return _EVIDENCE_HASH_ALGORITHM_PREFIX + digest
