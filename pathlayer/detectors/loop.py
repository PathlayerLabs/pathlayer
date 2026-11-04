"""LoopDetector (class i: repetition, LOOP-1 + LOOP-2 + LOOP-3).

Table stakes. Canonicalizes each event into an action token (see
:func:`pathlayer.detectors.canonical.canonicalize`) and combines two
components into a single continuous ``loop`` signal:

- **LOOP-1 -- exact trailing repetition.** Looks for an exact repeated block at
  the *tail* of the stream: a looping agent keeps replaying the same action(s).
  The repeating unit can be a single action (``read, read, read``) or a
  multi-action cycle (``a, b, a, b``). ``k`` is the minimum number of
  consecutive repetitions before any confidence is reported; confidence then
  rises toward 1 as the loop persists. This component has full authority up to
  1.0 and drives ``STOP`` on its own, as today.

- **LOOP-2 -- n-gram coverage (recall).** Catches a cycle that dominates the
  window but never accumulates ``k`` *consecutive* exact repeats because it is
  regularly interrupted by a distinct one-off action (e.g. ``a, b, a, b,
  probe_1, a, b, a, b, probe_2, ...``) -- LOOP-1's strict trailing-block match
  resets on every interruption and stays silent forever on this shape, even
  though the ``a, b`` cycle covers most of the window (proven empirically: see
  the LOOP-2 go/no-go spike). This component only ever *opens a new
  detection path*; it never boosts a loop LOOP-1 already sees, by design (see
  ``evaluate``): it is capped at ``ngram_max_confidence`` (default ``0.85``,
  below the default ``stop_threshold`` of ``0.9``, mirroring ``RET-2``'s
  ``dominance_max_confidence``) since it is a new, uncalibrated heuristic --
  precision over recall until validated on the golden set (see the LOOP-2
  discussion: an additive/OR design here would only ever *increase* false
  positives on the component already flagged as the FP hotspot, which is the
  opposite of what's wanted; see also the ``loop <- progress`` suppression in
  ``pathlayer.policies.base.SuppressionRule``, which this component inherits
  for free by staying under the same ``"loop"`` signal name).

- **LOOP-3 -- windowed (continuous) scoring for LOOP-2.** Two go/no-go
  spikes (see the LOOP-3 discussion) ruled out the two most obvious designs:
  fusing LOOP-1 and LOOP-2 into one blended score (pointless -- on every known
  case where LOOP-2 fires, LOOP-1 independently reports exactly ``0.0``, so a
  fusion would only reopen the additive-FP risk already rejected for LOOP-2
  above) and simply lowering/removing the ``ngram_min_hits`` gate (unsafe --
  empirically regresses a purpose-built false-positive-boundary fixture,
  ``traces/healthy/retry_then_succeed.json``, from ``0.0`` straight to
  ``0.333``). What *was* validated: the ``ngram_min_hits`` gate in
  ``_ngram_coverage_confidence`` was a hard step (exactly ``0.0`` below the
  threshold, a discontinuous jump to the raw coverage value at/above it) --
  replaced here with a continuous sigmoid ramp centered on ``ngram_min_hits``,
  so confidence rises smoothly with repetition count instead of jumping. The
  hard floor of 2 occurrences (``_NGRAM_HITS_FLOOR``, not configurable) is kept
  as-is -- it is what protects ``retry_then_succeed.json``, and the spike
  showed removing it (not just softening the threshold above it) is what
  caused the regression.

``ngram_max_confidence`` itself is intentionally left untouched by LOOP-3:
across the entire traces corpus, only two fixtures ever reach the LOOP-2
fallback path at all, both around ``0.34``-``0.35`` -- nowhere near the
``0.85`` cap, which has therefore never been the binding constraint on any
known trace. Moving it would be an uncalibrated guess, exactly what the
`repository rules <https://github.com/PathlayerLabs/pathlayer/blob/main/CONTRIBUTING.md>`_
rule out; raising it to grant LOOP-2 solo-stop
authority is left as an explicitly open question for a richer, non-synthetic
corpus (``VAL-5``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.canonical import ToolMasks, canonicalize
from pathlayer.events import EventKind, ToolEvent

# Default n-gram sizes scanned by the LOOP-2 coverage component. Size 1 is
# deliberately excluded: a single fixed, argument-less checkpoint call (e.g.
# ``run_tests()`` with no target) recurring between genuinely distinct actions
# is a normal coding pattern (verified against traces/healthy/
# multi_file_refactor.json: including size 1 turns that healthy fixture into a
# false positive at confidence ~0.33). Size >= 2 requires a *pair* of actions
# to repeat together, which a single recurring checkpoint call does not form.
# This is a secondary, weaker argument, but worth noting: size 1 would also
# partially overlap with RET-2's tool-dominance niche (RetryDetector), though
# that niche differs in scope (an *uninterrupted* trailing run gated on
# failures, not a scattered count over the whole window).
_DEFAULT_NGRAM_SIZES: tuple[int, ...] = (2, 3, 4, 5)

# Hard structural minimum for the LOOP-2 coverage component, deliberately not
# configurable (unlike ngram_min_hits, which is now the *center* of the LOOP-3
# ramp rather than a gate -- see the module docstring). A single occurrence is
# not a repeat at all and can trivially dominate a short window (a lone repeat
# in a 4-token window already "covers" 50%): verified in the LOOP-3 spike that
# lowering this below 2 regresses traces/healthy/retry_then_succeed.json.
_NGRAM_HITS_FLOOR: int = 2


def _sigmoid(x: float, steepness: float) -> float:
    """Logistic curve in (0, 1), steep enough to approximate a step but smooth.

    Used to replace a hard threshold gate with a continuous ramp centered at
    ``x == 0``. The *exponent* (``-steepness * x``), not ``x`` alone, is
    clamped: ``steepness`` has no configured upper bound (only ``> 0``), and
    ``ngram_min_hits`` has no upper bound either, so clamping ``x`` first does
    not prevent their product from overflowing ``math.exp`` (verified: e.g.
    ``ngram_min_hits=60, ngram_steepness=20.0`` raises ``OverflowError`` with
    an ``x``-only clamp). Clamping the exponent to +/-700 keeps
    ``math.exp`` within ``float64`` range regardless of ``steepness``, while
    saturating to exactly 0.0 or 1.0 well before that -- no observable effect
    on the ramp's shape.
    """
    exponent = max(-700.0, min(700.0, -steepness * x))
    return 1.0 / (1.0 + math.exp(exponent))


class LoopDetector(Detector):
    name = "loop"
    # Only agent actions can form a loop; ignore LLM-cost events so a run of
    # consecutive LLM calls is not mistaken for a repeated action.
    event_kinds = frozenset({EventKind.TOOL_CALL})

    def __init__(
        self,
        k: int = 2,
        window: int = 256,
        tool_masks: ToolMasks | None = None,
        *,
        ngram_sizes: Sequence[int] = _DEFAULT_NGRAM_SIZES,
        ngram_min_hits: int = 3,
        ngram_min_coverage: float = 0.3,
        ngram_max_confidence: float = 0.85,
        ngram_steepness: float = 3.0,
    ) -> None:
        if k < 2:
            raise ValueError("k must be >= 2: a loop needs at least two repetitions")
        if window < 2 * k:
            raise ValueError("window must be >= 2 * k to fit at least k repetitions")
        sizes = tuple(ngram_sizes)
        if not sizes or any(size < 2 for size in sizes):
            raise ValueError("ngram_sizes must be non-empty and each size must be >= 2")
        if ngram_min_hits < 2:
            raise ValueError("ngram_min_hits must be >= 2: coverage needs at least two occurrences")
        if not 0.0 < ngram_min_coverage <= 1.0:
            raise ValueError("ngram_min_coverage must be in (0, 1]")
        if not 0.0 <= ngram_max_confidence <= 1.0:
            raise ValueError("ngram_max_confidence must be in [0, 1]")
        if ngram_steepness <= 0.0:
            raise ValueError("ngram_steepness must be > 0: it controls a sigmoid growth rate")
        self._k = k
        self._window = window
        self._tool_masks = tool_masks
        self._ngram_sizes = sizes
        self._ngram_min_hits = ngram_min_hits
        self._ngram_min_coverage = ngram_min_coverage
        self._ngram_max_confidence = ngram_max_confidence
        self._ngram_steepness = ngram_steepness

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        history = self._relevant(history)
        # Bound per-step cost: only the recent tail window is canonicalized and
        # scanned. Pipeline.process() runs every detector on every event, so an
        # unbounded scan would be ~O(N^3) over a long run; the window keeps it
        # O(window^2) per step regardless of history length.
        tokens = [canonicalize(event, self._tool_masks) for event in history[-self._window :]]
        confidence, reason = self._exact_repeat_confidence(tokens)
        if confidence == 0.0:
            # LOOP-2 only ever fires as a *fallback* when LOOP-1 sees nothing at
            # all -- never as a boost on top of an existing LOOP-1 hit. This
            # makes non-regression on every exact-repeat trace trivial (same
            # code path, byte-for-byte identical output) and guarantees LOOP-2
            # can only ever *add* detections, never inflate an already-detected
            # one past what LOOP-1 alone would report.
            confidence, reason = self._ngram_coverage_confidence(tokens)
        return Signal(self.name, confidence=confidence, reason=reason)

    def _exact_repeat_confidence(self, tokens: list[str]) -> tuple[float, str]:
        """LOOP-1: exact repeated block at the tail of the stream."""
        reps, period = self._max_trailing_repetition(tokens)
        if reps < self._k:
            return 0.0, ""
        # Saturating map: reps == k -> 0.5, then halve the gap to 1 per extra rep.
        confidence = 1.0 - 0.5 ** (reps - self._k + 1)
        reason = f"{reps}x repeat of {period}-action block"
        return confidence, reason

    @staticmethod
    def _max_trailing_repetition(tokens: list[str]) -> tuple[int, int]:
        """Return (repetitions, period) of the strongest repeated tail block."""
        n = len(tokens)
        best_reps, best_period = 1, 0
        for period in range(1, n // 2 + 1):
            block = tokens[n - period :]
            reps = 1
            idx = n - period
            while idx - period >= 0 and tokens[idx - period : idx] == block:
                reps += 1
                idx -= period
            if reps > best_reps:
                best_reps, best_period = reps, period
        return best_reps, best_period

    def _ngram_coverage_confidence(self, tokens: list[str]) -> tuple[float, str]:
        """LOOP-2/LOOP-3: share of overlapping n-gram windows dominated by one
        pattern, damped by a continuous ramp on how many times it repeats.

        "Coverage" here is the fraction of *overlapping size-n windows* (start
        positions) that are the single most common n-gram, not a literal
        percentage of raw tokens in the window -- e.g. a fully periodic ``a, b,
        a, b`` run only counts the ``(a, b)`` starts, not the interleaved
        ``(b, a)`` ones, so it is a conservative lower bound on how much of the
        window the cycle actually occupies.

        Overlapping n-grams (not a non-overlapping tiling) on purpose: tiling
        from a fixed phase misses a cycle whose period does not evenly divide
        the window, or whose interrupting action falls *inside* the tiled
        period (verified empirically: a 2-of-3 stable/varying cycle scores near
        zero under tiling but ~1/3 coverage under the overlapping count, which
        is exactly the proven recall-gap shape for LOOP-2). Anti-double-counting
        (the ``A B A B A`` concern from the design discussion): only the
        *single best* n-gram size is reported, never a sum across sizes --
        several sizes describing the same underlying cycle must not inflate the
        score.

        Candidate sizes are ranked by raw coverage down to the
        hard structural floor (``_NGRAM_HITS_FLOOR``, 2 occurrences -- a single
        occurrence is not a repeat and can trivially dominate a short window).
        ``ngram_min_hits`` is no longer a hard gate on top of that floor: it is
        the *center* of a sigmoid that smoothly damps the winning candidate's
        confidence based on how many times it actually repeated, so confidence
        rises continuously with repetition instead of jumping from ``0.0`` to
        the raw coverage value the instant the old threshold was crossed.
        ``ngram_min_coverage`` remains a hard gate, unlike ``ngram_min_hits``:
        the LOOP-3 spike only found a discontinuity problem on the *hits* axis
        (which tracks repetition count, the DoD's "rises smoothly as repetition
        increases"), not on the coverage-share axis, which stays a semantic
        "does this pattern dominate the window at all" check.
        """
        n = len(tokens)
        best_coverage = 0.0
        best_hits = 0
        best_size = 0
        best_total = 0
        for size in self._ngram_sizes:
            if size > n:
                continue
            grams = [tuple(tokens[i : i + size]) for i in range(n - size + 1)]
            _, hits = Counter(grams).most_common(1)[0]
            if hits < _NGRAM_HITS_FLOOR:
                continue
            coverage = hits / len(grams)
            if coverage > best_coverage:
                best_coverage, best_hits, best_size, best_total = coverage, hits, size, len(grams)
        if best_size == 0 or best_coverage < self._ngram_min_coverage:
            return 0.0, ""
        hits_factor = _sigmoid(best_hits - self._ngram_min_hits, self._ngram_steepness)
        confidence = min(self._ngram_max_confidence, best_coverage * hits_factor)
        # `best_coverage` is the fraction of *overlapping best_size-action
        # windows* dominated by the single most common one, not a literal
        # percentage of raw tokens in the window -- spelled out as a ratio so
        # the reason string (surfaced in logs/postmortems) is not misread as
        # "X% of the window's tokens".
        reason = (
            f"{best_hits}/{best_total} overlapping {best_size}-action windows are the "
            f"same recurring pattern ({best_coverage:.0%})"
        )
        return confidence, reason
