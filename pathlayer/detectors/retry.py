"""RetryDetector (RET-1 + RET-2 + RET-3).

Catches two flavours of the "keep hammering, expect a different result" failure,
folded into a single ``retry`` signal (``confidence = max`` of the two):

- **RET-1 -- identical failing retry.** The agent hits an error and replays the
  *identical* call. We look at the tail of the stream and count how many
  consecutive events share the same canonical action token (see
  :func:`pathlayer.detectors.canonical.canonicalize`) *and* the same error.
  Using the canonical token means volatile noise (request ids, timestamps) does
  not hide a retry, and a genuine change of arguments breaks the streak.
  Confidence ramps from the 2nd identical failure (the first actual retry) up to
  1.0 at ``threshold`` consecutive failures.

- **RET-2 -- same tool regardless of error.** The agent pounds one tool over and
  over with genuinely different arguments and/or mixed errors, so RET-1's
  exact-match streak stays silent yet the run is clearly stuck on that tool. We
  measure the *trailing uninterrupted run* of one tool over a sliding ``window``
  and fire when it both dominates the window (length and ratio thresholds) *and*
  fails at least ``min_failures`` times inside the run -- a pure all-success
  burst (legitimate exploration / polling) stays silent. Crucially this
  component is **capped** at ``dominance_max_confidence`` (default 0.85, below
  the default ``stop_threshold`` of 0.9): tool dominance is weak,
  framework-agnostic evidence, so with the default policy it reinforces a stop
  via fusion with a real no-progress signal but cannot stop a run on its own.
  The exact-match RET-1 component keeps its full authority up to 1.0.

- **RET-3 -- reset on a genuinely different action.** The retry counters reset
  when the agent genuinely moves on, at the granularity each component reasons
  about:

    * RET-1 already resets on a different *canonical token*: its streak breaks
      as soon as ``canonicalize(event)`` changes (a different call, different
      args, or a different error). This is the canonical-token reset called for
      in the issue.
    * RET-2 resets on a different *tool*: because the dominance component
      deliberately ignores args and errors (its whole niche is one tool hammered
      with genuinely different args, where every canonical token differs),
      "genuinely different" here means a different tool. Only the trailing run
      of the same tool counts, so a different tool anywhere in the window breaks
      the run. Resetting RET-2 on a canonical-token change instead would nullify
      it -- its storms carry a distinct token on every call.

  Either way, an old storm the agent has since moved past no longer contributes:
  two separate retry bursts split by real work are not merged into a false
  alarm.

Both components are pure functions of the observed history: same events in, same
signal out. No clock, no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.canonical import ToolMasks, canonicalize
from pathlayer.events import EventKind, ToolEvent


def _error_of(event: ToolEvent) -> str | None:
    """Extract a comparable error marker, or None if the call did not fail.

    A falsy marker (missing, ``None`` or empty string) means *no error*, so an
    explicit ``{"error": null}`` / ``{"error": ""}`` success is not mistaken for
    a failed retry.
    """
    result = event.result
    if isinstance(result, dict) and result.get("error"):
        return str(result["error"])
    meta_error = event.metadata.get("error")
    if meta_error:
        return str(meta_error)
    return None


class RetryDetector(Detector):
    name = "retry"
    # A stuck retry is a repeated *tool* call; LLM-cost events between attempts
    # must not break the streak, so reason over tool events only.
    event_kinds = frozenset({EventKind.TOOL_CALL})

    def __init__(
        self,
        threshold: int = 3,
        tool_masks: ToolMasks | None = None,
        *,
        window: int = 16,
        dominance_min_calls: int = 5,
        dominance_ratio: float = 0.75,
        dominance_max_confidence: float = 0.85,
        min_failures: int = 2,
    ) -> None:
        if threshold < 2:
            raise ValueError("threshold must be >= 2: a retry needs at least two attempts")
        if dominance_min_calls < 2:
            raise ValueError("dominance_min_calls must be >= 2")
        if window < dominance_min_calls:
            raise ValueError("window must be >= dominance_min_calls")
        if not 0.0 < dominance_ratio <= 1.0:
            raise ValueError("dominance_ratio must be in (0, 1]")
        if not 0.0 <= dominance_max_confidence <= 1.0:
            raise ValueError("dominance_max_confidence must be in [0, 1]")
        if min_failures < 1:
            raise ValueError("min_failures must be >= 1")
        self._threshold = threshold
        self._tool_masks = tool_masks
        self._window = window
        self._dominance_min_calls = dominance_min_calls
        self._dominance_ratio = dominance_ratio
        self._dominance_max_confidence = dominance_max_confidence
        self._min_failures = min_failures

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        history = self._relevant(history)
        if not history:
            return Signal(self.name, confidence=0.0)

        streak_conf, streak_reason = self._consecutive_failure_confidence(history)
        dom_conf, dom_reason = self._tool_dominance_confidence(history)

        # Fold the two components into one signal: report the stronger one, so
        # RET-1's full-authority streak wins over the capped dominance whenever
        # both fire, and the reason names the component that actually drove it.
        if streak_conf >= dom_conf:
            confidence, reason = streak_conf, streak_reason
        else:
            confidence, reason = dom_conf, dom_reason
        return Signal(self.name, confidence=confidence, reason=reason)

    def _consecutive_failure_confidence(self, history: Sequence[ToolEvent]) -> tuple[float, str]:
        """RET-1: consecutive identical failing calls at the tail of the stream."""
        last = history[-1]
        error = _error_of(last)
        if error is None:
            # The most recent attempt did not fail -> not stuck retrying.
            return 0.0, ""

        token = canonicalize(last, self._tool_masks)
        reps = 1
        for event in reversed(history[:-1]):
            if canonicalize(event, self._tool_masks) == token and _error_of(event) == error:
                reps += 1
            else:
                break

        if reps < 2:
            return 0.0, ""

        confidence = min(1.0, (reps - 1) / (self._threshold - 1))
        reason = f"{reps}x identical failure of {last.tool!r} (error {error!r})"
        return confidence, reason

    def _tool_dominance_confidence(self, history: Sequence[ToolEvent]) -> tuple[float, str]:
        """RET-2 + RET-3: one tool dominates the *most recent uninterrupted* run.

        Counts raw tool names (not the canonical token) so genuinely different
        arguments and mixed results still count as the same tool. Gated three
        ways -- length of the trailing run, its share of the window, and a
        minimum number of failures inside that run -- so a legitimate all-success
        burst of one tool (exploration, polling) stays silent even if unrelated
        calls fail.

        RET-3: only the trailing run of the *same* tool counts. A genuinely
        different tool anywhere in the window breaks the run, so an old storm
        that the agent has since moved past no longer counts -- the counter
        resets on a genuinely different action instead of accumulating a bag of
        occurrences across unrelated phases. Reset is tool-level here on purpose:
        RET-2 ignores args/errors (that is its niche), so a canonical-token reset
        would fire on every call and nullify it. The canonical-token reset the
        issue asks for is provided by the RET-1 component above.

        The confidence is a fixed value capped at ``dominance_max_confidence``,
        which defaults below ``stop_threshold``: with the default policy tool
        dominance reinforces a stop via fusion but cannot stop a run on its own.
        """
        window = history[-self._window :]
        if len(window) < self._dominance_min_calls:
            return 0.0, ""

        tool = window[-1].tool
        count = 0
        for event in reversed(window):
            if event.tool != tool:
                break
            count += 1

        if count < self._dominance_min_calls or count / len(window) < self._dominance_ratio:
            return 0.0, ""

        # Gate on failures inside the trailing run itself: a tool that keeps
        # succeeding is not stuck, even if other calls around it fail.
        run = window[-count:]
        failures = sum(1 for event in run if _error_of(event) is not None)
        if failures < self._min_failures:
            return 0.0, ""

        # The run is at least ``dominance_min_calls`` long, so dominance is
        # always a fixed, capped nudge by design -- weak, framework-agnostic
        # evidence, never a STOP-strength signal on its own.
        confidence = self._dominance_max_confidence
        reason = (
            f"{tool!r} dominates the last {count} calls regardless of result ({failures} failed)"
        )
        return confidence, reason
