"""RetryDetector (RET-1).

Catches the "retry the exact same failing call and expect a different result"
pattern: the agent hits an error and replays the identical call, burning budget
on something that will never succeed.

It looks at the tail of the stream and counts how many consecutive events share
the same canonical action token (see
:func:`pathlayer.detectors.canonical.canonicalize`) *and* the same error. Using
the canonical token means volatile noise (request ids, timestamps) does not hide
a retry, and a genuine change of arguments breaks the streak.

The signal is continuous: confidence ramps from the 2nd identical failure (the
first actual retry) up to 1.0 at ``threshold`` consecutive failures. A streak
that ends in success, a changed call, or a different error reports 0.

RET-2 (same tool regardless of error) and RET-3 (reset on a genuinely different
action) build on this.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.canonical import canonicalize
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

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 2:
            raise ValueError("threshold must be >= 2: a retry needs at least two attempts")
        self._threshold = threshold

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        history = self._relevant(history)
        if not history:
            return Signal(self.name, confidence=0.0)

        last = history[-1]
        error = _error_of(last)
        if error is None:
            # The most recent attempt did not fail -> not stuck retrying.
            return Signal(self.name, confidence=0.0)

        token = canonicalize(last)
        reps = 1
        for event in reversed(history[:-1]):
            if canonicalize(event) == token and _error_of(event) == error:
                reps += 1
            else:
                break

        if reps < 2:
            return Signal(self.name, confidence=0.0)

        confidence = min(1.0, (reps - 1) / (self._threshold - 1))
        reason = f"{reps}x identical failure of {last.tool!r} (error {error!r})"
        return Signal(self.name, confidence=confidence, reason=reason)
