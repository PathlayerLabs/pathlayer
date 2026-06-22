"""ProgressAnchor interface and the built-in coding anchors (NOP-1).

A :class:`ProgressAnchor` turns the observed event history into a *progress
level*: a scalar that rises when the agent makes real, verifiable headway and
stays flat when it is busy but stuck. ``NoProgressDetector`` aggregates one or
more anchors and never references a concrete one, so the same detector serves
coding, email, scraping, ... agents -- only the anchors change.

Auto-gating: ``measure`` returns ``None`` when the anchor sees none of its
evidence in the history (e.g. the coding anchors on an email agent). ``None``
means "no opinion", never "no progress" -- this is what prevents a false
positive on an agent the anchor was never built for.

The coding anchors are grounded on deterministic ground truth the events
already carry -- test ``passed`` / ``failed`` counts and process ``exit_code``
-- not text similarity. They are split so each anchor stays on a *single* scale:
folding a cumulative success count and a ``passed - failed`` level into one
anchor would compare an early exit-code prefix against a later test level (a
spurious "drop" -> false positive). ``ExitCodeProgressAnchor`` also abstains
whenever any test evidence is present, so tests take precedence and a
never-decreasing success count can never mask a real test stall under the
detector's optimistic (most-progress-wins) aggregation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pathlayer.events import ToolEvent

_PASSED = "passed"
_FAILED = "failed"
_EXIT_CODE = "exit_code"
_TEST_KEYS = (_PASSED, _FAILED)


class ProgressAnchor(ABC):
    """Maps an event history to a continuous progress level, or ``None``."""

    name: str = "anchor"

    @abstractmethod
    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        """Return the current progress level, or ``None`` if not applicable.

        Pure and deterministic: the same history must yield the same value, with
        no clock, randomness, network or LLM. The value is a *level*, so a
        detector may call ``measure`` on prefixes of the history to recover the
        progress series over time.
        """
        raise NotImplementedError


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _evidence(event: ToolEvent, keys: tuple[str, ...]) -> Mapping[str, Any] | None:
    """Return the structured source (event result or metadata) carrying any of
    ``keys``, or ``None`` if the event carries none."""
    result = event.result
    if isinstance(result, Mapping) and any(key in result for key in keys):
        return result
    if any(key in event.metadata for key in keys):
        return event.metadata
    return None


class TestProgressAnchor(ProgressAnchor):
    """Ground-truth coding progress from test results.

    The level is ``passed - failed`` of the most recent test event. Returns
    ``None`` when the history carries no test evidence, so it abstains on a
    test-less (e.g. non-coding) agent.
    """

    name = "coding_tests"

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        level: float | None = None
        for event in history:
            source = _evidence(event, _TEST_KEYS)
            if source is None:
                continue
            passed = _as_int(source.get(_PASSED, 0), default=0)
            failed = _as_int(source.get(_FAILED, 0), default=0)
            level = float(passed - failed)
        return level


class ExitCodeProgressAnchor(ProgressAnchor):
    """Fallback coding progress from process exit codes.

    The level is the running count of successful (``exit_code == 0``) commands.
    It abstains (``None``) when the history carries no exit-code evidence *or*
    when it carries any test evidence: tests are the stronger ground truth and
    take precedence, which also keeps this never-decreasing count from masking a
    real test stall under the detector's optimistic aggregation.
    """

    name = "coding_exit_code"

    def measure(self, history: Sequence[ToolEvent]) -> float | None:
        successes = 0
        saw_exit_code = False
        for event in history:
            if _evidence(event, _TEST_KEYS) is not None:
                # Tests present anywhere -> defer to TestProgressAnchor.
                return None
            source = _evidence(event, (_EXIT_CODE,))
            if source is None:
                continue
            saw_exit_code = True
            if _as_int(source.get(_EXIT_CODE), default=-1) == 0:
                successes += 1
        return float(successes) if saw_exit_code else None


def coding_anchors() -> list[ProgressAnchor]:
    """The built-in coding anchors, ready to wire into ``NoProgressDetector``.

    Tests first (the stronger signal), exit codes as the fallback for agents
    that report process status but no test results.
    """
    return [TestProgressAnchor(), ExitCodeProgressAnchor()]
