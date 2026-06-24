"""Arbitration: combine detector signals into a graded action.

Default policy thresholds (see cursor.md):
    max_same_error: 4
    max_tool_retries: 3
    max_cost_usd: 5
    stop_threshold: 0.9

Sprint 1 ships a deliberately minimal arbitration (ARB-2): a single threshold
mapping the strongest detector confidence to ``STOP``. With one detector,
"fusion" is just taking that signal, so weighted fusion (ARB-1) is unnecessary
yet. The full 5-tier ladder (warn -> summarize -> fallback -> escalate -> stop)
and hysteresis are deferred to Sprint 3. The input stays continuous: detectors
report confidence in [0, 1], the policy maps it to an action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pathlayer.detectors.base import Signal
from pathlayer.policies.actions import Action

# Signals delivered in "observe" mode: computed and logged, but excluded from
# the STOP decision. NOP-2 (no_progress) ships here because, on its own, it
# cannot tell "task done" from "stuck" (a plateau after success looks like a
# stall). It earns STOP authority once NOP-3 (activity vs progress) and
# hysteresis (ARB-3) are in place. Minimal precursor to dry-run mode (OBS-3).
DEFAULT_ADVISORY_SIGNALS: frozenset[str] = frozenset({"no_progress"})


@dataclass(slots=True)
class PolicyConfig:
    max_same_error: int = 4
    max_tool_retries: int = 3
    max_cost_usd: float = 5.0
    # Strongest detector confidence at/above which the run is stopped.
    stop_threshold: float = 0.9
    # Detector names whose signals are observe-only: logged but never STOP.
    advisory_signals: frozenset[str] = DEFAULT_ADVISORY_SIGNALS

    def __post_init__(self) -> None:
        # stop_threshold is a confidence in [0, 1]; reject misconfigurations
        # early so they don't silently turn into always-stop / never-stop.
        if not 0.0 <= self.stop_threshold <= 1.0:
            raise ValueError("stop_threshold must be in [0, 1]")


@dataclass(slots=True)
class Policy:
    config: PolicyConfig = field(default_factory=PolicyConfig)

    def decide(self, signals: Sequence[Signal]) -> Action:
        """Map the strongest *actionable* signal confidence to a graded action.

        Minimal single-threshold ladder (ARB-2): if any non-advisory detector's
        confidence reaches ``stop_threshold``, return ``STOP``; otherwise
        ``NONE``. Advisory signals (see ``advisory_signals``) are still computed
        and logged upstream, they just cannot STOP on their own.
        """
        actionable = [s for s in signals if s.name not in self.config.advisory_signals]
        if not actionable:
            return Action.NONE
        peak = max(signal.confidence for signal in actionable)
        if peak >= self.config.stop_threshold:
            return Action.STOP
        return Action.NONE
