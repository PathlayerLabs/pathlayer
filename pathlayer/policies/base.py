"""Arbitration: combine detector signals into a graded action.

Default policy thresholds (see cursor.md):
    max_same_error: 4
    max_tool_retries: 3
    max_cost_usd: 5
    no_progress_steps: 5
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pathlayer.detectors.base import Signal
from pathlayer.policies.actions import Action


@dataclass(slots=True)
class PolicyConfig:
    max_same_error: int = 4
    max_tool_retries: int = 3
    max_cost_usd: float = 5.0
    no_progress_steps: int = 5


@dataclass(slots=True)
class Policy:
    config: PolicyConfig = field(default_factory=PolicyConfig)

    def decide(self, signals: Sequence[Signal]) -> Action:
        """Fuse detector signals into a single graded action.

        TODO: rules table combining channels -> confidence -> graded action.
        """
        return Action.NONE
