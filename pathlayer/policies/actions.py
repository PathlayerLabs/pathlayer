"""Graded intervention actions, from least to most disruptive."""

from __future__ import annotations

from enum import Enum
from functools import total_ordering
from typing import Any


@total_ordering
class Action(str, Enum):
    """Ordered intervention tiers with a canonical, JSON-safe wire value.

    Actions used to inherit from :class:`IntEnum`.  That made a generic
    ``dataclasses.asdict`` + ``json.dumps`` export silently turn ``NONE`` into
    ``0``.  A string enum keeps the public representation canonical even on
    that generic path; ``__lt__`` preserves the policy's severity ordering.
    """

    NONE = "NONE"
    WARN = "WARN"
    SUMMARIZE = "SUMMARIZE"
    FALLBACK = "FALLBACK"
    ESCALATE = "ESCALATE"
    STOP = "STOP"

    def _compare(self, other: object) -> int | None:
        if not isinstance(other, Action):
            return None
        self_tier = _ACTION_TIERS.get(self)
        other_tier = _ACTION_TIERS.get(other)
        assert self_tier is not None and other_tier is not None
        return self_tier - other_tier

    def __lt__(self, other: object) -> bool:
        comparison = self._compare(other)
        return NotImplemented if comparison is None else comparison < 0

    def __le__(self, other: object) -> bool:
        comparison = self._compare(other)
        return NotImplemented if comparison is None else comparison <= 0

    def __gt__(self, other: object) -> bool:
        comparison = self._compare(other)
        return NotImplemented if comparison is None else comparison > 0

    def __ge__(self, other: object) -> bool:
        comparison = self._compare(other)
        return NotImplemented if comparison is None else comparison >= 0


_ACTION_TIERS: dict[Action, int] = {action: index for index, action in enumerate(Action)}
_ACTIONS_BY_LEGACY_TIER: tuple[Action, ...] = tuple(Action)


def action_from_export(value: Any, *, field: str = "action") -> Action:
    """Read an action exported as its canonical name or a legacy numeric tier.

    Public exports use the enum's name.  This reader keeps old numeric journals
    replayable while returning the same ``Action`` object as a live decision.
    Booleans are rejected despite being ``int`` subclasses.
    """
    if isinstance(value, Action):
        return value
    if isinstance(value, bool):
        raise ValueError(f"invalid {field}: boolean is not an action")
    if isinstance(value, int):
        if 0 <= value < len(_ACTIONS_BY_LEGACY_TIER):
            return _ACTIONS_BY_LEGACY_TIER[value]
        raise ValueError(f"invalid {field}: {value!r}")
    if isinstance(value, str):
        if value in Action.__members__:
            return Action[value]
        if value.isascii() and value.isdecimal():
            return action_from_export(int(value), field=field)
    raise ValueError(f"invalid {field}: expected an action name or legacy numeric tier")


def action_name(value: Action | str | int, *, field: str = "action") -> str:
    """Return the sole public representation of an action: its enum name."""
    return action_from_export(value, field=field).name
