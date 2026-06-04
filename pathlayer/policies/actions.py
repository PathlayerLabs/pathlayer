"""Graded intervention actions, from least to most disruptive."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    NONE = 0
    WARN = 1
    SUMMARIZE = 2
    FALLBACK = 3
    ESCALATE = 4
    STOP = 5
