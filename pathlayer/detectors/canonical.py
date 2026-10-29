"""Canonicalization: turn a tool call into a stable action token.

This is the extraction step that ~80% of the real engineering difficulty lives
in. Two genuinely identical calls must produce the *same* token even when they
carry volatile noise (timestamps, UUIDs, PIDs, pagination cursors); two
genuinely different calls must produce *different* tokens (no false merge).

``canonicalize`` is a pure function: same event in, same token out. It masks
volatile data two ways -- by key name (e.g. ``request_id``, ``created_at``,
``cursor``) and by value shape (UUID / ISO-8601 timestamp) -- then serializes
``tool`` + canonicalized ``args`` deterministically. The event's own ``ts`` /
``result`` / ``cost_usd`` are not part of the action identity and are ignored.

CAN-3 (no false merge): the ``_id`` suffix is intentionally *not* in the default
volatile suffixes, because ``file_id`` / ``job_id`` / ``order_id`` are usually
the field that distinguishes two genuinely different calls. Truly volatile ids
stay masked by their explicit names (``request_id``, ``trace_id`` ...). When a
tool carries an id that *is* volatile by value (e.g. a UUID-valued ``file_id``),
declare it via a per-tool :class:`ToolMask` ``keep`` to stop it from merging
distinct calls.

CAN-2 (configurable masks): the complementary ``mask`` direction lets a tool
force-mask a field the default heuristic would keep -- e.g. an incrementing
``attempt_id`` that is just noise for that tool. ``keep`` and ``mask`` are the
two halves of the same per-tool override and must not overlap.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pathlayer.events import ToolEvent

# Argument keys whose value is volatile regardless of content.
VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "ts",
        "timestamp",
        "time",
        "datetime",
        "created_at",
        "updated_at",
        "uuid",
        "id",
        "request_id",
        "trace_id",
        "span_id",
        "correlation_id",
        "session_id",
        "pid",
        "process_id",
        "cursor",
        "next_cursor",
        "prev_cursor",
        "page_token",
        "next_page_token",
        "continuation_token",
        "etag",
        "nonce",
    }
)

# Suffixes that mark a key as volatile (e.g. ``started_at``, ``page_cursor``).
# ``_id`` is deliberately excluded: ids like ``file_id`` usually distinguish two
# different calls, so masking them by suffix causes false merges. A
# genuinely volatile id is either listed by name above or declared per tool.
VOLATILE_KEY_SUFFIXES: tuple[str, ...] = ("_at", "_uuid", "_ts", "_cursor", "_token")

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$",
)

MASK = "<volatile>"


@dataclass(frozen=True, slots=True)
class ToolMask:
    """Per-tool canonicalization override.

    ``keep`` lists argument key names that must never be masked for this tool,
    even when the default heuristic would treat them as volatile (by name or by
    value shape). This preserves a genuinely distinguishing field -- e.g. a
    UUID-valued ``file_id`` -- so distinct calls do not falsely merge.

    ``mask`` lists argument key names that must always be masked for this tool,
    even when the default heuristic would keep them. This force-masks a field
    that is just noise for that tool -- e.g. an incrementing ``attempt_id`` --
    so two calls differing only by it collapse to the same token.

    Matching is case-insensitive. ``keep`` and ``mask`` are contradictory by
    construction, so a key declared in both raises ``ValueError``.
    """

    keep: frozenset[str] = frozenset()
    mask: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        overlap = {k.lower() for k in self.keep} & {m.lower() for m in self.mask}
        if overlap:
            raise ValueError(
                "ToolMask keys cannot be both kept and masked "
                f"(case-insensitive): {sorted(overlap)}"
            )


# Maps a tool name to its canonicalization override.
ToolMasks = Mapping[str, ToolMask]


def _is_volatile_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in VOLATILE_KEYS or lowered.endswith(VOLATILE_KEY_SUFFIXES)


def _is_volatile_value(value: Any) -> bool:
    return isinstance(value, str) and bool(_UUID_RE.match(value) or _ISO_DATETIME_RE.match(value))


def _canonicalize_entry(key: str, val: Any, kept: frozenset[str], masked: frozenset[str]) -> Any:
    # Precedence (highest first): ``keep`` > ``mask`` > default heuristics.
    # ``kept``/``masked`` are already lowercased in ``canonicalize`` and cannot
    # overlap (enforced by ``ToolMask``), so the order between them is moot, but
    # we check ``keep`` first to make the "keep wins" intent explicit.
    lowered = key.lower()
    # ``keep`` wins over every default: value emitted verbatim, bypassing both
    # name-based and value-shape masking (and recursion into it).
    if lowered in kept:
        return val
    # ``mask`` force-masks a field the default heuristic would otherwise keep.
    if lowered in masked:
        return MASK
    if _is_volatile_key(key):
        return MASK
    return _canonicalize_value(val, kept, masked)


def _canonicalize_value(value: Any, kept: frozenset[str], masked: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize_entry(key, val, kept, masked) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_canonicalize_value(item, kept, masked) for item in value]
    if _is_volatile_value(value):
        return MASK
    return value


def canonicalize(event: ToolEvent, tool_masks: ToolMasks | None = None) -> str:
    """Return a stable, deterministic action token for ``event``.

    ``tool_masks`` optionally overrides the default masking per tool (see
    :class:`ToolMask`). Passing ``None`` keeps the default behavior.
    """
    kept: frozenset[str] = frozenset()
    masked: frozenset[str] = frozenset()
    if tool_masks is not None and event.tool in tool_masks:
        # Normalize once so keep/mask matching is case-insensitive like key masking.
        tool_mask = tool_masks[event.tool]
        kept = frozenset(k.lower() for k in tool_mask.keep)
        masked = frozenset(m.lower() for m in tool_mask.mask)
    canonical_args = _canonicalize_value(dict(event.args), kept, masked)
    return json.dumps(
        {"tool": event.tool, "args": canonical_args},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
