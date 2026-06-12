"""Canonicalization (CAN-1): turn a tool call into a stable action token.

This is the extraction step that ~80% of the real engineering difficulty lives
in. Two genuinely identical calls must produce the *same* token even when they
carry volatile noise (timestamps, UUIDs, PIDs, pagination cursors); two
genuinely different calls must produce *different* tokens (no false merge).

``canonicalize`` is a pure function: same event in, same token out. It masks
volatile data two ways -- by key name (e.g. ``request_id``, ``created_at``,
``cursor``) and by value shape (UUID / ISO-8601 timestamp) -- then serializes
``tool`` + canonicalized ``args`` deterministically. The event's own ``ts`` /
``result`` / ``cost_usd`` are not part of the action identity and are ignored.

Per-tool configurable masks are CAN-2; this module ships sensible defaults.
"""

from __future__ import annotations

import json
import re
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

# Suffixes that mark a key as volatile (e.g. ``order_id``, ``started_at``).
VOLATILE_KEY_SUFFIXES: tuple[str, ...] = ("_id", "_at", "_uuid", "_ts", "_cursor", "_token")

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$",
)

MASK = "<volatile>"


def _is_volatile_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in VOLATILE_KEYS or lowered.endswith(VOLATILE_KEY_SUFFIXES)


def _is_volatile_value(value: Any) -> bool:
    return isinstance(value, str) and bool(_UUID_RE.match(value) or _ISO_DATETIME_RE.match(value))


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: MASK if _is_volatile_key(key) else _canonicalize_value(val)
            for key, val in value.items()
        }
    if isinstance(value, list | tuple):
        return [_canonicalize_value(item) for item in value]
    if _is_volatile_value(value):
        return MASK
    return value


def canonicalize(event: ToolEvent) -> str:
    """Return a stable, deterministic action token for ``event``."""
    canonical_args = _canonicalize_value(dict(event.args))
    return json.dumps(
        {"tool": event.tool, "args": canonical_args},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
