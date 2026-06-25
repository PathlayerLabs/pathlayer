"""Tests for canonicalization (CAN-1).

Two axes matter and pull against each other:
- positive: identical-up-to-noise calls collapse to the same token;
- negative: genuinely different calls never collapse (no false merge).
"""

from __future__ import annotations

from pathlayer.detectors.canonical import ToolMask, canonicalize
from pathlayer.events import ToolEvent


def test_identical_calls_share_token() -> None:
    a = ToolEvent(tool="read", args={"path": "a.py"})
    b = ToolEvent(tool="read", args={"path": "a.py"})
    assert canonicalize(a) == canonicalize(b)


def test_token_is_deterministic() -> None:
    event = ToolEvent(tool="read", args={"path": "a.py", "mode": "r"})
    assert canonicalize(event) == canonicalize(event)


def test_arg_key_order_does_not_matter() -> None:
    a = ToolEvent(tool="call", args={"x": 1, "y": 2})
    b = ToolEvent(tool="call", args={"y": 2, "x": 1})
    assert canonicalize(a) == canonicalize(b)


def test_top_level_ts_is_ignored() -> None:
    a = ToolEvent(tool="read", args={"path": "a.py"}, ts=1.0)
    b = ToolEvent(tool="read", args={"path": "a.py"}, ts=999.0)
    assert canonicalize(a) == canonicalize(b)


def test_result_and_cost_are_ignored() -> None:
    a = ToolEvent(tool="read", args={"path": "a.py"}, result="ok", cost_usd=0.1)
    b = ToolEvent(tool="read", args={"path": "a.py"}, result="boom", cost_usd=5.0)
    assert canonicalize(a) == canonicalize(b)


def test_volatile_keys_are_masked() -> None:
    a = ToolEvent(tool="fetch", args={"url": "/x", "request_id": "r-1", "cursor": "c1"})
    b = ToolEvent(tool="fetch", args={"url": "/x", "request_id": "r-2", "cursor": "c2"})
    assert canonicalize(a) == canonicalize(b)


def test_volatile_key_suffixes_are_masked() -> None:
    # ``_at`` (and the other suffixes) still mark a key volatile by suffix.
    a = ToolEvent(tool="db", args={"table": "users", "created_at": "t1"})
    b = ToolEvent(tool="db", args={"table": "users", "created_at": "t2"})
    assert canonicalize(a) == canonicalize(b)


def test_id_suffix_is_not_masked_so_distinct_ids_stay_distinct() -> None:
    # CAN-3: ``_id`` is no longer volatile by suffix. ``file_id`` distinguishes
    # two genuinely different reads, so they must NOT collapse to one token.
    a = ToolEvent(tool="read_file", args={"file_id": 1})
    b = ToolEvent(tool="read_file", args={"file_id": 2})
    assert canonicalize(a) != canonicalize(b)


def test_named_volatile_ids_are_still_masked() -> None:
    # Truly volatile ids stay masked by their explicit name (not by suffix).
    a = ToolEvent(tool="fetch", args={"url": "/x", "request_id": "r-1", "session_id": "s-1"})
    b = ToolEvent(tool="fetch", args={"url": "/x", "request_id": "r-2", "session_id": "s-2"})
    assert canonicalize(a) == canonicalize(b)


def test_unnamed_volatile_id_suffix_is_now_visible_tradeoff() -> None:
    # CAN-3 trade-off (the other direction): dropping the `_id` suffix means a
    # volatile-but-unnamed id (e.g. an incrementing attempt_id) is no longer
    # masked. Two calls that differ only by it no longer share a token by
    # default; re-masking such a field per tool is the `mask` side (CAN-2).
    a = ToolEvent(tool="deploy", args={"service": "api", "attempt_id": 1})
    b = ToolEvent(tool="deploy", args={"service": "api", "attempt_id": 2})
    assert canonicalize(a) != canonicalize(b)


def test_uuid_values_are_masked() -> None:
    a = ToolEvent(tool="get", args={"ref": "550e8400-e29b-41d4-a716-446655440000"})
    b = ToolEvent(tool="get", args={"ref": "f47ac10b-58cc-4372-a567-0e02b2c3d479"})
    assert canonicalize(a) == canonicalize(b)


def test_iso_timestamp_values_are_masked() -> None:
    a = ToolEvent(tool="log", args={"at": "2026-06-09T08:00:00Z"})
    b = ToolEvent(tool="log", args={"at": "2026-06-09T09:30:15.123+02:00"})
    assert canonicalize(a) == canonicalize(b)


def test_nested_volatile_is_masked() -> None:
    a = ToolEvent(tool="call", args={"payload": {"q": "foo", "trace_id": "t-1"}})
    b = ToolEvent(tool="call", args={"payload": {"q": "foo", "trace_id": "t-2"}})
    assert canonicalize(a) == canonicalize(b)


def test_volatile_inside_list_is_masked() -> None:
    a = ToolEvent(tool="batch", args={"items": ["x", "550e8400-e29b-41d4-a716-446655440000"]})
    b = ToolEvent(tool="batch", args={"items": ["x", "f47ac10b-58cc-4372-a567-0e02b2c3d479"]})
    assert canonicalize(a) == canonicalize(b)


def test_meaningful_list_values_are_kept() -> None:
    a = ToolEvent(tool="batch", args={"items": ["x", "y"]})
    b = ToolEvent(tool="batch", args={"items": ["x", "z"]})
    assert canonicalize(a) != canonicalize(b)


# --- negative: no false merges -------------------------------------------------


def test_different_tool_differs() -> None:
    a = ToolEvent(tool="read", args={"path": "a.py"})
    b = ToolEvent(tool="write", args={"path": "a.py"})
    assert canonicalize(a) != canonicalize(b)


def test_different_meaningful_arg_differs() -> None:
    a = ToolEvent(tool="read", args={"path": "a.py"})
    b = ToolEvent(tool="read", args={"path": "b.py"})
    assert canonicalize(a) != canonicalize(b)


def test_non_uuid_string_is_kept() -> None:
    a = ToolEvent(tool="get", args={"ref": "alpha"})
    b = ToolEvent(tool="get", args={"ref": "beta"})
    assert canonicalize(a) != canonicalize(b)


def test_nested_meaningful_difference_differs() -> None:
    a = ToolEvent(tool="call", args={"payload": {"q": "foo", "trace_id": "t-1"}})
    b = ToolEvent(tool="call", args={"payload": {"q": "bar", "trace_id": "t-1"}})
    assert canonicalize(a) != canonicalize(b)


# --- CAN-3: per-tool keep override ---------------------------------------------


def test_keep_preserves_uuid_valued_id() -> None:
    masks = {"read_file": ToolMask(keep=frozenset({"file_id"}))}
    a = ToolEvent(tool="read_file", args={"file_id": "550e8400-e29b-41d4-a716-446655440000"})
    b = ToolEvent(tool="read_file", args={"file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"})
    # Without config the UUID values are masked by shape -> false merge.
    assert canonicalize(a) == canonicalize(b)
    # With keep, the distinguishing id is preserved -> distinct tokens.
    assert canonicalize(a, masks) != canonicalize(b, masks)


def test_keep_applies_to_nested_keys() -> None:
    masks = {"call": ToolMask(keep=frozenset({"trace_id"}))}
    a = ToolEvent(tool="call", args={"payload": {"q": "foo", "trace_id": "t-1"}})
    b = ToolEvent(tool="call", args={"payload": {"q": "foo", "trace_id": "t-2"}})
    assert canonicalize(a) == canonicalize(b)  # default: nested trace_id masked
    assert canonicalize(a, masks) != canonicalize(b, masks)  # keep -> visible


def test_keep_for_other_tool_does_not_apply() -> None:
    masks = {"other_tool": ToolMask(keep=frozenset({"ref"}))}
    a = ToolEvent(tool="get", args={"ref": "550e8400-e29b-41d4-a716-446655440000"})
    b = ToolEvent(tool="get", args={"ref": "f47ac10b-58cc-4372-a567-0e02b2c3d479"})
    # Tool name absent from config -> default masking still merges these UUIDs.
    assert canonicalize(a, masks) == canonicalize(b, masks)


def test_tool_masks_none_matches_default() -> None:
    event = ToolEvent(tool="read_file", args={"file_id": "550e8400-e29b-41d4-a716-446655440000"})
    assert canonicalize(event, None) == canonicalize(event)


def test_keep_token_is_deterministic() -> None:
    masks = {"read_file": ToolMask(keep=frozenset({"file_id"}))}
    event = ToolEvent(tool="read_file", args={"file_id": "abc", "path": "/x"})
    assert canonicalize(event, masks) == canonicalize(event, masks)


def test_keep_is_case_insensitive() -> None:
    # The kept key name matches the arg key regardless of casing, consistent
    # with the case-insensitive default masking.
    masks = {"read_file": ToolMask(keep=frozenset({"File_ID"}))}
    a = ToolEvent(tool="read_file", args={"file_id": "550e8400-e29b-41d4-a716-446655440000"})
    b = ToolEvent(tool="read_file", args={"file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"})
    assert canonicalize(a, masks) != canonicalize(b, masks)


def test_empty_keep_is_default_masking() -> None:
    masks = {"read_file": ToolMask()}
    a = ToolEvent(tool="read_file", args={"file_id": "550e8400-e29b-41d4-a716-446655440000"})
    b = ToolEvent(tool="read_file", args={"file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"})
    assert canonicalize(a, masks) == canonicalize(b, masks)
