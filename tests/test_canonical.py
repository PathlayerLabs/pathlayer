"""Tests for canonicalization (CAN-1).

Two axes matter and pull against each other:
- positive: identical-up-to-noise calls collapse to the same token;
- negative: genuinely different calls never collapse (no false merge).
"""

from __future__ import annotations

from pathlayer.detectors.canonical import canonicalize
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
    a = ToolEvent(tool="db", args={"table": "users", "order_id": 1, "created_at": "t1"})
    b = ToolEvent(tool="db", args={"table": "users", "order_id": 2, "created_at": "t2"})
    assert canonicalize(a) == canonicalize(b)


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
