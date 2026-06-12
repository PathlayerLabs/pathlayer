"""Tests for the in-memory replay adapter (INF-2).

Pin the contract the pipeline relies on: events reach the sink in order, the
adapter stays a passive observer, replay is deterministic, and detach is inert.
"""

from __future__ import annotations

from pathlayer.adapters import Adapter, InMemoryAdapter
from pathlayer.adapters.base import EventSink
from pathlayer.events import ToolEvent


def _events() -> list[ToolEvent]:
    return [
        ToolEvent(tool="read", args={"path": "a.py"}, ts=0.0),
        ToolEvent(tool="search", args={"q": "foo"}, ts=1.0),
        ToolEvent(tool="read", args={"path": "b.py"}, ts=2.0),
    ]


def test_is_an_adapter() -> None:
    adapter = InMemoryAdapter(lambda _event: None)
    assert isinstance(adapter, Adapter)


def test_attach_replays_events_in_order() -> None:
    captured: list[ToolEvent] = []
    sink: EventSink = captured.append
    events = _events()

    adapter = InMemoryAdapter(sink, events)
    adapter.attach()

    assert captured == events
    assert [e.tool for e in captured] == ["read", "search", "read"]


def test_attach_with_no_events_emits_nothing() -> None:
    captured: list[ToolEvent] = []
    InMemoryAdapter(captured.append).attach()
    assert captured == []


def test_feed_emits_single_event() -> None:
    captured: list[ToolEvent] = []
    adapter = InMemoryAdapter(captured.append)

    event = ToolEvent(tool="probe")
    adapter.feed(event)

    assert captured == [event]


def test_adapter_is_passive_observer() -> None:
    # The adapter must not mutate the events it forwards.
    events = _events()
    snapshot = [(e.tool, dict(e.args), e.ts) for e in events]

    InMemoryAdapter(lambda _event: None, events).attach()

    assert [(e.tool, dict(e.args), e.ts) for e in events] == snapshot


def test_replay_is_deterministic() -> None:
    events = _events()

    first: list[ToolEvent] = []
    InMemoryAdapter(first.append, events).attach()

    second: list[ToolEvent] = []
    InMemoryAdapter(second.append, events).attach()

    assert first == second


def test_detach_is_inert() -> None:
    captured: list[ToolEvent] = []
    adapter = InMemoryAdapter(captured.append, _events())
    adapter.attach()
    before = list(captured)

    adapter.detach()

    assert captured == before
