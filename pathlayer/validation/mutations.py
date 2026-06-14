"""VAL-4: mutation engine -- inject a fault into a healthy trace, derive labels.

Each function takes an existing :class:`~pathlayer.validation.traces.Trace` and
returns a **new** trace with an injected fault plus machine-derived ground
truth. The input trace is never mutated (events are deep-copied).

Ground truth is written to ``trace.metadata["expectations"]`` as a list of
``{"detector", "should_fire", "earliest_event_idx"}`` entries. This per-detector
expectation -- not the coarse ``label`` -- is the authoritative signal the VAL-3
replay harness will score against, because the 3-class VAL-1 label
(``loop`` / ``no_progress`` / ``healthy``) cannot express "the budget detector
should fire at step N".

Label mapping (the coarse bucket; ``VALID_LABELS`` is deliberately left
unchanged so the VAL-1/VAL-2 corpus stays valid):

==================  =========  ============================================
mutation            label      authoritative expectation(s)
==================  =========  ============================================
inject_loop         loop       loop
inject_retry_storm  loop       retry (primary) + loop (a storm is a period-1
                               repetition, so the loop detector catches it too)
inject_no_progress  no_progress no_progress
inject_cost_spike   unchanged  budget  (a cost spike does not change the
                               behavioural class, only the spend, so the base
                               trace's label is preserved)
==================  =========  ============================================

Only ``LoopDetector`` is implemented today; the retry / no_progress / budget
detectors are stubs. The expectations are therefore forward-looking for those
three, and become checkable once the detectors land.
"""

from __future__ import annotations

import copy
from typing import Any

from pathlayer.events import ToolEvent
from pathlayer.validation.traces import Trace

# Events after the injection point before the targeted detector can reasonably
# be expected to fire. Conservative; consumers read the per-expectation
# ``earliest_event_idx`` rather than assuming a fixed delay.
DETECTION_MARGIN = 1


def _clone_events(events: list[ToolEvent]) -> list[ToolEvent]:
    return [copy.deepcopy(event) for event in events]


def _expectation(
    detector: str, earliest_event_idx: int, *, should_fire: bool = True
) -> dict[str, Any]:
    return {
        "detector": detector,
        "should_fire": should_fire,
        "earliest_event_idx": earliest_event_idx,
    }


def _build(
    base: Trace,
    events: list[ToolEvent],
    label: str,
    mutation: dict[str, Any],
    expectations: list[dict[str, Any]],
    note: str,
) -> Trace:
    metadata = copy.deepcopy(base.metadata)
    metadata["mutation"] = mutation
    metadata["expectations"] = expectations
    description = note if not base.description else f"{base.description} | {note}"
    return Trace(label=label, events=events, description=description, metadata=metadata)


def inject_loop(trace: Trace, start_idx: int, sequence_length: int, n_repeats: int) -> Trace:
    """Duplicate ``events[start_idx:start_idx+sequence_length]`` ``n_repeats`` times."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if start_idx < 0 or start_idx + sequence_length > len(trace.events):
        raise ValueError("start_idx + sequence_length is out of range")

    events = _clone_events(list(trace.events))
    block = events[start_idx : start_idx + sequence_length]
    insert_at = start_idx + sequence_length
    extra: list[ToolEvent] = []
    for _ in range(n_repeats):
        extra.extend(_clone_events(block))
    new_events = events[:insert_at] + extra + events[insert_at:]

    earliest = start_idx + 2 * sequence_length - 1
    mutation = {
        "type": "loop",
        "start_idx": start_idx,
        "sequence_length": sequence_length,
        "n_repeats": n_repeats,
    }
    note = f"injected {n_repeats}x loop of a {sequence_length}-action block at {start_idx}"
    return _build(trace, new_events, "loop", mutation, [_expectation("loop", earliest)], note)


def inject_retry_storm(trace: Trace, start_idx: int, n_events: int, tool: str, error: str) -> Trace:
    """Insert ``n_events`` identical failing calls to ``tool`` at ``start_idx``."""
    if n_events < 2:
        raise ValueError("n_events must be >= 2 to form a storm")
    if start_idx < 0 or start_idx > len(trace.events):
        raise ValueError("start_idx is out of range")

    events = _clone_events(list(trace.events))
    template = ToolEvent(
        tool=tool,
        args={"attempt": "retry"},
        result={"error": error},
        metadata={"error": error},
    )
    storm = [copy.deepcopy(template) for _ in range(n_events)]
    new_events = events[:start_idx] + storm + events[start_idx:]

    expectations = [
        _expectation("retry", start_idx + DETECTION_MARGIN),
        _expectation("loop", start_idx + 1),
    ]
    mutation = {
        "type": "retry_storm",
        "start_idx": start_idx,
        "n_events": n_events,
        "tool": tool,
        "error": error,
    }
    note = f"injected {n_events}x retry storm on {tool!r} (error {error!r}) at {start_idx}"
    return _build(trace, new_events, "loop", mutation, expectations, note)


def inject_no_progress(trace: Trace, start_idx: int) -> Trace:
    """Neutralize progress from ``start_idx`` onward (events stay distinct)."""
    if start_idx < 0 or start_idx >= len(trace.events):
        raise ValueError("start_idx is out of range")

    events = _clone_events(list(trace.events))
    for i in range(start_idx, len(events)):
        event = events[i]
        metadata = dict(event.metadata)
        metadata["progress"] = False
        metadata.pop("ground_truth", None)
        events[i] = ToolEvent(
            tool=event.tool,
            args=event.args,
            result=event.result,
            cost_usd=event.cost_usd,
            ts=event.ts,
            kind=event.kind,
            run_id=event.run_id,
            metadata=metadata,
        )

    earliest = start_idx + DETECTION_MARGIN
    mutation = {"type": "no_progress", "start_idx": start_idx}
    note = f"neutralized progress from index {start_idx}"
    return _build(
        trace, events, "no_progress", mutation, [_expectation("no_progress", earliest)], note
    )


def inject_cost_spike(trace: Trace, event_idx: int, multiplier: float) -> Trace:
    """Inflate the cost of the event at ``event_idx`` by ``multiplier``."""
    if multiplier <= 1.0:
        raise ValueError("multiplier must be > 1.0 to be a spike")
    if event_idx < 0 or event_idx >= len(trace.events):
        raise ValueError("event_idx is out of range")

    events = _clone_events(list(trace.events))
    event = events[event_idx]
    # Fall back to a unit base so the multiplier still bites on zero-cost events.
    base_cost = event.cost_usd if event.cost_usd else 1.0
    events[event_idx] = ToolEvent(
        tool=event.tool,
        args=event.args,
        result=event.result,
        cost_usd=base_cost * multiplier,
        ts=event.ts,
        kind=event.kind,
        run_id=event.run_id,
        metadata=dict(event.metadata),
    )

    mutation = {"type": "cost_spike", "event_idx": event_idx, "multiplier": multiplier}
    note = f"inflated cost x{multiplier} at index {event_idx}"
    # A cost spike does not change the behavioural class -> keep the base label.
    return _build(trace, events, trace.label, mutation, [_expectation("budget", event_idx)], note)
