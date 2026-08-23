"""ResourceDetector (class iii: resource / lifecycle runaways).

Covers the sub-agent RAM-exhaustion loop and the crash-resume loop -- both
invisible to a tool-call-only model. It is disabled by default; once enabled,
its signal is actionable and can select STOP under the configured policy.
Memory enforcement remains deferred to a later runtime phase.

Signals: sub-agent spawn count, delegation depth, run-restart count and a hard
runtime timeout (RES-2). Time is represented by lifecycle events injected at
the runtime boundary; this detector never reads a clock or interprets an
adapter-specific timestamp. RES-3 (RSS) remains a later addition to the same
``ResourceState``/``resource_delta`` shape.
Cross-run state persists in local SQLite via the ordinary journal-replay path
(see ``advance_resource_state``'s docstring); no baseline seed is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pathlayer.detectors.base import Detector, RuntimeLimitProvider, Signal
from pathlayer.events import EventKind, ToolEvent

# Consumed as a machine-matched prefix (not just display text) by
# pathlayer.policies.base._triggered_overrides via the config.py wiring of
# conditional_overrides[ResourceDetector.name] -- see test_resource.py's
# test_timeout_reason_starts_with_the_exported_prefix.
DURATION_REASON_PREFIX = "run duration reached"


def _validated_depth(value: object) -> tuple[int, bool]:
    """Return ``(depth, invalid)``. Bad input yields ``(0, True)``.

    ``bool`` is an ``int`` subclass and must be rejected explicitly, same trap
    ``budget_delta``'s ``_validated_cost`` already guards against.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0, True
    return value, False


@dataclass(frozen=True, slots=True)
class ResourceState:
    """Public, immutable fold of resource counters and timeout lifecycle.

    ``max_depth`` has a dual meaning depending on where it appears: as a
    *delta* (the return value of :func:`resource_delta` for one event) it is
    that event's own reported depth; as a *state* (accumulated via
    :func:`advance_resource_state`) it is the running high-water mark across
    the whole history -- the same delta/state duality ``BudgetState`` uses for
    ``captured_usd``, just combined with ``max`` instead of ``+``.

    ``invalid_depths`` counts events whose ``metadata["depth"]`` was present
    but unusable (not a non-negative, non-bool ``int``) and was therefore
    ignored, mirroring ``BudgetState.invalid_costs``.

    ``timed_out`` is sticky once a ``RUN_TIMEOUT`` event appears. Its binary
    state is what makes the configured duration an exact hard ceiling.
    """

    spawn_count: int = 0
    restart_count: int = 0
    max_depth: int = 0
    invalid_depths: int = 0
    timed_out: bool = False


def resource_delta(event: ToolEvent) -> ResourceState:
    """Return the deterministic contribution of one event.

    ``spawn_count``/``restart_count`` are 1 iff the event kind matches;
    ``max_depth`` (delta meaning: this event's own depth) comes from
    ``event.metadata["depth"]`` on *any* event kind -- depth can ride on an
    ordinary ``TOOL_CALL``/``LLM_CALL``, not just spawn/restart events, so
    restricting to those kinds would silently drop depth observations.
    """
    spawn = 1 if event.kind is EventKind.SUBAGENT_SPAWN else 0
    restart = 1 if event.kind is EventKind.RUN_RESTART else 0
    timed_out = event.kind is EventKind.RUN_TIMEOUT

    invalid_depths = 0
    max_depth = 0
    if "depth" in event.metadata:
        depth, invalid = _validated_depth(event.metadata["depth"])
        max_depth = depth
        invalid_depths = int(invalid)

    return ResourceState(
        spawn_count=spawn,
        restart_count=restart,
        max_depth=max_depth,
        invalid_depths=invalid_depths,
        timed_out=timed_out,
    )


def advance_resource_state(state: ResourceState, delta: ResourceState) -> ResourceState:
    """Add one event's contribution to an accumulated resource state.

    Sums ``spawn_count``/``restart_count``/``invalid_depths``; takes the
    running max for ``max_depth`` so a depth sequence like ``1 -> 3 -> 1``
    still reports confidence from the peak of ``3``, not the latest reading.
    """
    return ResourceState(
        spawn_count=state.spawn_count + delta.spawn_count,
        restart_count=state.restart_count + delta.restart_count,
        max_depth=max(state.max_depth, delta.max_depth),
        invalid_depths=state.invalid_depths + delta.invalid_depths,
        timed_out=state.timed_out or delta.timed_out,
    )


class ResourceDetector(Detector, RuntimeLimitProvider):
    name = "resource"

    def __init__(
        self,
        max_spawn_count: int = 5,
        max_delegation_depth: int = 3,
        max_restart_count: int = 3,
        max_duration_seconds: int = 14400,
    ) -> None:
        self._max_spawn_count = _validated_threshold(max_spawn_count, "max_spawn_count")
        self._max_delegation_depth = _validated_threshold(
            max_delegation_depth, "max_delegation_depth"
        )
        self._max_restart_count = _validated_threshold(max_restart_count, "max_restart_count")
        self._max_duration_seconds = _validated_threshold(
            max_duration_seconds, "max_duration_seconds"
        )

    @property
    def max_duration_seconds(self) -> int:
        """Hard runtime limit consumed by the runtime watchdog."""
        return self._max_duration_seconds

    def evaluate(self, history: Sequence[ToolEvent]) -> Signal:
        state = self.create_state()
        for event in history:
            state = self._advance(state, event)
        return self._signal(state)

    def create_state(self) -> ResourceState:
        return ResourceState()

    def step(self, state: ResourceState, event: ToolEvent) -> tuple[ResourceState, Signal]:
        next_state = self._advance(state, event)
        return next_state, self._signal(next_state)

    def _advance(self, state: ResourceState, event: ToolEvent) -> ResourceState:
        return advance_resource_state(state, resource_delta(event))

    def _signal(self, state: ResourceState) -> Signal:
        spawn_conf = min(1.0, state.spawn_count / self._max_spawn_count)
        depth_conf = min(1.0, state.max_depth / self._max_delegation_depth)
        restart_conf = min(1.0, state.restart_count / self._max_restart_count)
        # A timeout is a discrete event emitted at the exact deadline. Keeping
        # the signal binary prevents the default 0.9 STOP threshold from firing
        # at 90% of a value configured and documented as a hard maximum.
        duration_conf = 1.0 if state.timed_out else 0.0

        # Each candidate carries its own (observed, threshold) alongside the
        # reason string, so the "also at threshold" diagnostics below read
        # them straight off `candidates` instead of a second, easily-desynced
        # lookup keyed by name.
        candidates = (
            (
                spawn_conf,
                "spawn",
                f"{state.spawn_count} sub-agent spawn(s) / {self._max_spawn_count} threshold",
                state.spawn_count,
                self._max_spawn_count,
            ),
            (
                depth_conf,
                "depth",
                f"{state.max_depth} delegation depth / {self._max_delegation_depth} threshold",
                state.max_depth,
                self._max_delegation_depth,
            ),
            (
                restart_conf,
                "restart",
                f"{state.restart_count} run restart(s) / {self._max_restart_count} threshold",
                state.restart_count,
                self._max_restart_count,
            ),
            (
                duration_conf,
                "duration",
                f"{DURATION_REASON_PREFIX} {self._max_duration_seconds}s hard limit",
                int(state.timed_out),
                1,
            ),
        )
        # Before a timeout, preserve RES-1's historical tie-break order (spawn,
        # depth, restart). Once the one-shot hard-ceiling fact exists, duration
        # must be the actionable reason even if an older counter is also at
        # 1.0: config.py's duration-only STOP override deliberately keys off
        # that reason while the co-occurring counters remain diagnostic.
        selected = candidates[-1] if state.timed_out else max(candidates[:-1], key=lambda c: c[0])
        confidence, winner, reason, _, _ = selected

        diagnostics: list[str] = []
        if state.invalid_depths:
            diagnostics.append(f"{state.invalid_depths} invalid depth value(s) ignored")
        for conf, name, _, observed, threshold in candidates:
            if name != winner and conf >= 1.0:
                diagnostics.append(f"{name} also at threshold ({observed}/{threshold})")

        if confidence <= 0.0:
            return Signal(self.name, confidence=0.0, diagnostics=tuple(diagnostics))
        return Signal(
            self.name,
            confidence=confidence,
            reason=reason,
            diagnostics=tuple(diagnostics),
        )


def _validated_threshold(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1")
    return value
