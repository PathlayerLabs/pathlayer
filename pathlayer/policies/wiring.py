"""Wiring: connect a Pipeline's decisions to an Actuator's effects.

``Pipeline.process()`` already computes and returns the graded
:class:`~pathlayer.policies.Action` for each event; :meth:`Pipeline.sink`
simply discards it (see ``pathlayer/pipeline.py``). :func:`wire` is the only
piece of glue needed to change that: it builds an
:data:`~pathlayer.adapters.base.EventSink` that runs the pipeline and forwards
its decision to an :class:`~pathlayer.policies.actuators.Actuator`.

The returned object remains callable as an EventSink and additionally exposes
``start_run()``/``end_run()``. Those lifecycle hooks own the RES-2 watchdog;
the pipeline stays a pure, adapter-agnostic decision engine.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from pathlayer.adapters.base import RunAwareEventSink
from pathlayer.events import ToolEvent
from pathlayer.policies.actions import Action
from pathlayer.policies.actuators import Actuator
from pathlayer.runtime import Scheduler, WatchdogEventSink, wrap_with_rss_sampling

if TYPE_CHECKING:
    # Deferred to a type-checking-only import: pathlayer.pipeline imports
    # pathlayer.policies (for Action/Policy), so importing Pipeline here at
    # runtime would create a circular import (policies -> wiring -> pipeline
    # -> policies). wire() only needs Pipeline's public surface (process,
    # dry_run, post_mortem), never isinstance -- the annotation is enough.
    from pathlayer.pipeline import Pipeline


def wire(
    pipeline: Pipeline,
    actuator: Actuator,
    *,
    logger: logging.Logger | None = None,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    scheduler: Scheduler | None = None,
) -> RunAwareEventSink:
    """Build an :data:`EventSink` that runs ``pipeline`` then applies its Action.

    Only calls ``actuator.apply_with_context`` (ACT-1; its default delegates to
    ``actuator.apply``) on a *transition* to a new non-``NONE`` action -- the
    same "once per transition, not once per event" rule ``Pipeline`` itself uses
    for ``would_be_actions`` in dry-run (see ``Pipeline._observe``). Without
    this, every event after a ``STOP`` (which ``Pipeline.process`` keeps
    returning; see its ``_stopped`` short-circuit) would re-invoke the actuator
    for no reason -- harmless for the idempotent ``RunControl.request_drain``,
    but not a contract a generic ``Actuator`` should have to assume.

    The returned run-aware sink serializes adapter calls and the watchdog timer
    with one private lock. ``clock``/``monotonic``/``scheduler`` are injectable
    so deadline behavior can be tested without sleeping.

    Raises ``ValueError`` if ``pipeline.dry_run`` is ``True``: a dry-run
    pipeline's ``process()`` always returns ``Action.NONE`` (the action it
    *would* have taken lives in ``pipeline.would_be_actions`` instead), so
    wiring a real actuator to it would look wired while silently never firing.
    Use :attr:`Pipeline.would_be_actions` with a
    :class:`~pathlayer.policies.actuators.LoggingActuator` directly for
    dry-run tuning instead.

    Any exception raised by the actuator is caught and logged (fail-open): it
    must never propagate back through the sink into the adapter's callback and
    break event capture on a live agent run.
    """
    if pipeline.dry_run:
        raise ValueError(
            "wire() requires an enforcing pipeline (dry_run=False); a dry-run "
            "pipeline's process() always returns Action.NONE -- inspect "
            "pipeline.would_be_actions instead of wiring a real actuator to it"
        )
    log = logger if logger is not None else logging.getLogger(__name__)
    last_applied = Action.NONE

    def sink(event: ToolEvent) -> None:
        nonlocal last_applied
        action = pipeline.process(event)
        if action is last_applied:
            return
        # Update on every change, including a de-escalation to NONE (mirrors
        # Pipeline._observe/_last_would_be): otherwise a later re-escalation
        # to the *same* tier after a NONE in between would never be re-applied.
        last_applied = action
        if action is Action.NONE:
            return
        try:
            context = pipeline.actuation_context
            assert context is not None
            actuator.apply_with_context(context)
        except Exception:
            log.exception(
                "actuator.apply failed for action=%s; event capture continues", action.name
            )

    return WatchdogEventSink(
        wrap_with_rss_sampling(sink, max_duration_seconds=pipeline.max_duration_seconds),
        max_duration_seconds=pipeline.max_duration_seconds,
        history=lambda: pipeline.history,
        clock=clock,
        monotonic=monotonic,
        scheduler=scheduler,
        terminal=lambda: pipeline.stopped,
        logger=log,
    )
