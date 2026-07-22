"""Actuation context (ACT-1): the immutable explanation handed to an Actuator.

ACT-0 wired an already-arbitrated :class:`~pathlayer.policies.actions.Action` to
``Actuator.apply(action, post_mortem)``, but a non-STOP tier received no
post-mortem and therefore no detector, score, reason or current-event context --
enough to *stop* a run, not enough to *explain* a WARN to the agent. ACT-1 adds
this minimal, framework-agnostic context so an actuator can execute an
intermediate tier (WARN..ESCALATE) without re-running arbitration.

The context is produced by :class:`~pathlayer.pipeline.Pipeline` from the very
same ``signals``/``evidence`` that drove the decision, so the actuator never
re-interprets signals -- it only executes a decision already made (same split as
adapters: capture, never decide). It is frozen/immutable on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlayer.detectors.base import Signal
from pathlayer.events import EventKind
from pathlayer.policies.actions import Action
from pathlayer.postmortem import PostMortem


@dataclass(slots=True, frozen=True)
class EventSnapshot:
    """A minimal, non-sensitive view of the event that triggered the action.

    Deliberately excludes ``args``/``result``/``metadata``: an actuator may
    inject this context into the agent's own prompt (e.g. a WARN
    ``SystemMessage``), and those fields can carry secrets, PII or file contents
    (same concern as the HTTP error leak tracked by SEC-3, #133). Only these
    scalars -- safe to surface to the agent -- are exposed.
    """

    tool: str
    kind: EventKind
    ts: float
    index: int
    cost_usd: float


@dataclass(slots=True, frozen=True)
class ActuationContext:
    """Everything an :class:`~pathlayer.policies.actuators.Actuator` needs to act.

    - ``action``: the already-arbitrated tier (the actuator does not decide it).
    - ``fused``: the fused confidence in ``[0, 1]`` -- the value actually
      compared to the tier thresholds (:meth:`Policy.fuse`), not a single
      detector's raw confidence. Under an ARB-4 override this can sit below the
      committed tier's threshold; ``reason`` names the real driver in that case.
    - ``reason``: a concise human-readable driver (from the dominant/override
      signal), reused from the pipeline's post-mortem reasoning.
    - ``signals``: the raw detector signals at this tick, in detector order (the
      same honest, un-interpreted view as a dry-run ``WouldBeAction``).
    - ``event``: the safe snapshot of the triggering event (see
      :class:`EventSnapshot`).
    - ``post_mortem``: non-``None`` exactly when ``action`` is
      :attr:`Action.STOP`, and then identical to ``Pipeline.post_mortem``.
    """

    action: Action
    fused: float
    reason: str
    signals: tuple[Signal, ...]
    event: EventSnapshot
    post_mortem: PostMortem | None = None
