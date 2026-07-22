"""Actuator contract (ACT-0): turn an already-arbitrated Action into a real effect.

An :class:`Actuator` never decides -- the :class:`~pathlayer.policies.Action` is
computed upstream by :class:`~pathlayer.policies.Policy`/`Pipeline`. It only
executes it against a live agent (e.g. draining a LangGraph run, or -- in a
future, PERF-2-gated ticket -- blocking an OpenClaw tool call). This mirrors
the adapters/policies split: an adapter captures and never decides, an
actuator executes and never decides either.

See :func:`pathlayer.policies.wiring.wire` for how an ``Actuator`` is wired to
a :class:`~pathlayer.pipeline.Pipeline` without any change to the pipeline's
own pure evaluate-and-return contract.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from pathlayer.policies.actions import Action
from pathlayer.policies.actuation import ActuationContext
from pathlayer.postmortem import PostMortem


class Actuator(ABC):
    """Executes a graded :class:`Action` against a live agent run.

    ``apply`` is called once per *transition* to a new non-``NONE`` action
    (see :func:`~pathlayer.policies.wiring.wire`), never once per event for a
    sustained action. ``post_mortem`` is non-``None`` exactly when ``action``
    is :attr:`Action.STOP` (the only tier that builds one today); it is
    ``None`` for every other tier.

    ``apply`` should not raise for expected/control-flow reasons: the caller
    (``wire()``) is fail-open around it (an exception is logged, never
    propagated to the agent), but that is a last-resort safety net, not a
    substitute for an actuator handling its own expected failure modes.

    ACT-1 adds :meth:`apply_with_context`, the richer entry point ``wire()``
    actually calls. Override **exactly one** of ``apply`` / ``apply_with_context``:
    the default ``apply_with_context`` delegates to ``apply`` so every pre-ACT-1
    actuator keeps working unchanged, while an actuator that needs the detector
    /score/reason/current-event context (e.g. to inject a WARN into the agent)
    overrides ``apply_with_context`` instead.
    """

    @abstractmethod
    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        raise NotImplementedError

    def apply_with_context(self, context: ActuationContext) -> None:
        """Execute an arbitrated action with full context (ACT-1).

        Default: delegate to :meth:`apply` for backward compatibility. This is
        the method :func:`~pathlayer.policies.wiring.wire` invokes, never
        ``apply`` directly, so a context-aware actuator overrides this one while
        legacy actuators (which only implement ``apply``) are reached through
        this delegation with the exact same ``(action, post_mortem)`` pair.
        """
        self.apply(context.action, context.post_mortem)


class LoggingActuator(Actuator):
    """Reference no-op actuator: logs the action, no framework, no real effect.

    Safe default and the fixture used to test :func:`~pathlayer.policies.wiring.wire`
    without depending on any live agent framework.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger if logger is not None else logging.getLogger(__name__)

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        reason = post_mortem.reason if post_mortem is not None else action.name.lower()
        self._logger.warning(
            "actuator: %s (%s) -- no real effect (LoggingActuator)", action.name, reason
        )
