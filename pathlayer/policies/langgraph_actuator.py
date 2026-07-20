"""LangGraph actuator (ACT-0): drain a live run on STOP via ``RunControl``.

Uses LangGraph's native cooperative graceful-shutdown mechanism (>= 1.2,
``langgraph.runtime.RunControl`` / ``langgraph.errors.GraphDrained``):
thread-safe (a single attribute write), checkpointed, no new network
endpoint, no new timeout-safety problem. This is why LangGraph is the first
concrete actuator (see the ACT-0 plan): unlike an OpenClaw ``before_tool_call``
actuator, it needs neither PERF-2 (no synchronous blocking hook waiting on
Pathlayer) nor a new authenticated endpoint.

Only ``Action.STOP`` is enforced: WARN/SUMMARIZE/FALLBACK/ESCALATE have no
native LangGraph equivalent found for a cooperative, non-blocking action, and
implementing one is explicitly out of scope here (see ``ACT-1..4``, #29-32).

This module imports ``langgraph`` and so is an **optional** import, like
:mod:`pathlayer.adapters.langgraph`: install the ``langgraph`` extra
(``pip install pathlayer[langgraph]``). It is deliberately *not* re-exported
from :mod:`pathlayer.policies`.
"""

from __future__ import annotations

import threading

from langgraph.runtime import RunControl

from pathlayer.policies.actions import Action
from pathlayer.policies.actuators import Actuator
from pathlayer.postmortem import PostMortem

_DEFAULT_STOP_REASON = "pathlayer_stop"


class LangGraphActuator(Actuator):
    """``Action.STOP`` -> ``RunControl.request_drain()``; other tiers are no-ops.

    Lifecycle: one instance per graph run, mirroring
    :class:`~pathlayer.adapters.langgraph.LangGraphAdapter`. Call :meth:`attach`
    once before starting the run -- it creates a fresh ``RunControl`` (a
    drained ``RunControl`` cannot be reused, per its own docstring) -- then
    pass :attr:`control` to ``graph.invoke()``/``ainvoke()``::

        actuator = LangGraphActuator()
        actuator.attach()
        adapter = LangGraphAdapter(wire(pipeline, actuator))
        adapter.attach()
        try:
            result = graph.invoke(state, config=adapter.as_config(), control=actuator.control)
        except GraphDrained as e:
            ...  # checkpoint saved; e.reason carries the drain reason
        finally:
            adapter.detach()

    ``GraphDrained`` (``from langgraph.errors import GraphDrained``) must be
    caught by the integrator's own code around ``invoke``/``ainvoke``:
    Pathlayer does not own that call.

    A lock guards ``_control`` because :meth:`apply` is invoked from
    ``LangGraphAdapter``'s lock-held sink callback (a worker thread during
    ``ainvoke``) while :meth:`attach`/:attr:`control` may run on the caller's
    own thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._control = RunControl()

    @property
    def control(self) -> RunControl:
        """The current run's ``RunControl``; pass as ``control=`` to ``invoke``."""
        with self._lock:
            return self._control

    def attach(self) -> None:
        """Create a fresh ``RunControl`` for a new run. Call before ``adapter.attach()``."""
        with self._lock:
            self._control = RunControl()

    def apply(self, action: Action, post_mortem: PostMortem | None) -> None:
        if action is not Action.STOP:
            return
        reason = post_mortem.reason if post_mortem is not None else _DEFAULT_STOP_REASON
        with self._lock:
            self._control.request_drain(reason)
