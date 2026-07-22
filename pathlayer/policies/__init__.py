"""Policy layer: arbitration + graded actions + their execution (ACT-0).

Fuses detector signals into a single decision and maps the continuous
confidence to a graded action (warn -> summarize -> fallback -> escalate ->
stop) rather than a binary verdict. This layer also owns *executing* that
decision against a live agent (:class:`Actuator`, :func:`wire`): arbitration
(``Policy``) and execution (``Actuator``) are both "policy" in the
architecture sense -- deciding and acting on the decision, as opposed to
``adapters/`` (capture only) or ``detectors/`` (signal extraction only).

Framework-specific actuators (e.g. ``LangGraphActuator``) are *not*
re-exported here, the same way ``pathlayer.adapters.langgraph`` is not
re-exported from ``pathlayer.adapters``: they carry an optional dependency and
are imported directly from their module.
"""

from pathlayer.policies.actions import Action
from pathlayer.policies.actuation import ActuationContext, EventSnapshot
from pathlayer.policies.actuators import Actuator, LoggingActuator
from pathlayer.policies.base import HysteresisState, OverrideRule, Policy, SuppressionRule
from pathlayer.policies.wiring import wire

__all__ = [
    "Action",
    "ActuationContext",
    "Actuator",
    "EventSnapshot",
    "HysteresisState",
    "LoggingActuator",
    "OverrideRule",
    "Policy",
    "SuppressionRule",
    "wire",
]
