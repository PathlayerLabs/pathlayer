"""OpenClaw lifecycle helpers (run identity + hook classification).

Lifecycle hooks (``before_agent_run``, ``agent_end``) are handled by the run
recorder and are **not** forced into :class:`~pathlayer.events.ToolEvent`.
Only ``after_tool_call`` is translated into tool events for the detector
pipeline. ``before_tool_call`` is a decision query, not an observation: the
recorder answers it and never records it. ``before_compaction`` and
``after_compaction`` are journaled as facts about the run and never analyzed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

BEFORE_AGENT_RUN = "before_agent_run"
BEFORE_TOOL_CALL = "before_tool_call"
AFTER_TOOL_CALL = "after_tool_call"
AGENT_END = "agent_end"
LLM_OUTPUT = "llm_output"
BEFORE_COMPACTION = "before_compaction"
AFTER_COMPACTION = "after_compaction"
COMPACTION_HOOKS = frozenset({BEFORE_COMPACTION, AFTER_COMPACTION})

# Longest error text kept from an ``agent_end`` (manifest, trace, health reason).
MAX_ATTEMPT_ERROR_CHARS = 200

# Filesystem-safe directory names derived from external run IDs.
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class OpenClawHookKind(str, Enum):
    RUN_START = "run_start"
    TOOL = "tool"
    RUN_END = "run_end"
    LLM = "llm"
    TOOL_DECISION = "tool_decision"
    COMPACTION = "compaction"
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class OpenClawContext:
    run_id: str | None
    session_key: str | None
    agent_id: str | None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def extract_openclaw_context(payload: Any) -> OpenClawContext:
    """Extract correlation fields from a hook payload."""
    if not isinstance(payload, dict):
        return OpenClawContext(run_id=None, session_key=None, agent_id=None)
    ctx = _as_dict(payload.get("ctx"))
    return OpenClawContext(
        run_id=extract_openclaw_run_id(payload),
        session_key=_first_str(ctx.get("sessionKey")),
        agent_id=_first_str(ctx.get("agentId")),
    )


def extract_openclaw_run_id(payload: Any) -> str | None:
    """Return the canonical run identifier for one OpenClaw agent run.

    Priority: ``ctx.runId`` then ``event.runId``. ``event.toolCallId`` is
    **never** used as a run identifier. A blank ``ctx.runId`` counts as absent:
    it would otherwise shadow a valid ``event.runId`` and then be refused as a
    run directory name, losing the payload before anything is journaled.
    """
    if not isinstance(payload, dict):
        return None
    ctx = _as_dict(payload.get("ctx"))
    event = _as_dict(payload.get("event"))
    return _first_str(ctx.get("runId"), event.get("runId"))


def classify_openclaw_hook(payload: Any) -> OpenClawHookKind:
    if not isinstance(payload, dict):
        return OpenClawHookKind.IGNORED
    hook = payload.get("hook")
    if hook == BEFORE_AGENT_RUN:
        return OpenClawHookKind.RUN_START
    if hook == AFTER_TOOL_CALL:
        return OpenClawHookKind.TOOL
    if hook == AGENT_END:
        return OpenClawHookKind.RUN_END
    if hook == LLM_OUTPUT:
        return OpenClawHookKind.LLM
    if hook == BEFORE_TOOL_CALL:
        return OpenClawHookKind.TOOL_DECISION
    if hook in COMPACTION_HOOKS:
        return OpenClawHookKind.COMPACTION
    return OpenClawHookKind.IGNORED


def attempt_end_failed(event: Any) -> bool:
    """Whether the attempt an ``agent_end`` closes failed.

    OpenClaw reports a failed attempt in three ways, and only the first sets
    ``success`` to ``False``:

    - ``success: false``: the attempt was aborted, by the user or by an
      exception (a prompt error, such as the context-overflow precheck);
    - a non-blank ``error``;
    - ``success: true`` with a last assistant message whose ``stopReason`` is
      ``"error"``: the provider answered with an error (a context overflow it
      detected itself, an overload), which agent-core turns into an assistant
      message instead of raising. OpenClaw then compacts and retries under the
      same ``runId``, or gives up and tells the user, with no further hook.

    Only the last message counts: an error the agent recovered from earlier in
    the session history says nothing about this attempt.
    """
    if not isinstance(event, dict):
        return False
    return event.get("success") is False or attempt_end_error(event) is not None


def attempt_end_error(event: Any) -> str | None:
    """The error text of a failed attempt's ``agent_end``, when OpenClaw gives one.

    ``event.error`` first, then the last assistant message's ``errorMessage``,
    truncated to :data:`MAX_ATTEMPT_ERROR_CHARS`. ``None`` when the end reports
    no error text (``success: false`` alone, or no failure at all).
    """
    if not isinstance(event, dict):
        return None
    error = event.get("error")
    if isinstance(error, str) and error.strip():
        return _truncate(error.strip())
    last_assistant_error = _last_assistant_error(event.get("messages"))
    if last_assistant_error is not None:
        return _truncate(last_assistant_error)
    return None


def _last_assistant_error(messages: Any) -> str | None:
    """The error of the last message when it is an assistant message that ended in error."""
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return None
    if last.get("stopReason") != "error":
        return None
    message = last.get("errorMessage")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return "the provider ended the reply with an error"


def _truncate(text: str) -> str:
    if len(text) <= MAX_ATTEMPT_ERROR_CHARS:
        return text
    return text[: MAX_ATTEMPT_ERROR_CHARS - 3] + "..."


def sanitize_run_id_for_path(run_id: str, *, max_length: int = 120) -> str:
    """Sanitize an external run ID for use as a single path component."""
    trimmed = run_id.strip()
    if not trimmed:
        raise ValueError("run_id must not be empty")
    if trimmed in {".", ".."} or trimmed.startswith(("/", "\\")):
        raise ValueError(f"unsafe run_id: {run_id!r}")
    if "/" in trimmed or "\\" in trimmed or ".." in trimmed:
        raise ValueError(f"unsafe run_id: {run_id!r}")
    safe = _UNSAFE_PATH_CHARS.sub("_", trimmed).strip("._")
    if not safe:
        raise ValueError(f"run_id sanitizes to empty: {run_id!r}")
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe


def _first_str(*values: Any) -> str | None:
    """The first present value as a string; ``None`` and blank strings are absent."""
    for value in values:
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        return str(value)
    return None
