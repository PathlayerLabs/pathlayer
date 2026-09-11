"""OpenClaw lifecycle helpers (run identity + hook classification).

Lifecycle hooks (``before_agent_run``, ``agent_end``) are handled by the run
recorder and are **not** forced into :class:`~pathlayer.events.ToolEvent`.
Only ``after_tool_call`` is translated into tool events for the detector
pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

BEFORE_AGENT_RUN = "before_agent_run"
AFTER_TOOL_CALL = "after_tool_call"
AGENT_END = "agent_end"
LLM_OUTPUT = "llm_output"

# Filesystem-safe directory names derived from external run IDs.
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class OpenClawHookKind(str, Enum):
    RUN_START = "run_start"
    TOOL = "tool"
    RUN_END = "run_end"
    LLM = "llm"
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
    **never** used as a run identifier.
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
    return OpenClawHookKind.IGNORED


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
    for value in values:
        if value is not None:
            return str(value)
    return None
