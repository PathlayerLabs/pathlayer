"""OpenClaw adapter (first concrete adapter) -- not yet implemented.

This is a placeholder stub: ``attach`` / ``detach`` raise ``NotImplementedError``
until the real capture is wired. The work (translate each OpenClaw per-action
hook into a normalized ``ToolEvent``) is tracked in ADP-1 (issue #40). The
framework's actual hook surface is to be confirmed there rather than linked to an
unverified external URL.
"""

from __future__ import annotations

from pathlayer.adapters.base import Adapter


class OpenClawAdapter(Adapter):
    def attach(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError("OpenClaw hooks not wired yet")

    def detach(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError
