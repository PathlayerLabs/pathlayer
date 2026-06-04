"""OpenClaw adapter (first concrete adapter).

TODO: wire into OpenClaw's per-action hooks and translate each tool call into
a normalized `ToolEvent`. See https://docs.openclaw.ai/tools/loop-detection/
for the framework's own hook surface.
"""

from __future__ import annotations

from pathlayer.adapters.base import Adapter


class OpenClawAdapter(Adapter):
    def attach(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError("OpenClaw hooks not wired yet")

    def detach(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError
