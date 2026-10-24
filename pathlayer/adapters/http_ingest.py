"""Generic local HTTP ingestion adapter (out-of-process capture transport).

Some agent frameworks run as a separate process from Pathlayer -- OpenClaw's
Gateway, for instance, is a Node.js daemon. For those, an in-process adapter
like :class:`~pathlayer.adapters.langgraph.LangGraphAdapter` is not an option:
there is no shared call stack to hook into. Instead, a small framework-side
"forwarder" (as dumb as possible: read the framework's native hook payload,
POST it here, done) sends each event to a local HTTP endpoint that this
adapter listens on. All the actual translation logic (turning a framework's
raw hook payload into a normalized :class:`~pathlayer.events.ToolEvent`) stays
in Python, where it is reviewed, typed and unit-tested like everything else.

This adapter is framework-agnostic on purpose: it dispatches each incoming
request to a per-source translator function, so the same transport serves
OpenClaw today (see :mod:`pathlayer.adapters.openclaw`) and any other
out-of-process framework tomorrow -- only a new translator (and a new tiny
forwarder on the framework side) is needed, not a new HTTP server.

Security: binds to loopback only -- ``host`` is restricted to
``_ALLOWED_HOSTS`` (rejects e.g. ``"0.0.0.0"`` at construction time), so a
configuration mistake can never expose this endpoint to the network. There is
no authentication in V1, matching "local-first, zero config": any local
process can POST to it, an accepted trade-off on a single-user dev machine.
Request bodies are capped (``_MAX_BODY_BYTES``) and the socket has a read
timeout, so a misbehaving local client cannot tie up a request thread
indefinitely or exhaust memory with an oversized body.

Ordering caveat: ``ts`` reflects arrival order over the network, not an
in-process guarantee like :class:`~pathlayer.adapters.langgraph.LangGraphAdapter`
provides -- there is no shared call stack to serialize against upstream, only
the order requests happen to reach this server.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pathlayer.adapters.base import Adapter, EventSink
from pathlayer.events import ToolEvent

logger = logging.getLogger(__name__)

# One entry per known "source" (framework). Receives the request's raw
# ``payload`` field and returns zero or more normalized events. Must be pure
# and reasonably fast: it runs on the HTTP server's request thread, under the
# adapter's lock, so a slow or raising translator stalls/fails that request
# (never the server itself -- see ``_Handler.do_POST``).
Translator = Callable[[Any], list[ToolEvent]]

# Optional, per source: answers a request with a decision for the forwarder
# (for instance "block this tool call"), returned under the response's
# ``decision`` key. ``None`` means no decision. It runs after the translator,
# under the same lock, so it sees every event ingested before its request.
# A raising decider is logged and answered with no decision: a forwarder that
# acts on decisions must treat their absence as "allow" (fail open).
Decider = Callable[[Any], Mapping[str, Any] | None]


class _UnknownSourceError(LookupError):
    """Raised only when no translator is registered for a source."""


INGEST_PATH = "/ingest"

# Enforced in __init__: no configuration mistake should be able to turn this
# unauthenticated endpoint into one reachable from the network (e.g. passing
# "0.0.0.0" by accident). Limited to what ThreadingHTTPServer's IPv4-only
# default socket family actually supports -- "::1" fails to bind with a
# "nodename nor servname" error, so it is deliberately excluded rather than
# accepted-but-broken.
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})

# A single tool call's args/result should never approach this; caps memory
# use and the time spent reading a request from a misbehaving local client.
_MAX_BODY_BYTES = 1_000_000

# Bounds how long a stalled/slow client can hold a request thread (e.g. a
# partial write that never completes) -- keeps this in the same spirit as
# the body-size cap above.
_SOCKET_TIMEOUT_S = 5.0

# Depth of the kernel's accept queue. ``socketserver`` defaults it to 5, which
# is below the burst an out-of-process forwarder produces: sends are
# fire-and-forget and concurrent, so a handful of hooks are routinely in flight
# at once while this process' single accept loop competes for the GIL with the
# request threads it just spawned. A connection that does not fit the queue is
# dropped by the kernel *before* the server accepts it, so the request is never
# read and its event never exists -- and the forwarder, which does not retry,
# only logs a warning. 128 is deeper than any plausible local burst, costs
# nothing while idle, and is clamped by the OS (``net.core.somaxconn``) anyway.
_LISTEN_BACKLOG = 128


class _IngestHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` with an accept queue sized for concurrent posts."""

    request_queue_size = _LISTEN_BACKLOG


class HttpIngestAdapter(Adapter):
    """Loopback HTTP server that turns POSTed framework events into ``ToolEvent``s.

    Usage::

        adapter = HttpIngestAdapter(sink, translators={"openclaw": translate_openclaw_event})
        adapter.attach()
        print(adapter.port)  # tell the out-of-process forwarder where to POST
        ...
        adapter.detach()

    A POST to ``/ingest`` must carry a JSON body ``{"source": str, "payload":
    <framework-specific>}``. ``source`` selects the translator; ``payload`` is
    passed to it as-is. The response is ``{"ok": true}``, plus a ``decision``
    object when a decider registered for the source returns one.
    """

    def __init__(
        self,
        sink: EventSink,
        translators: Mapping[str, Translator],
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        deciders: Mapping[str, Decider] | None = None,
    ) -> None:
        super().__init__(sink)
        if not translators:
            raise ValueError("translators must not be empty")
        if host not in _ALLOWED_HOSTS:
            raise ValueError(
                f"host must be loopback ({sorted(_ALLOWED_HOSTS)}), got {host!r} -- "
                "this endpoint has no authentication and must never be reachable "
                "from the network"
            )
        self._translators = dict(translators)
        self._deciders = dict(deciders or {})
        self._host = host
        self._requested_port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ts = 0

    @property
    def port(self) -> int:
        """The bound port, resolved after :meth:`attach` (e.g. when ``port=0``)."""
        if self._server is None:
            raise RuntimeError("adapter is not attached")
        return int(self._server.server_address[1])

    def attach(self) -> None:
        """Start the loopback HTTP server in a background thread.

        Resets the ``ts`` counter, like every other adapter's ``attach``, so a
        fresh run is reproducible.
        """
        if self._server is not None:
            raise RuntimeError("adapter is already attached")
        with self._lock:
            self._ts = 0
        self._server = _IngestHTTPServer((self._host, self._requested_port), _make_handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def detach(self) -> None:
        """Stop the server and release the socket."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def _ingest(self, source: str, payload: Any) -> Mapping[str, Any] | None:
        """Translate one request body, emit the resulting events, then decide.

        Raises ``_UnknownSourceError`` for an unknown source; lets a translator's own
        exceptions propagate. The HTTP handler turns both into an error
        response without crashing the server thread (see ``_Handler.do_POST``).

        The lock spans translation *and* ``ts`` allocation *and* emission, so
        concurrent requests (``ThreadingHTTPServer`` handles each on its own
        thread) never interleave: ``ts`` stays contiguous and the downstream
        sink sees one event at a time, matching every other adapter's
        contract.

        Returns the source's decision for this request, if any (see
        :data:`Decider`).
        """
        try:
            translator = self._translators[source]
        except KeyError as exc:
            raise _UnknownSourceError(source) from exc
        with self._lock:
            events = translator(payload)
            for event in events:
                event.ts = self._ts
                self._ts += 1
                self.emit(event)
            decider = self._deciders.get(source)
            if decider is None:
                return None
            try:
                return decider(payload)
            except Exception:
                logger.exception("decider for source=%r failed; answering no decision", source)
                return None


def _make_handler(adapter: HttpIngestAdapter) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        timeout = _SOCKET_TIMEOUT_S

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("%s - %s", self.address_string(), format % args)

        def do_POST(self) -> None:
            if self.path != INGEST_PATH:
                self._respond(404, {"error": f"unknown path {self.path!r}"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                self._respond(400, {"error": "invalid Content-Length header"})
                return
            if length > _MAX_BODY_BYTES:
                # Drain in bounded chunks (rather than skipping the read) so
                # the client's write completes instead of getting a TCP RST
                # from an early response on an unread socket -- while still
                # never buffering more than one chunk of the oversized body.
                self._drain(length)
                self._respond(400, {"error": f"body exceeds {_MAX_BODY_BYTES} bytes"})
                return
            raw = self.rfile.read(length) if length else b""
            try:
                body: Any = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                self._respond(400, {"error": f"invalid JSON: {exc}"})
                return
            if not isinstance(body, dict) or "source" not in body or "payload" not in body:
                self._respond(
                    400, {"error": "body must be a JSON object with 'source' and 'payload' keys"}
                )
                return
            source = body["source"]
            if not isinstance(source, str):
                self._respond(
                    400, {"error": f"source must be a string, got {type(source).__name__}"}
                )
                return
            try:
                decision = adapter._ingest(source, body["payload"])
            except _UnknownSourceError:
                self._respond(400, {"error": f"unknown source {source!r}"})
                return
            except Exception:  # never let a bad event kill the server thread
                logger.exception("translator for source=%r failed", source)
                self._respond(500, {"error": "internal error (see server log)"})
                return
            if decision is None:
                self._respond(200, {"ok": True})
            else:
                self._respond(200, {"ok": True, "decision": dict(decision)})

        def _drain(self, length: int, chunk_size: int = 65_536) -> None:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)

        def _respond(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _Handler
