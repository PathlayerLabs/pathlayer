"""Tests for the generic loopback HTTP ingestion adapter.

Drives the adapter as a real black box over HTTP (stdlib ``urllib``) rather
than calling its internals directly, since the whole point of this adapter is
the wire contract an out-of-process forwarder relies on.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any

import pytest

from pathlayer.adapters.http_ingest import INGEST_PATH, HttpIngestAdapter
from pathlayer.events import EventKind, ToolEvent


def _post(port: int, body: dict[str, Any], path: str = INGEST_PATH) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post_raw(port: int, raw: bytes) -> int:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{INGEST_PATH}",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _echo_translator(payload: Any) -> list[ToolEvent]:
    return [ToolEvent(tool=str(payload["tool"]), args=payload.get("args", {}))]


def test_translators_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="translators"):
        HttpIngestAdapter(lambda _e: None, translators={})


@pytest.mark.parametrize("bad_host", ["0.0.0.0", "192.168.1.10", ""])
def test_non_loopback_host_is_rejected(bad_host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        HttpIngestAdapter(lambda _e: None, translators={"x": _echo_translator}, host=bad_host)


@pytest.mark.parametrize("good_host", ["127.0.0.1", "localhost"])
def test_loopback_hosts_are_accepted(good_host: str) -> None:
    adapter = HttpIngestAdapter(
        lambda _e: None, translators={"x": _echo_translator}, host=good_host, port=0
    )
    adapter.attach()
    adapter.detach()


def test_port_before_attach_raises() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"x": _echo_translator})
    with pytest.raises(RuntimeError, match="not attached"):
        _ = adapter.port


def test_double_attach_raises() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"x": _echo_translator})
    adapter.attach()
    try:
        with pytest.raises(RuntimeError, match="already attached"):
            adapter.attach()
    finally:
        adapter.detach()


def test_detach_without_attach_is_a_noop() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"x": _echo_translator})
    adapter.detach()  # must not raise


def test_valid_post_is_translated_and_emitted() -> None:
    events: list[ToolEvent] = []
    adapter = HttpIngestAdapter(events.append, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        status, body = _post(adapter.port, {"source": "demo", "payload": {"tool": "read_file"}})
        assert status == 200
        assert body == {"ok": True}
    finally:
        adapter.detach()

    assert len(events) == 1
    assert events[0].tool == "read_file"
    assert events[0].kind is EventKind.TOOL_CALL
    assert events[0].ts == 0


def test_unknown_path_is_404() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        status, body = _post(adapter.port, {"source": "demo", "payload": {}}, path="/other")
        assert status == 404
        assert "error" in body
    finally:
        adapter.detach()


def test_malformed_json_is_400() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        status = _post_raw(adapter.port, b"{not json")
        assert status == 400
    finally:
        adapter.detach()


def test_missing_source_key_is_400() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        status, body = _post(adapter.port, {"payload": {}})
        assert status == 400
        assert "source" in body["error"]
    finally:
        adapter.detach()


def test_missing_payload_key_is_400_not_translated_as_none() -> None:
    # A missing 'payload' must be a clean 400, not silently become None and
    # let a translator's own exception on `None` surface as an opaque 500.
    events: list[ToolEvent] = []
    adapter = HttpIngestAdapter(events.append, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        status, body = _post(adapter.port, {"source": "demo"})
        assert status == 400
        assert "payload" in body["error"]
    finally:
        adapter.detach()
    assert events == []


def test_non_object_body_is_400() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        status = _post_raw(adapter.port, b"[1, 2, 3]")
        assert status == 400
    finally:
        adapter.detach()


def test_non_string_source_is_400() -> None:
    events: list[ToolEvent] = []
    adapter = HttpIngestAdapter(events.append, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        status, body = _post(adapter.port, {"source": ["demo"], "payload": {}})
        assert status == 400
        assert "source must be a string" in body["error"]
    finally:
        adapter.detach()
    assert events == []


def test_invalid_content_length_header_is_400() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{adapter.port}{INGEST_PATH}",
            data=b'{"source": "demo", "payload": {}}',
            headers={"Content-Type": "application/json", "Content-Length": "not-a-number"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 400
    finally:
        adapter.detach()


def test_oversized_body_is_400() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        huge_payload = {"tool": "x", "args": {"blob": "a" * 2_000_000}}
        status, body = _post(adapter.port, {"source": "demo", "payload": huge_payload})
        assert status == 400
        assert "exceeds" in body["error"]
    finally:
        adapter.detach()


def test_unknown_source_is_400() -> None:
    events: list[ToolEvent] = []
    adapter = HttpIngestAdapter(events.append, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        status, body = _post(adapter.port, {"source": "nope", "payload": {}})
        assert status == 400
        assert "nope" in body["error"]
    finally:
        adapter.detach()
    assert events == []


def test_translator_exception_is_hidden_logged_and_does_not_crash_server(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "SEC3_DO_NOT_LEAK_7f42"

    def _boom(_payload: Any) -> list[ToolEvent]:
        raise KeyError(f"translator failed with {secret}")

    events: list[ToolEvent] = []
    adapter = HttpIngestAdapter(
        events.append,
        translators={"broken": _boom, "healthy": _echo_translator},
    )
    adapter.attach()
    try:
        with caplog.at_level(logging.ERROR, logger="pathlayer.adapters.http_ingest"):
            status, body = _post(adapter.port, {"source": "broken", "payload": {}})

        assert status == 500
        assert body == {"error": "internal error (see server log)"}
        serialized_body = json.dumps(body)
        assert "KeyError" not in serialized_body
        assert secret not in serialized_body

        error_records = [
            record
            for record in caplog.records
            if record.name == "pathlayer.adapters.http_ingest"
            and record.levelno == logging.ERROR
            and record.exc_info is not None
        ]
        assert len(error_records) == 1
        exc_info = error_records[0].exc_info
        assert exc_info is not None
        traceback_text = logging.Formatter().formatException(exc_info)
        assert "Traceback (most recent call last)" in traceback_text
        assert secret in traceback_text

        # The server must still be alive and serving other requests after a
        # translator crash -- a hostile/buggy event must never take it down.
        status, body = _post(
            adapter.port,
            {"source": "healthy", "payload": {"tool": "ok"}},
        )
        assert status == 200
        assert body == {"ok": True}
    finally:
        adapter.detach()
    assert [event.tool for event in events] == ["ok"]


def test_translator_returning_multiple_events_are_all_emitted_in_order() -> None:
    def _fan_out(payload: Any) -> list[ToolEvent]:
        return [ToolEvent(tool=f"tool-{i}") for i in range(payload["n"])]

    events: list[ToolEvent] = []
    adapter = HttpIngestAdapter(events.append, translators={"demo": _fan_out})
    adapter.attach()
    try:
        _post(adapter.port, {"source": "demo", "payload": {"n": 3}})
    finally:
        adapter.detach()

    assert [e.tool for e in events] == ["tool-0", "tool-1", "tool-2"]
    assert [e.ts for e in events] == [0, 1, 2]


def test_translator_returning_no_events_is_fine() -> None:
    def _ignore(_payload: Any) -> list[ToolEvent]:
        return []

    events: list[ToolEvent] = []
    adapter = HttpIngestAdapter(events.append, translators={"demo": _ignore})
    adapter.attach()
    try:
        status, _ = _post(adapter.port, {"source": "demo", "payload": {}})
        assert status == 200
    finally:
        adapter.detach()
    assert events == []


def test_reattach_resets_ts_counter() -> None:
    events: list[ToolEvent] = []
    adapter = HttpIngestAdapter(events.append, translators={"demo": _echo_translator})
    adapter.attach()
    _post(adapter.port, {"source": "demo", "payload": {"tool": "a"}})
    adapter.detach()

    events.clear()
    adapter.attach()
    try:
        _post(adapter.port, {"source": "demo", "payload": {"tool": "b"}})
    finally:
        adapter.detach()
    assert events[0].ts == 0


def test_binds_to_loopback_only_by_default() -> None:
    adapter = HttpIngestAdapter(lambda _e: None, translators={"demo": _echo_translator})
    adapter.attach()
    try:
        assert adapter._server is not None
        assert adapter._server.server_address[0] == "127.0.0.1"
    finally:
        adapter.detach()


def test_concurrent_posts_lose_no_events_and_ts_stays_contiguous() -> None:
    events: list[ToolEvent] = []
    bookkeeping = threading.Lock()

    def sink(event: ToolEvent) -> None:
        with bookkeeping:
            events.append(event)

    adapter = HttpIngestAdapter(sink, translators={"demo": _echo_translator})
    adapter.attach()

    n = 32
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            status, _ = _post(adapter.port, {"source": "demo", "payload": {"tool": f"t{i}"}})
            assert status == 200
        except BaseException as exc:  # surfaced via `errors`, not lost in a thread
            with bookkeeping:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    adapter.detach()

    assert errors == []
    assert len(events) == n
    assert sorted(e.ts for e in events) == list(range(n))
