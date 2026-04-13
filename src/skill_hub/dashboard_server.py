"""Tiny localhost HTTP server for the interactive dashboard.

Lazy-started singleton. Binds 127.0.0.1 only. Stdlib ThreadingHTTPServer.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import dashboard_api as _api

_log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"

_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None
_server_url: str | None = None
_store_ref: Any = None

# Dispatch table: path segment -> api function
_DISPATCH = {
    "metrics": _api.api_metrics,
    "verdicts": _api.api_verdicts_list,
    "verdicts_delete": _api.api_verdicts_delete,
    "verdicts_flip": _api.api_verdicts_flip,
    "verdicts_pin": _api.api_verdicts_pin,
    "verdicts_promote_yaml": _api.api_verdicts_promote_yaml,
    "tasks_list": _api.api_tasks_list,
    "tasks_rename": _api.api_tasks_rename,
    "tasks_delete": _api.api_tasks_delete,
    "tasks_merge": _api.api_tasks_merge,
    "tasks_teach": _api.api_tasks_teach,
    "skills_usage": _api.api_skills_usage,
    "skills_for_task": _api.api_skills_for_task,
    "vector_viz": _api.api_vector_viz,
    "vector_classify": _api.api_vector_classify,
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # silence stdout
        return

    def _send_json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        try:
            data = TEMPLATE.read_bytes()
        except OSError:
            self.send_error(500, "template missing")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, path: str, body: dict | None) -> None:
        # path is like "/api/<name>"
        parts = path.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "api":
            self._send_json(404, {"error": "not found"})
            return
        fn = _DISPATCH.get(parts[1])
        if fn is None:
            self._send_json(404, {"error": f"unknown endpoint {parts[1]}"})
            return
        try:
            result = fn(_store_ref, body or {})
        except Exception as e:  # noqa: BLE001
            _log.exception("api error: %s", parts[1])
            self._send_json(500, {"error": str(e)})
            return
        self._send_json(200, result)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            self._send_html()
            return
        if self.path.startswith("/api/"):
            self._dispatch(self.path, None)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body: dict = {}
        if length:
            try:
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "invalid json"})
                return
        self._dispatch(self.path, body)


def start(store: Any, port: int = 8765, host: str = "127.0.0.1") -> str | None:
    """Start the server if not running. Returns URL or None on failure."""
    global _server, _server_url, _store_ref
    with _lock:
        if _server is not None and _server_url is not None:
            _store_ref = store  # refresh store reference
            return _server_url
        try:
            srv = ThreadingHTTPServer((host, port), _Handler)
        except OSError as e:
            _log.warning("dashboard server bind failed on %s:%d: %s", host, port, e)
            return None
        _store_ref = store
        _server = srv
        _server_url = f"http://{host}:{port}/"
        t = threading.Thread(
            target=srv.serve_forever, name="dashboard-server", daemon=True
        )
        t.start()
        _log.info("dashboard server listening at %s", _server_url)
        return _server_url


def stop() -> None:
    global _server, _server_url
    with _lock:
        if _server is not None:
            try:
                _server.shutdown()
                _server.server_close()
            except OSError:
                pass
        _server = None
        _server_url = None


def url() -> str | None:
    return _server_url
