from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlparse

from .adapter import DataContainerAdapter
from .contracts import API_PREFIX, DEFAULT_PREVIEW_LIMIT, JsonObject


class ViewerServer:
    """HTTP server that exposes DataContainer visualization APIs and static assets."""

    def __init__(self, adapter: DataContainerAdapter, host: str = "127.0.0.1", port: int = 0):
        self._adapter = adapter
        self._host = host
        self._port = port
        self._url = ""
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Get the viewer URL.

        Returns:
            str: URL of the running server.

        Raises:
            RuntimeError: If the server has not been started.
        """
        if not self._url:
            raise RuntimeError("Viewer server is not running.")
        return self._url

    def start(self, open_browser: bool = True) -> str:
        """Start the server.

        Args:
            open_browser (bool): If True, open the viewer URL in default browser.

        Returns:
            str: URL of the running server.
        """
        if self._server is not None:
            return self.url

        handler = self._build_handler()
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        self._url = f"http://{self._host}:{self._server.server_port}"
        if open_browser:
            webbrowser.open(self._url)

        return self._url

    def stop(self) -> None:
        """Stop the server if it is running."""
        if self._server is None:
            return

        self._server.shutdown()
        self._server.server_close()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self._server = None
        self._thread = None
        self._url = ""

    def wait(self, poll_interval: float = 0.2) -> None:
        """Block until server thread exits.

        Uses periodic timed joins to remain interruptible via KeyboardInterrupt
        across platforms.

        Raises:
            RuntimeError: If the server has not been started.
            ValueError: If poll_interval is not positive.
        """
        if self._thread is None:
            raise RuntimeError("Viewer server is not running.")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")

        while True:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return
            thread.join(timeout=poll_interval)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        adapter = self._adapter
        static_dir = files("pythermondt.visualization").joinpath("static")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed_url = urlparse(self.path)
                route = parsed_url.path
                query = parse_qs(parsed_url.query)

                try:
                    if route in ("/", "/index.html"):
                        self._send_static_file(
                            static_dir.joinpath("index.html").read_bytes(), "text/html; charset=utf-8"
                        )
                        return

                    if route == "/app.js":
                        self._send_static_file(
                            static_dir.joinpath("app.js").read_bytes(), "application/javascript; charset=utf-8"
                        )
                        return

                    if route == "/styles.css":
                        self._send_static_file(
                            static_dir.joinpath("styles.css").read_bytes(), "text/css; charset=utf-8"
                        )
                        return

                    if route == f"{API_PREFIX}/health":
                        self._send_json(HTTPStatus.OK, {"status": "ok"})
                        return

                    if route == f"{API_PREFIX}/tree":
                        path = self._get_query_value(query, "path", "/")
                        self._send_json(HTTPStatus.OK, adapter.list_children(path))
                        return

                    if route == f"{API_PREFIX}/node":
                        path = self._get_query_value(query, "path", "/")
                        self._send_json(HTTPStatus.OK, adapter.get_node_details(path))
                        return

                    if route == f"{API_PREFIX}/preview":
                        path = self._get_query_value(query, "path", "/")
                        offset = self._get_query_int(query, "offset", 0)
                        limit = self._get_query_int(query, "limit", DEFAULT_PREVIEW_LIMIT)
                        self._send_json(HTTPStatus.OK, adapter.get_dataset_preview(path, offset=offset, limit=limit))
                        return

                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Route '{route}' not found."})

                except KeyError as error:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": self._exception_message(error)})
                except (TypeError, ValueError) as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                except Exception as error:  # pylint: disable=broad-except
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Unexpected server error: {error}"})

            def _send_json(self, status: HTTPStatus, payload: JsonObject) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_static_file(self, content: bytes, content_type: str) -> None:
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            @staticmethod
            def _get_query_value(query: dict[str, list[str]], key: str, default: str) -> str:
                values = query.get(key)
                if not values:
                    return default
                return values[0]

            @staticmethod
            def _get_query_int(query: dict[str, list[str]], key: str, default: int) -> int:
                value = Handler._get_query_value(query, key, str(default))
                return int(value)

            @staticmethod
            def _exception_message(error: Exception) -> str:
                if error.args:
                    return str(error.args[0])
                return str(error)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        return Handler
