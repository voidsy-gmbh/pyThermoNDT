import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from pythermondt.visualization import view
from pythermondt.visualization.adapter import DataContainerAdapter
from pythermondt.visualization.server import ViewerServer


def _read_json(url: str) -> tuple[int, dict]:
    with urlopen(url, timeout=3) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload


def test_viewer_health_endpoint(viewer_container):
    """Viewer starts and serves health endpoint."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        status, payload = _read_json(f"{viewer.url}/api/v1/health")
    finally:
        viewer.stop()

    assert status == 200
    assert payload == {"status": "ok"}


def test_viewer_tree_and_node_endpoints(viewer_container):
    """Viewer serves tree and node metadata payloads."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        tree_status, tree_payload = _read_json(f"{viewer.url}/api/v1/tree?path=/")
        node_status, node_payload = _read_json(f"{viewer.url}/api/v1/node?path=/Data/Tdata")
    finally:
        viewer.stop()

    assert tree_status == 200
    assert [child["path"] for child in tree_payload["children"]] == ["/Data", "/GroundTruth", "/MetaData"]

    assert node_status == 200
    assert node_payload["node_type"] == "dataset"
    assert node_payload["shape"] == [2, 3, 4]


def test_viewer_preview_endpoint(viewer_container):
    """Viewer serves dataset preview slices."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        status, payload = _read_json(f"{viewer.url}/api/v1/preview?path=/Data/Tdata&offset=4&limit=4")
    finally:
        viewer.stop()

    assert status == 200
    assert payload["offset"] == 4
    assert payload["returned"] == 4
    assert payload["values"] == [4.0, 5.0, 6.0, 7.0]


def test_viewer_invalid_path_returns_404(viewer_container):
    """Invalid node paths return a JSON error response."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        with pytest.raises(HTTPError) as error:
            urlopen(f"{viewer.url}/api/v1/node?path=/does/not/exist", timeout=3)  # noqa: S310
        response_body = json.loads(error.value.read().decode("utf-8"))
    finally:
        viewer.stop()

    assert error.value.code == 404
    assert "does not exist" in response_body["error"]


def test_viewer_serves_index_html(viewer_container):
    """Viewer serves the static frontend."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        with urlopen(f"{viewer.url}/", timeout=3) as response:  # noqa: S310
            status = response.status
            html = response.read().decode("utf-8")
    finally:
        viewer.stop()

    assert status == 200
    assert "PyThermoNDT BaseViewer" in html


def test_wait_uses_polling_join(viewer_container):
    """Server wait uses timed joins for interruptible blocking."""

    class _ThreadStub:
        def __init__(self):
            self.join_calls = []
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            self._alive = False

    server = ViewerServer(DataContainerAdapter(viewer_container))
    thread = _ThreadStub()
    server._thread = thread  # type: ignore[assignment]

    server.wait()

    assert thread.join_calls == [0.2]


def test_wait_rejects_non_positive_interval(viewer_container):
    """Server wait validates polling interval arguments."""
    server = ViewerServer(DataContainerAdapter(viewer_container))

    class _ThreadStub:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    server._thread = _ThreadStub()  # type: ignore[assignment]

    with pytest.raises(ValueError, match="poll_interval must be > 0"):
        server.wait(poll_interval=0)
