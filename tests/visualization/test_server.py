import json
from http.client import HTTPMessage
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from pythermondt.visualization import view
from pythermondt.visualization.adapter import DataContainerAdapter
from pythermondt.visualization.server import ViewerServer


def _read_json(url: str) -> tuple[int, dict]:
    with urlopen(url, timeout=3) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload


def _read_binary(url: str) -> tuple[int, HTTPMessage, bytes]:
    with urlopen(url, timeout=3) as response:  # noqa: S310
        payload = response.read()
        return response.status, response.headers, payload


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


def test_viewer_plot_meta_endpoint(viewer_container):
    """Viewer serves plot metadata for datasets."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        status, payload = _read_json(f"{viewer.url}/api/v1/plot/meta?path=/Data/Tdata")
    finally:
        viewer.stop()

    assert status == 200
    assert payload["render_mode"] == "frames"
    assert payload["default_mode"] == "heatmap"
    assert payload["available_modes"] == ["matrix", "line", "heatmap"]
    assert payload["default_frame_axis"] == 2
    assert payload["default_frame_count"] == 4


def test_viewer_vector_binary_endpoint(viewer_container):
    """Viewer serves vector payloads as binary float32."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        status, headers, payload = _read_binary(
            f"{viewer.url}/api/v1/plot/vector.bin?path=/MetaData/DomainValues&start=0&count=2&stride=2"
        )
    finally:
        viewer.stop()

    assert status == 200
    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["X-PTNDT-Dtype"] == "float32"
    assert headers["X-PTNDT-Shape"] == "2"
    assert headers["X-PTNDT-Ndim"] == "1"

    values = np.frombuffer(payload, dtype=np.float32)
    assert np.allclose(values, np.array([0.0, 0.2], dtype=np.float32))


def test_viewer_frame_binary_endpoint(viewer_container):
    """Viewer serves frame payloads as binary float32."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        status, headers, payload = _read_binary(
            f"{viewer.url}/api/v1/plot/frame.bin?path=/Data/Tdata&frame_axis=2&frame_index=1"
        )
    finally:
        viewer.stop()

    assert status == 200
    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["X-PTNDT-Dtype"] == "float32"
    assert headers["X-PTNDT-Shape"] == "2,3"
    assert headers["X-PTNDT-Frame-Axis"] == "2"
    assert headers["X-PTNDT-Frame-Index"] == "1"

    values = np.frombuffer(payload, dtype=np.float32)
    assert np.array_equal(values, np.array([1, 5, 9, 13, 17, 21], dtype=np.float32))


def test_viewer_matrix_endpoint(viewer_container):
    """Viewer serves matrix table payloads for dataset inspection."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        status, payload = _read_json(
            f"{viewer.url}/api/v1/plot/matrix?path=/Data/Tdata&frame_axis=2&frame_index=2&row_count=2&col_count=2"
        )
    finally:
        viewer.stop()

    assert status == 200
    assert payload["shape"] == [2, 3]
    assert payload["row_start"] == 0
    assert payload["row_end"] == 2
    assert payload["col_end"] == 2
    assert payload["values"] == [[2.0, 6.0], [14.0, 18.0]]


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


def test_viewer_serves_static_assets(viewer_container):
    """Viewer serves the static frontend and vendored scripts."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        with urlopen(f"{viewer.url}/", timeout=3) as index_response:  # noqa: S310
            index_status = index_response.status
            html = index_response.read().decode("utf-8")

        with urlopen(f"{viewer.url}/vendor/plotly-3.4.0.min.js", timeout=3) as script_response:  # noqa: S310
            script_status = script_response.status
            script_content_type = script_response.headers["Content-Type"]
            script_head = script_response.read(128).decode("utf-8", errors="ignore")
    finally:
        viewer.stop()

    assert index_status == 200
    assert "PyThermoNDT BaseViewer" in html
    assert script_status == 200
    assert "javascript" in script_content_type
    assert "plotly" in script_head.lower()


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
