import numpy as np
import pytest

from pythermondt.visualization.adapter import DataContainerAdapter


def test_list_children_root(viewer_container):
    """Root listing returns only direct child nodes."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.list_children("/")

    assert payload["path"] == "/"
    child_paths = [child["path"] for child in payload["children"]]
    assert child_paths == ["/Data", "/GroundTruth", "/MetaData"]


def test_get_node_details_dataset(viewer_container):
    """Dataset details contain shape, dtype, and unit metadata."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.get_node_details("/Data/Tdata")

    assert payload["node_type"] == "dataset"
    assert payload["shape"] == [2, 3, 4]
    assert payload["dtype"] == "torch.float32"
    assert payload["numel"] == 24
    assert payload["unit_label"] == "arbitrary"
    assert payload["attributes"]["Description"] == "Thermal data"


def test_get_plot_meta_frames_mode(viewer_container):
    """3D datasets expose frame mode metadata."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.get_plot_meta("/Data/Tdata")

    assert payload["render_mode"] == "frames"
    assert payload["default_mode"] == "heatmap"
    assert payload["available_modes"] == ["matrix", "line", "heatmap"]
    assert payload["shape"] == [2, 3, 4]
    assert payload["frame_axis_options"] == [0, 1, 2]
    assert payload["default_frame_axis"] == 2
    assert payload["default_frame_count"] == 4
    assert payload["default_frame_shape"] == [2, 3]


def test_get_plot_meta_line_mode(viewer_container):
    """1D datasets expose line mode metadata."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.get_plot_meta("/MetaData/DomainValues")

    assert payload["render_mode"] == "line"
    assert payload["default_mode"] == "line"
    assert payload["available_modes"] == ["matrix", "line"]
    assert payload["shape"] == [4]


def test_get_plot_meta_high_dim_line_only(viewer_container):
    """Higher-dimensional datasets expose line-only mode."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.get_plot_meta("/Data/Tensor4D")

    assert payload["render_mode"] == "line"
    assert payload["default_mode"] == "line"
    assert payload["available_modes"] == ["line"]
    assert "reason" in payload


def test_get_vector_binary_returns_float32_payload(viewer_container):
    """Vector endpoint returns binary float32 payload with metadata."""
    adapter = DataContainerAdapter(viewer_container)

    payload, metadata = adapter.get_vector_binary("/MetaData/DomainValues", start=0, count=2, stride=2)
    values = np.frombuffer(payload, dtype=np.float32)

    assert metadata["dtype"] == "float32"
    assert metadata["shape"] == [2]
    assert metadata["count"] == 2
    assert np.allclose(values, np.array([0.0, 0.2], dtype=np.float32))


def test_get_vector_binary_flattens_multidimensional_dataset(viewer_container):
    """Vector endpoint flattens multidimensional datasets for generic line mode."""
    adapter = DataContainerAdapter(viewer_container)

    payload, metadata = adapter.get_vector_binary("/Data/Image2D", start=0, count=4, stride=1)
    values = np.frombuffer(payload, dtype=np.float32)

    assert metadata["shape"] == [4]
    assert np.array_equal(values, np.array([0, 1, 2, 3], dtype=np.float32))


def test_get_frame_binary_returns_expected_slice(viewer_container):
    """Frame endpoint returns selected 2D slice from 3D tensor."""
    adapter = DataContainerAdapter(viewer_container)

    payload, metadata = adapter.get_frame_binary("/Data/Tdata", frame_axis=2, frame_index=1)
    values = np.frombuffer(payload, dtype=np.float32)

    assert metadata["dtype"] == "float32"
    assert metadata["shape"] == [2, 3]
    assert metadata["frame_axis"] == 2
    assert metadata["frame_index"] == 1
    assert np.array_equal(values, np.array([1, 5, 9, 13, 17, 21], dtype=np.float32))


def test_get_frame_binary_rejects_invalid_axis(viewer_container):
    """Frame endpoint validates frame axis bounds."""
    adapter = DataContainerAdapter(viewer_container)

    with pytest.raises(ValueError, match="frame_axis must be in range"):
        adapter.get_frame_binary("/Data/Tdata", frame_axis=3, frame_index=0)


def test_get_matrix_data_for_1d_dataset(viewer_container):
    """Matrix mode represents 1D data as a single-row table."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.get_matrix_data("/MetaData/DomainValues", row_start=0, row_count=1, col_start=0, col_count=3)

    assert payload["shape"] == [1, 4]
    assert payload["row_start"] == 0
    assert payload["row_end"] == 1
    assert payload["col_end"] == 3
    assert np.allclose(np.array(payload["values"], dtype=np.float32), np.array([[0.0, 0.1, 0.2]], dtype=np.float32))


def test_get_matrix_data_for_3d_dataset_uses_frame_slice(viewer_container):
    """Matrix mode for 3D data uses selected 2D frame slice."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.get_matrix_data("/Data/Tdata", frame_axis=2, frame_index=2, row_count=2, col_count=2)

    assert payload["shape"] == [2, 3]
    assert payload["frame_axis"] == 2
    assert payload["frame_index"] == 2
    assert payload["values"] == [[2.0, 6.0], [14.0, 18.0]]


def test_list_children_rejects_dataset(viewer_container):
    """Only root/group nodes can be used as tree parents."""
    adapter = DataContainerAdapter(viewer_container)

    with pytest.raises(TypeError, match="has no children"):
        adapter.list_children("/Data/Tdata")
