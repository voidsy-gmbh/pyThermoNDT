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


def test_get_node_details_group(viewer_container):
    """Group details expose attributes without dataset-only fields."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.get_node_details("/MetaData")

    assert payload["node_type"] == "group"
    assert "shape" not in payload


def test_get_dataset_preview_slice(viewer_container):
    """Preview endpoint returns bounded flattened slices."""
    adapter = DataContainerAdapter(viewer_container)

    payload = adapter.get_dataset_preview("/Data/Tdata", offset=2, limit=5)

    assert payload["path"] == "/Data/Tdata"
    assert payload["offset"] == 2
    assert payload["limit"] == 5
    assert payload["returned"] == 5
    assert payload["total"] == 24
    assert payload["values"] == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_get_dataset_preview_rejects_group(viewer_container):
    """Preview requests must target datasets."""
    adapter = DataContainerAdapter(viewer_container)

    with pytest.raises(TypeError, match="is not a dataset"):
        adapter.get_dataset_preview("/Data")


def test_list_children_rejects_dataset(viewer_container):
    """Only root/group nodes can be used as tree parents."""
    adapter = DataContainerAdapter(viewer_container)

    with pytest.raises(TypeError, match="has no children"):
        adapter.list_children("/Data/Tdata")
