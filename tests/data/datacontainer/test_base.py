import pytest
import torch

from pythermondt.data import DataContainer
from pythermondt.data.datacontainer.node import DataNode, GroupNode, RootNode
from pythermondt.data.datacontainer.utils import split_path


def test_path_exists(filled_container: DataContainer):
    # Test existing paths
    assert filled_container._path_exists("/TestGroup") is True
    assert filled_container._path_exists("/TestDataset") is True
    assert filled_container._path_exists("/TestGroup/NestedGroup") is True
    assert filled_container._path_exists("/TestGroup/TestDataset1") is True
    assert filled_container._path_exists("/TestGroup/NestedGroup/TestDataset2") is True

    # Test non-existing paths
    assert filled_container._path_exists("/NonExistentGroup") is False
    assert filled_container._path_exists("/TestGroup/NonExistentDataset") is False
    assert filled_container._path_exists("/TestGroup/NestedGroup/NonExistentDataset") is False


def test_is_datanode(filled_container: DataContainer):
    # Test existing DataNode paths
    assert filled_container._path_exists("/TestDataset") is True
    assert filled_container._is_datanode("/TestGroup/TestDataset1") is True
    assert filled_container._is_datanode("/TestGroup/NestedGroup/TestDataset2") is True

    # Test non-existing DataNode paths
    assert filled_container._is_datanode("/NonExistentGroup") is False
    assert filled_container._is_datanode("/TestGroup/NonExistentDataset") is False
    assert filled_container._is_datanode("/TestGroup/NestedGroup/NonExistentDataset") is False

    # Test existing paths that are not DataNodes
    assert filled_container._is_datanode("/TestGroup") is False
    assert filled_container._is_datanode("/TestGroup/NestedGroup") is False


def test_is_groupnode(filled_container: DataContainer):
    # Test existing GroupNode paths
    assert filled_container._is_groupnode("/TestGroup") is True
    assert filled_container._is_groupnode("/TestGroup/NestedGroup") is True

    # Test non-existing GroupNode paths
    assert filled_container._is_groupnode("/NonExistentGroup") is False
    assert filled_container._is_groupnode("/TestGroup/NonExistentGroup") is False

    # Test existing paths that are not GroupNodes
    assert filled_container._is_groupnode("/TestGroup/TestDataset") is False
    assert filled_container._is_groupnode("/TestGroup/NestedGroup/TestDataset2") is False


def test_is_rootnode(filled_container: DataContainer):
    # Test the root path
    print(split_path("/"))
    print(filled_container.nodes.keys())
    assert filled_container._is_rootnode("/") is True

    # Test non-root paths
    assert filled_container._is_rootnode("/TestGroup") is False
    assert filled_container._is_rootnode("/TestGroup/NestedGroup") is False
    assert filled_container._is_rootnode("/TestGroup/TestDataset") is False
    assert filled_container._is_rootnode("/TestGroup/NestedGroup/TestDataset2") is False

    # Test non-existing paths
    assert filled_container._is_rootnode("/NonExistentGroup") is False
    assert filled_container._is_rootnode("/TestGroup/NonExistentDataset") is False
    assert filled_container._is_rootnode("/TestGroup/NestedGroup/NonExistentDataset") is False


def test_parent_exists(filled_container: DataContainer):
    # Test existing parent paths
    assert filled_container._parent_exists("/TestGroup") is True
    assert filled_container._parent_exists("/TestGroup/NestedGroup") is True
    assert filled_container._parent_exists("/TestGroup/TestDataset") is True
    assert filled_container._parent_exists("/TestGroup/NestedGroup/TestDataset2") is True

    # Test non-existing parent paths
    assert filled_container._parent_exists("/NonExistentGroup/Child") is False
    assert filled_container._parent_exists("/TestGroup/NonExistentGroup/Child") is False

    # Test root path
    assert filled_container._parent_exists("/") is False

    # Test paths with non-GroupNode or non-RootNode parents
    assert filled_container._parent_exists("/TestGroup/TestDataset/Child") is False
    assert filled_container._parent_exists("/TestGroup/NestedGroup/TestDataset2/Child") is False


def test_memory_bytes_empty_container(empty_container: DataContainer):
    """Test memory calculation for empty container."""
    memory = empty_container.memory_bytes()
    assert isinstance(memory, int)
    assert memory > 0


def test_memory_bytes_filled_container(filled_container: DataContainer):
    """Test memory calculation for filled container."""
    memory = filled_container.memory_bytes()
    assert isinstance(memory, int)
    assert memory > 0


def test_memory_increases_with_content(empty_container: DataContainer, filled_container: DataContainer):
    """Test that memory increases with container content."""
    empty_memory = empty_container.memory_bytes()
    filled_memory = filled_container.memory_bytes()
    assert filled_memory > empty_memory


def test_memory_increases_with_large_tensors(empty_container: DataContainer):
    """Test memory increases with large tensor data."""
    small_tensor = torch.randn(5, 5)
    large_tensor = torch.randn(100, 100)

    empty_container.add_dataset("/", "small_data", small_tensor)
    memory_small = empty_container.memory_bytes()

    empty_container.add_dataset("/", "large_data", large_tensor)
    memory_large = empty_container.memory_bytes()

    assert memory_large > memory_small


def test_print_memory_usage(filled_container: DataContainer, capsys):
    """Test print_memory_usage output."""
    filled_container.print_memory_usage()
    captured = capsys.readouterr()
    assert "DataContainer Memory Usage:" in captured.out
    assert "B" in captured.out or "KB" in captured.out or "MB" in captured.out


def test_node_accessor_get_node_non_existent(empty_container: DataContainer):
    """Test that getting a non-existent node raises a KeyError."""
    with pytest.raises(KeyError, match="Node at path '/nonexistent' does not exist."):
        empty_container.nodes["/nonexistent"]


def test_node_accessor_get_node_wrong_type(filled_container: DataContainer):
    """Test that getting a node with the wrong type raises a TypeError."""
    with pytest.raises(TypeError, match="Node at path '/TestGroup' is not of type: DataNode."):
        filled_container.nodes("/TestGroup", DataNode)


def test_node_accessor_set_node_overwrite_fails(filled_container: DataContainer):
    """Test that overwriting an existing node raises a KeyError."""
    with pytest.raises(KeyError, match="Node at path '/TestGroup' already exists."):
        filled_container.nodes["/TestGroup"] = GroupNode("NewGroup")


def test_node_accessor_set_root_node_wrong_path(empty_container: DataContainer):
    """Test that setting a RootNode at a wrong path raises a ValueError."""
    with pytest.raises(ValueError, match="RootNode must be placed at the root path '/'"):
        # Need to delete the existing root to attempt to add a new one
        del empty_container.nodes["/"]
        empty_container.nodes["/wrong"] = RootNode()


def test_node_accessor_set_root_node_already_exists(empty_container: DataContainer):
    """Test that adding a second RootNode raises a ValueError."""
    with pytest.raises(ValueError, match="RootNode already exists in the DataContainer. RootNode must be unique"):
        empty_container.nodes["/"] = RootNode()


def test_node_accessor_set_node_with_no_root_node(empty_container: DataContainer):
    """Test that adding a node when no RootNode exists raises a KeyError."""
    del empty_container.nodes["/"]  # Ensure no root node exists
    with pytest.raises(KeyError, match="RootNode does not exist in this container."):
        empty_container.nodes["/new_node"] = GroupNode("NewNode")


def test_node_accessor_set_node_no_parent(empty_container: DataContainer):
    """Test that setting a node with a non-existent parent raises a KeyError."""
    with pytest.raises(KeyError, match="Parent node at path '/nonexistent' does not exist."):
        empty_container.nodes["/nonexistent/new"] = GroupNode("new")


def test_node_accessor_set_node_parent_is_datanode(filled_container: DataContainer):
    """Test that setting a node under a DataNode raises a TypeError."""
    with pytest.raises(TypeError, match="Parent node at path '/TestDataset' must be a RootNode or GroupNode."):
        filled_container.nodes["/TestDataset/new"] = GroupNode("new")


def test_node_accessor_delete_node_non_existent(empty_container: DataContainer):
    """Test that deleting a non-existent node raises a KeyError."""
    with pytest.raises(KeyError, match="Node at path '/nonexistent' does not exist."):
        del empty_container.nodes["/nonexistent"]


def test_node_accessor_delete_node_with_children(filled_container: DataContainer):
    """Test that deleting a group also deletes its children."""
    assert filled_container._path_exists("/TestGroup/NestedGroup/TestDataset2")
    del filled_container.nodes["/TestGroup"]
    assert not filled_container._path_exists("/TestGroup")
    assert not filled_container._path_exists("/TestGroup/NestedGroup")
    assert not filled_container._path_exists("/TestGroup/NestedGroup/TestDataset2")


# Only run the tests in this file if it is run directly
if __name__ == "__main__":
    pytest.main(["-v", __file__])
