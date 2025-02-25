import pytest

from pythermondt.data import DataContainer
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


# Only run the tests in this file if it is run directly
if __name__ == "__main__":
    pytest.main(["-v", __file__])
