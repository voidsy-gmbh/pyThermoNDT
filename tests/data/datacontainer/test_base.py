import pytest

from pythermondt.data import DataContainer
from pythermondt.data.datacontainer.utils import split_path


def test_path_exists(filled_container: DataContainer):
    # Test existing paths
    assert filled_container._path_exists("/TestGroup") == True
    assert filled_container._path_exists("/TestDataset") == True
    assert filled_container._path_exists("/TestGroup/NestedGroup") == True
    assert filled_container._path_exists("/TestGroup/TestDataset1") == True
    assert filled_container._path_exists("/TestGroup/NestedGroup/TestDataset2") == True

    # Test non-existing paths
    assert filled_container._path_exists("/NonExistentGroup") == False
    assert filled_container._path_exists("/TestGroup/NonExistentDataset") == False
    assert filled_container._path_exists("/TestGroup/NestedGroup/NonExistentDataset") == False


def test_is_datanode(filled_container: DataContainer):
    # Test existing DataNode paths
    assert filled_container._path_exists("/TestDataset") == True
    assert filled_container._is_datanode("/TestGroup/TestDataset1") == True
    assert filled_container._is_datanode("/TestGroup/NestedGroup/TestDataset2") == True

    # Test non-existing DataNode paths
    assert filled_container._is_datanode("/NonExistentGroup") == False
    assert filled_container._is_datanode("/TestGroup/NonExistentDataset") == False
    assert filled_container._is_datanode("/TestGroup/NestedGroup/NonExistentDataset") == False

    # Test existing paths that are not DataNodes
    assert filled_container._is_datanode("/TestGroup") == False
    assert filled_container._is_datanode("/TestGroup/NestedGroup") == False


def test_is_groupnode(filled_container: DataContainer):
    # Test existing GroupNode paths
    assert filled_container._is_groupnode("/TestGroup") == True
    assert filled_container._is_groupnode("/TestGroup/NestedGroup") == True

    # Test non-existing GroupNode paths
    assert filled_container._is_groupnode("/NonExistentGroup") == False
    assert filled_container._is_groupnode("/TestGroup/NonExistentGroup") == False

    # Test existing paths that are not GroupNodes
    assert filled_container._is_groupnode("/TestGroup/TestDataset") == False
    assert filled_container._is_groupnode("/TestGroup/NestedGroup/TestDataset2") == False


def test_is_rootnode(filled_container: DataContainer):
    # Test the root path
    print(split_path("/"))
    print(filled_container.nodes.keys())
    assert filled_container._is_rootnode("/") == True

    # Test non-root paths
    assert filled_container._is_rootnode("/TestGroup") == False
    assert filled_container._is_rootnode("/TestGroup/NestedGroup") == False
    assert filled_container._is_rootnode("/TestGroup/TestDataset") == False
    assert filled_container._is_rootnode("/TestGroup/NestedGroup/TestDataset2") == False

    # Test non-existing paths
    assert filled_container._is_rootnode("/NonExistentGroup") == False
    assert filled_container._is_rootnode("/TestGroup/NonExistentDataset") == False
    assert filled_container._is_rootnode("/TestGroup/NestedGroup/NonExistentDataset") == False


def test_parent_exists(filled_container: DataContainer):
    # Test existing parent paths
    assert filled_container._parent_exists("/TestGroup") == True
    assert filled_container._parent_exists("/TestGroup/NestedGroup") == True
    assert filled_container._parent_exists("/TestGroup/TestDataset") == True
    assert filled_container._parent_exists("/TestGroup/NestedGroup/TestDataset2") == True

    # Test non-existing parent paths
    assert filled_container._parent_exists("/NonExistentGroup/Child") == False
    assert filled_container._parent_exists("/TestGroup/NonExistentGroup/Child") == False

    # Test root path
    assert filled_container._parent_exists("/") == False

    # Test paths with non-GroupNode or non-RootNode parents
    assert filled_container._parent_exists("/TestGroup/TestDataset/Child") == False
    assert filled_container._parent_exists("/TestGroup/NestedGroup/TestDataset2/Child") == False


# Only run the tests in this file if it is run directly
if __name__ == "__main__":
    pytest.main(["-v", __file__])
