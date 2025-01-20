import pytest
from torch import Tensor
from pythermondt.data import DataContainer
from pythermondt.data.datacontainer.utils import split_path

def test_path_exists(base_container: DataContainer):
    # Test existing paths
    assert base_container._path_exists("/TestGroup") == True
    assert base_container._path_exists("/TestGroup/NestedGroup") == True
    assert base_container._path_exists("/TestGroup/TestDataset") == True
    assert base_container._path_exists("/TestGroup/NestedGroup/TestDataset2") == True

    # Test non-existing paths
    assert base_container._path_exists("/NonExistentGroup") == False
    assert base_container._path_exists("/TestGroup/NonExistentDataset") == False
    assert base_container._path_exists("/TestGroup/NestedGroup/NonExistentDataset") == False

def test_is_datanode(base_container: DataContainer):
    # Test existing DataNode paths
    assert base_container._is_datanode("/TestGroup/TestDataset") == True
    assert base_container._is_datanode("/TestGroup/NestedGroup/TestDataset2") == True

    # Test non-existing DataNode paths
    assert base_container._is_datanode("/NonExistentGroup") == False
    assert base_container._is_datanode("/TestGroup/NonExistentDataset") == False
    assert base_container._is_datanode("/TestGroup/NestedGroup/NonExistentDataset") == False

    # Test existing paths that are not DataNodes
    assert base_container._is_datanode("/TestGroup") == False
    assert base_container._is_datanode("/TestGroup/NestedGroup") == False

def test_is_groupnode(base_container: DataContainer):
    # Test existing GroupNode paths
    assert base_container._is_groupnode("/TestGroup") == True
    assert base_container._is_groupnode("/TestGroup/NestedGroup") == True

    # Test non-existing GroupNode paths
    assert base_container._is_groupnode("/NonExistentGroup") == False
    assert base_container._is_groupnode("/TestGroup/NonExistentGroup") == False

    # Test existing paths that are not GroupNodes
    assert base_container._is_groupnode("/TestGroup/TestDataset") == False
    assert base_container._is_groupnode("/TestGroup/NestedGroup/TestDataset2") == False

def test_is_rootnode(base_container: DataContainer):
    # Test the root path
    print(split_path("/"))
    print(base_container.nodes.keys())
    assert base_container._is_rootnode("/") == True

    # Test non-root paths
    assert base_container._is_rootnode("/TestGroup") == False
    assert base_container._is_rootnode("/TestGroup/NestedGroup") == False
    assert base_container._is_rootnode("/TestGroup/TestDataset") == False
    assert base_container._is_rootnode("/TestGroup/NestedGroup/TestDataset2") == False

    # Test non-existing paths
    assert base_container._is_rootnode("/NonExistentGroup") == False
    assert base_container._is_rootnode("/TestGroup/NonExistentDataset") == False
    assert base_container._is_rootnode("/TestGroup/NestedGroup/NonExistentDataset") == False

def test_parent_exists(base_container: DataContainer):
    # Test existing parent paths
    assert base_container._parent_exists("/TestGroup") == True
    assert base_container._parent_exists("/TestGroup/NestedGroup") == True
    assert base_container._parent_exists("/TestGroup/TestDataset") == True
    assert base_container._parent_exists("/TestGroup/NestedGroup/TestDataset2") == True

    # Test non-existing parent paths
    assert base_container._parent_exists("/NonExistentGroup/Child") == False
    assert base_container._parent_exists("/TestGroup/NonExistentGroup/Child") == False

    # Test root path
    assert base_container._parent_exists("/") == False

    # Test paths with non-GroupNode or non-RootNode parents
    assert base_container._parent_exists("/TestGroup/TestDataset/Child") == False
    assert base_container._parent_exists("/TestGroup/NestedGroup/TestDataset2/Child") == False

# Only run the tests in this file if it is run directly
if __name__ == '__main__':
    pytest.main(["-v", __file__])