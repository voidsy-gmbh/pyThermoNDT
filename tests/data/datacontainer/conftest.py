import pytest
from torch import Tensor
from pythermondt.data import DataContainer
from pythermondt.data.datacontainer.node import RootNode, GroupNode, DataNode

@pytest.fixture
def root_node():
    """Fixture for RootNode."""
    return RootNode()

@pytest.fixture
def group_node():
    """Fixture for GroupNode."""
    return GroupNode("test_group")

@pytest.fixture
def data_node(sample_tensor: Tensor):
    """Fixture for DataNode."""
    return DataNode("test_data", sample_tensor)

@pytest.fixture
def empty_container():
    """Fixture for DataContainer."""
    return DataContainer()

@pytest.fixture
def base_container(empty_container: DataContainer, sample_tensor: Tensor, sample_eye_tensor: Tensor):  
    """Container with basic structure for testing BaseOps"""
    # Add a testgroup
    empty_container.add_group("/", "TestGroup")

    # Add a nested group 
    empty_container.add_group("/TestGroup", "NestedGroup")

    # Add datasets
    empty_container.add_dataset("/TestGroup", "TestDataset", sample_tensor)
    empty_container.add_dataset("/TestGroup/NestedGroup", "TestDataset2", sample_eye_tensor)

    return empty_container