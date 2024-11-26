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
def empty_data_container():
    """Fixture for DataContainer."""
    return DataContainer()