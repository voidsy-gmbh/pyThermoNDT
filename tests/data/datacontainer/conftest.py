import pytest
from torch import Tensor
from pythermondt.data.datacontainer.node import RootNode, GroupNode, DataNode, NodeType

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