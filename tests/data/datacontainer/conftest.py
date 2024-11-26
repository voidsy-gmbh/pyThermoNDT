import pytest
import torch
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
def data_node():
    """Fixture for DataNode."""
    data = torch.eye(3)
    return DataNode("test_data", data)