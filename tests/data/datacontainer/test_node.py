import pytest
import torch
from pythermondt.data.datacontainer.node import NodeType, RootNode, GroupNode, DataNode

# Root Node tests
def test_root_node(root_node: RootNode):
    """Test RootNode initialization."""
    assert root_node.name == "root"
    assert root_node.type == NodeType.ROOT

# Group Node tests
def test_group_node(group_node: GroupNode):
    """Test GroupNode initialization."""
    assert group_node.name == "test_group"
    assert group_node.type == NodeType.GROUP

def test_group_node_attributes(group_node: GroupNode):
    """Test adding and getting attributes of GroupNode."""
    # Test GroupNode attributes
    group_node.add_attribute("test_attr", "test_value")
    assert group_node.get_attribute("test_attr") == "test_value"

def test_group_node_update_attribute(group_node: GroupNode):
    """Test updating attributes of GroupNode."""
    # Test GroupNode update attribute
    group_node.add_attribute("test_attr", "test_value")
    group_node.update_attribute("test_attr", "new_value")
    assert group_node.get_attribute("test_attr") == "new_value"

    # Try to update an attribute that does not exist
    with pytest.raises(KeyError):
        group_node.update_attribute("non_existent_attr", "new_value")

    # Try to update an attribute with a value that is not a string
    with pytest.raises(TypeError):
        group_node.update_attribute("test_attr", 123)

def test_group_node_remove_attribute(group_node: GroupNode):
    """Test removing attributes of GroupNode."""
    # Test GroupNode remove attribute
    group_node.add_attribute("test_attr", "test_value")
    group_node.remove_attribute("test_attr")
    with pytest.raises(KeyError):
        group_node.get_attribute("test_attr")

# Data Node tests
def test_data_node(data_node: DataNode):
    """Test DataNode initialization."""
    assert data_node.name == "test_data"
    assert data_node.type == NodeType.DATASET
    assert torch.equal(data_node.data, torch.eye(3))

def test_data_node_data_update(data_node: DataNode):
    """Test updating data of DataNode."""
    # Test DataNode update data
    new_data = torch.eye(3) * 2
    data_node.data = new_data
    assert torch.equal(data_node.data, new_data)

def test_data_node_attributes(data_node: DataNode):
    """Test adding and getting attributes of DataNode."""
    # Test DataNode attributes
    data_node.add_attribute("shape", list(data_node.data.shape))
    assert data_node.get_attribute("shape") == [3, 3]

def test_data_node_update_attribute(data_node: DataNode):
    """Test updating attributes of DataNode."""
    # Test DataNode update attribute
    data_node.add_attribute("shape", list(data_node.data.shape))
    data_node.update_attribute("shape", [5, 5])
    assert data_node.get_attribute("shape") == [5, 5]

    # Try to update an attribute that does not exist
    with pytest.raises(KeyError):
        data_node.update_attribute("non_existent_attr", "new_value")

    # Try to update an attribute with a value that is not a list
    with pytest.raises(TypeError):
        data_node.update_attribute("shape", "new_value")

# Node Type Enum tests
def test_node_type_enum():
    """Test NodeType enum."""
    assert NodeType.ROOT.value == "root"
    assert NodeType.GROUP.value == "group"
    assert NodeType.DATASET.value == "dataset"

# Only run the tests in this file if it is run directly
if __name__ == '__main__':
    pytest.main(["-v", __file__])

