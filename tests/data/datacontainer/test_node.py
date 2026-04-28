from sys import getsizeof

import pytest
import torch
from torch import Tensor

from pythermondt.data import units
from pythermondt.data.datacontainer.node import DataNode, GroupNode, NodeType, RootNode


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


def test_node_name_setter_updates_name(group_node: GroupNode):
    """Test BaseNode name setter."""
    group_node.name = "renamed_group"
    assert group_node.name == "renamed_group"


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
def test_data_node(data_node: DataNode, sample_tensor: Tensor):
    """Test DataNode initialization."""
    assert data_node.name == "test_data"
    assert data_node.type == NodeType.DATASET
    assert torch.equal(data_node.data, sample_tensor)


def test_data_node_data_update(data_node: DataNode, sample_eye_tensor: Tensor):
    """Test updating data of DataNode."""
    # Test DataNode update data
    data_node.data = sample_eye_tensor
    assert torch.equal(data_node.data, sample_eye_tensor)


def test_data_node_attributes(data_node: DataNode):
    """Test adding and getting attributes of DataNode."""
    # Test DataNode attributes
    data_node.add_attribute("shape", list(data_node.data.shape))
    assert data_node.get_attribute("shape") == list(data_node.data.shape)


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


def test_attribute_node_clear_attributes(group_node: GroupNode):
    """Test clearing all attributes from node."""
    group_node.add_attributes(a=1, b=2)
    group_node.clear_attributes()
    assert dict(group_node.attributes) == {}


def test_data_node_data_deleter_resets_to_empty_tensor(data_node: DataNode):
    """Test deleting data resets DataNode tensor."""
    del data_node.data
    assert torch.equal(data_node.data, torch.empty(0))


# Memory tests
def test_root_node_memory_bytes(root_node: RootNode):
    """Test memory calculation for RootNode."""
    memory = root_node.memory_bytes()
    assert isinstance(memory, int)
    assert memory > 0


def test_group_node_memory_bytes(group_node: GroupNode):
    """Test memory calculation for GroupNode."""
    memory = group_node.memory_bytes()
    assert isinstance(memory, int)
    assert memory > 0


def test_group_node_memory_with_attributes(group_node: GroupNode):
    """Test memory increases with attributes."""
    memory_empty = group_node.memory_bytes()

    group_node.add_attribute("test_attr", "test_value")
    memory_with_attr = group_node.memory_bytes()

    assert memory_with_attr > memory_empty


@pytest.mark.parametrize(
    ("attributes", "expected_attribute_bytes"),
    [
        ({}, getsizeof({})),
        (
            {"test_attr": "test_value"},
            getsizeof({"test_attr": "test_value"}) + getsizeof("test_value"),
        ),
        (
            {"shape": [5, 6]},
            getsizeof({"shape": [5, 6]}) + getsizeof([5, 6]) + getsizeof(5) + getsizeof(6),
        ),
        (
            {"Unit": units.kelvin},
            getsizeof({"Unit": units.kelvin})
            + getsizeof(units.kelvin)
            + getsizeof(units.kelvin.__dict__)
            + sum(getsizeof(value) for value in vars(units.kelvin).values()),
        ),
        (
            {
                "Source": "Simulation",
                "NoiseLevel": 0.1,
                "LabelIds": [1, 2, 3],
                "Unit": units.arbitrary,
            },
            getsizeof(
                {
                    "Source": "Simulation",
                    "NoiseLevel": 0.1,
                    "LabelIds": [1, 2, 3],
                    "Unit": units.arbitrary,
                }
            )
            + getsizeof("Simulation")
            + getsizeof(0.1)
            + getsizeof([1, 2, 3])
            + getsizeof(1)
            + getsizeof(2)
            + getsizeof(3)
            + getsizeof(units.arbitrary)
            + getsizeof(units.arbitrary.__dict__)
            + getsizeof("arbitrary")
            + getsizeof("a. u."),
        ),
    ],
)
def test_group_node_memory_bytes_matches_expected_values(attributes: dict[str, object], expected_attribute_bytes: int):
    """Test GroupNode memory using explicit `sys.getsizeof` formulas.

    This follows objsize's own testing strategy: compute the expectation from the
    known object graph instead of copying the deep-size traversal into the test.
    """
    group_node = GroupNode("parity_group")
    group_node.add_attributes(**attributes)  # type: ignore

    expected_memory_bytes = getsizeof(group_node) + getsizeof(group_node.name) + getsizeof(group_node.type)
    assert group_node.memory_bytes() == expected_memory_bytes + expected_attribute_bytes


def test_group_node_memory_bytes_matches_expected_values_with_shared_references():
    """Test GroupNode memory counts shared attribute objects only once."""
    shared_list = ["hot", "cold"]
    group_node = GroupNode("shared_group")
    group_node.add_attributes(a=shared_list, b=shared_list, meta={"labels": shared_list})

    expected_attribute_bytes = (
        getsizeof({"a": shared_list, "b": shared_list, "meta": {"labels": shared_list}})
        + getsizeof(shared_list)
        + getsizeof("hot")
        + getsizeof("cold")
        + getsizeof({"labels": shared_list})
    )
    expected_memory_bytes = getsizeof(group_node) + getsizeof(group_node.name) + getsizeof(group_node.type)
    assert group_node.memory_bytes() == expected_memory_bytes + expected_attribute_bytes


def test_data_node_memory_bytes(data_node: DataNode):
    """Test memory calculation for DataNode."""
    memory = data_node.memory_bytes()
    assert isinstance(memory, int)
    assert memory > 0


def test_data_node_memory_scales_with_tensor_size():
    """Test memory scales with tensor size."""
    small_tensor = torch.randn(2, 2)
    large_tensor = torch.randn(10, 10)

    small_node = DataNode("small", small_tensor)
    large_node = DataNode("large", large_tensor)

    small_memory = small_node.memory_bytes()
    large_memory = large_node.memory_bytes()

    assert large_memory > small_memory


def test_data_node_empty_tensor_memory():
    """Test memory calculation with empty tensor."""
    empty_node = DataNode("empty")
    memory = empty_node.memory_bytes()
    assert isinstance(memory, int)
    assert memory > 0


def test_data_node_memory_bytes_matches_expected_values_for_attributes(sample_tensor: Tensor):
    """Test DataNode memory using explicit `sys.getsizeof` formulas."""
    data_node = DataNode("parity_data", sample_tensor)
    data_node.add_attributes(
        Source="Simulation",
        LabelIds=[1, 2, 3],
        Shapes={"Circle": 2, "Line": 1},
        Unit=units.kelvin,
    )

    expected_attribute_bytes = (
        getsizeof(
            {
                "Source": "Simulation",
                "LabelIds": [1, 2, 3],
                "Shapes": {"Circle": 2, "Line": 1},
                "Unit": units.kelvin,
            }
        )
        + getsizeof("Simulation")
        + getsizeof([1, 2, 3])
        + getsizeof(1)
        + getsizeof(2)
        + getsizeof(3)
        + getsizeof({"Circle": 2, "Line": 1})
        + getsizeof(units.kelvin)
        + getsizeof(units.kelvin.__dict__)
        + sum(getsizeof(value) for value in vars(units.kelvin).values())
    )
    expected_memory_bytes = (
        getsizeof(data_node)
        + getsizeof(data_node.name)
        + getsizeof(data_node.type)
        + expected_attribute_bytes
        + getsizeof(data_node.data)
        + data_node.data.element_size() * data_node.data.numel()
    )

    assert data_node.memory_bytes() == expected_memory_bytes


# Node Type Enum tests
def test_node_type_enum():
    """Test NodeType enum."""
    assert NodeType.ROOT.value == "root"
    assert NodeType.GROUP.value == "group"
    assert NodeType.DATASET.value == "dataset"
