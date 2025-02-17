from typing import Any

import pytest
from torch import Tensor

from pythermondt.data import DataContainer, Units


@pytest.fixture
def attr_container(empty_container: DataContainer, sample_tensor: Tensor):
    """Container fixture for testing attribute operations"""
    # Add a group and dataset for testing
    empty_container.add_group("/", "testgroup")
    empty_container.add_dataset("/", "testdata", sample_tensor)
    return empty_container

@pytest.mark.parametrize("path, key, value", [
    ("/testgroup", "str_attr", "test_string"),  # string attribute on group
    ("/testdata", "int_attr", 42),  # int attribute on dataset
    ("/testgroup", "float_attr", 3.14),  # float attribute on group
    ("/testdata", "list_attr", [1, 2, 3]),  # list attribute on dataset
    ("/testgroup", "tuple_attr", (4, 5, 6)),  # tuple attribute on group
    ("/testdata", "dict_attr", {"key": "value"}),  # dict attribute on dataset
    ("/testdata", "unit_attr", Units.kelvin),  # unit attribute on dataset
])
def test_add_attribute(attr_container: DataContainer, path: str, key: str, value: Any):
    # Add attribute
    attr_container.add_attribute(path, key, value)

    # Verify attribute was added correctly
    retrieved_value = attr_container.get_attribute(path, key)
    assert retrieved_value == value
    assert type(retrieved_value) == type(value)

def test_add_attributes(attr_container: DataContainer):
    # Test adding multiple attributes at once
    attrs = {
        "str_attr": "test_string",
        "int_attr": 42,
        "float_attr": 3.14,
        "list_attr": [1, 2, 3]
    }

    # Add attributes
    attr_container.add_attributes("/testgroup", **attrs)

    # Verify all attributes
    for key, value in attrs.items():
        retrieved = attr_container.get_attribute("/testgroup", key)
        assert retrieved == value
        assert type(retrieved) == type(value)

@pytest.mark.parametrize("path, key, value", [
    ("/testgroup", "existing_attr", "duplicate"),  # duplicate on group
    ("/testdata", "existing_attr", 42),  # duplicate on dataset
])
def test_add_attribute_existing(attr_container: DataContainer, path: str, key: str, value: Any):
    # Add initial attribute
    attr_container.add_attribute(path, key, "initial_value")

    # Try to add duplicate attribute
    with pytest.raises(KeyError):
        attr_container.add_attribute(path, key, value)

@pytest.mark.parametrize("path, key", [
    ("/nonexistent", "attr"),  # non-existent path
    ("/testgroup/nonexistent", "attr"),  # non-existent nested path
])
def test_add_attribute_invalid_path(attr_container: DataContainer, path: str, key: str):
    with pytest.raises(KeyError):
        attr_container.add_attribute(path, key, "value")

def test_get_attributes(attr_container: DataContainer):
    # Add multiple attributes
    attrs = {
        "str_attr": "test_string",
        "int_attr": 42,
        "float_attr": 3.14
    }
    attr_container.add_attributes("/testgroup", **attrs)

    # Test getting multiple attributes
    retrieved_values = attr_container.get_attributes("/testgroup", "str_attr", "int_attr", "float_attr")
    expected_values = tuple(attrs.values())
    assert retrieved_values == expected_values

def test_get_all_attributes(attr_container: DataContainer):
    # Add multiple attributes
    attrs = {
        "str_attr": "test_string",
        "int_attr": 42,
        "float_attr": 3.14
    }
    attr_container.add_attributes("/testgroup", **attrs)

    # Get all attributes
    all_attrs = attr_container.get_all_attributes("/testgroup")
    assert all_attrs == attrs

def test_unit_operations(attr_container: DataContainer):
    # Test adding unit
    attr_container.add_unit("/testdata", Units.kelvin)

    # Test getting unit
    unit = attr_container.get_unit("/testdata")
    assert unit == Units.kelvin

    # Test updating unit
    attr_container.update_unit("/testdata", Units.celsius)
    updated_unit = attr_container.get_unit("/testdata")
    assert updated_unit == Units.celsius

@pytest.mark.parametrize("path, key, initial_value, update_value", [
    ("/testgroup", "str_attr", "initial", "updated"),
    ("/testdata", "int_attr", 42, 100),
    ("/testgroup", "float_attr", 3.14, 2.718),
    ("/testdata", "list_attr", [1, 2], [3, 4]),
    ("/testgroup", "dict_attr", {"old": "value"}, {"new": "value"}),
])
def test_update_attribute(attr_container: DataContainer, path: str, key: str,
                         initial_value: Any, update_value: Any):
    # Add initial attribute
    attr_container.add_attribute(path, key, initial_value)

    # Update attribute
    attr_container.update_attribute(path, key, update_value)

    # Verify update
    retrieved = attr_container.get_attribute(path, key)
    assert retrieved == update_value
    assert type(retrieved) == type(update_value)

def test_update_attributes(attr_container: DataContainer):
    # Add initial attributes
    initial_attrs = {
        "str_attr": "initial",
        "int_attr": 42,
        "float_attr": 3.14
    }
    attr_container.add_attributes("/testgroup", **initial_attrs)

    # Update attributes
    updated_attrs = {
        "str_attr": "updated",
        "int_attr": 100,
        "float_attr": 2.718
    }
    attr_container.update_attributes("/testgroup", **updated_attrs)

    # Verify updates
    for key, value in updated_attrs.items():
        retrieved = attr_container.get_attribute("/testgroup", key)
        assert retrieved == value

@pytest.mark.parametrize("path, key, value", [
    ("/testgroup", "nonexistent", "value"),  # non-existent attribute
    ("/testdata", "nonexistent", 42),  # non-existent attribute
])
def test_update_attribute_nonexistent(attr_container: DataContainer, path: str, key: str, value: Any):
    with pytest.raises(KeyError):
        attr_container.update_attribute(path, key, value)

def test_update_attribute_wrong_type(attr_container: DataContainer):
    # Add string attribute
    attr_container.add_attribute("/testgroup", "str_attr", "string")

    # Try to update with different type
    with pytest.raises(TypeError):
        attr_container.update_attribute("/testgroup", "str_attr", 42)

@pytest.mark.parametrize("path, key", [
    ("/testgroup", "str_attr"),
    ("/testdata", "int_attr"),
])
def test_remove_attribute(attr_container: DataContainer, path: str, key: str):
    # Add attribute
    attr_container.add_attribute(path, key, "value")

    # Remove attribute
    attr_container.remove_attribute(path, key)

    # Verify removal
    with pytest.raises(KeyError):
        attr_container.get_attribute(path, key)

def test_remove_nonexistent_attribute(attr_container: DataContainer):
    with pytest.raises(KeyError):
        attr_container.remove_attribute("/testgroup", "nonexistent")

# Only run the tests in this file if it is run directly
if __name__ == '__main__':
    pytest.main(["-v", __file__])
