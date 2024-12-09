import pytest
from pythermondt.data import DataContainer
from pythermondt.data.datacontainer.node import GroupNode
from pythermondt.data.datacontainer.utils import validate_path

@pytest.fixture
def group_container(empty_container: DataContainer):
    """Container fixture for testing group operations"""
    empty_container.add_group("/", "testgroup") # root level group
    empty_container.add_group("/testgroup", "nestedgroup") # nested group
    return empty_container

# Test adding a group to the container
@pytest.mark.parametrize("path, name", [
    ("/", "group0"),  # add to root
    ("/testgroup", "group1"),  # add to existing group
    ("/testgroup", "group2"),  # add to same level
    ("/testgroup/nestedgroup", "group3"),  # add to nested group
])
def test_add_group(group_container: DataContainer, path: str, name: str):
    # Add a group
    group_container.add_group(path, name)
    
    # Generate the full path for verification
    key = validate_path(path, name)
    
    # Verify group was added correctly
    assert key in group_container.nodes.keys()
    assert isinstance(group_container.nodes[key], GroupNode)
    assert group_container.nodes[key].name == name

# Test adding a group that already exists
@pytest.mark.parametrize("path, name", [
    ("/testgroup", "existinggroup"),  # duplicate in group
    ("/", "duplicategroup"),  # duplicate in root
])
def test_add_group_existing(group_container: DataContainer, path: str, name: str):
    # Add initial group
    group_container.add_group(path, name)
    
    # Try to add the same group again
    with pytest.raises(KeyError):
        group_container.add_group(path, name)

# Test adding groups with invalid paths
@pytest.mark.parametrize("path, name, expected_error", [
    ("/nonexistent", "group1", KeyError),  # parent doesn't exist
    ("/testgroup/nonexistent/group1", "group2", KeyError),  # nested parent doesn't exist
])
def test_add_group_invalid_path(group_container: DataContainer, path: str, name: str, expected_error: type[Exception]):
    with pytest.raises(expected_error):
        group_container.add_group(path, name)

# Test getting all groups
def test_get_all_groups(empty_container: DataContainer):
    # Create a hierarchy of groups
    groups = [
        ("/", "group1"),
        ("/", "group2"),
        ("/group1", "nested1"),
        ("/group2", "nested2"),
    ]
    
    # Add all groups
    for path, name in groups:
        empty_container.add_group(path, name)
    
    # Get all groups and verify
    all_groups = empty_container.get_all_groups()
    expected_names = {"group1", "group2", "nested1", "nested2"}
    assert set(all_groups) == expected_names

# Test removing groups
@pytest.mark.parametrize("setup_groups, remove_path", [
    # (list of (path, name) to setup, path to remove)
    ([("/", "group1")], "/group1"),  # simple root level removal
    ([("/", "parent"), ("/parent", "child")], "/parent"),  # remove group with nested group
])
def test_remove_group(empty_container: DataContainer, setup_groups: list[tuple[str, str]], remove_path: str):
    # Setup initial groups
    for path, name in setup_groups:
        empty_container.add_group(path, name)
    
    # Remove the specified group
    empty_container.remove_group(remove_path)
    
    # Verify group was removed
    assert remove_path not in empty_container.nodes.keys()
    
    # If we removed a parent group, verify all children were removed
    for key in empty_container.nodes.keys():
        assert not key.startswith(remove_path + "/")

# Test removing groups with invalid paths
@pytest.mark.parametrize("path, expected_error", [
    ("/nonexistent", KeyError),  # group doesn't exist
    ("/testgroup/nonexistent", KeyError),  # nested group doesn't exist
])
def test_remove_group_invalid(group_container: DataContainer, path: str, expected_error: type[Exception]):
    with pytest.raises(expected_error):
        group_container.remove_group(path)

# Test removing a group that contains datasets
def test_remove_group_with_datasets(empty_container: DataContainer, sample_tensor):
    # Setup group with dataset
    empty_container.add_group("/", "group1")
    empty_container.add_dataset("/group1", "dataset1", sample_tensor)
    
    # Remove group
    empty_container.remove_group("/group1")
    
    # Verify group and dataset were removed
    assert "/group1" not in empty_container.nodes.keys()
    assert "/group1/dataset1" not in empty_container.nodes.keys()

# Only run the tests in this file if it is run directly
if __name__ == '__main__':
    pytest.main(["-v", __file__])