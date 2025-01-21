import pytest
from pythermondt.data import DataContainer

@pytest.mark.parametrize("container_fixture", [
    "empty_container",
    "filled_container", 
    "complex_container"
])
def test_serialize_deserialize(container_fixture: str, request: pytest.FixtureRequest):
    """Test serialization and deserialization of DataContainer.
    
    Parameters:
        container_fixture: Name of the fixture to test
        request: Pytest fixture request object to get the fixture
    
    Tests:
        1. Serialization to HDF5 bytes
        2. Deserialization back to DataContainer
        3. Equality comparison of original and deserialized containers
    """
    # Get the container from the fixture
    original_container = request.getfixturevalue(container_fixture) #type: DataContainer
    
    # Serialize the DataContainer
    hdf5_bytes = original_container.serialize_to_hdf5()
    
    # Deserialize into a new container
    deserialized_container = DataContainer(hdf5_bytes)
    
    # Check if the deserialized container is equal to the original container
    assert deserialized_container == original_container