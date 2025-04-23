import io

import pytest

from pythermondt.data import DataContainer
from pythermondt.utils import IOPathWrapper


@pytest.mark.parametrize("container_fixture", ["empty_container", "filled_container", "complex_container"])
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
    original_container = request.getfixturevalue(container_fixture)  # type: DataContainer

    # Serialize the DataContainer
    hdf5_bytes = original_container.serialize_to_hdf5()

    # Check if the serialized data is a bytes object and is not empty
    assert isinstance(hdf5_bytes, IOPathWrapper)
    assert hdf5_bytes.file_obj.getbuffer().nbytes > 0

    # Deserialize into a new container
    deserialized_container = DataContainer(hdf5_bytes)

    # Check if the deserialized container is equal to the original container
    assert deserialized_container == original_container


@pytest.mark.parametrize("container_fixture", ["empty_container", "filled_container", "complex_container"])
def test_serialize_file_operations(container_fixture: str, request: pytest.FixtureRequest, tmp_path):
    """Test save_to_hdf5 and load_from_hdf5 file operations."""
    # Create temporary file path
    file_path = tmp_path / "test.hdf5"

    # Get the container from the fixture
    original_container = request.getfixturevalue(container_fixture)  # type: DataContainer

    # Save container to file
    original_container.save_to_hdf5(str(file_path))

    # Load into new container
    loaded_container = DataContainer()
    loaded_container.load_from_hdf5(str(file_path))

    # Verify equality
    assert loaded_container == original_container


def test_serialize_empty_bytes():
    """Test that deserializing empty bytes raises appropriate error."""
    empty_bytes = IOPathWrapper(io.BytesIO())
    with pytest.raises(ValueError, match="The given BytesIO object is empty."):
        DataContainer(empty_bytes)


def test_serialize_invalid_hdf5():
    """Test that deserializing invalid HDF5 data raises appropriate error."""
    invalid_bytes = IOPathWrapper(io.BytesIO(b"not an hdf5 file"))
    with pytest.raises(ValueError, match="The given BytesIO object does not contain a valid HDF5 file."):
        DataContainer(invalid_bytes)
