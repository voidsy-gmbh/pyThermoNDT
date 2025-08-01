import io
import pickle
from typing import Literal

import pytest
import torch

from pythermondt.data import DataContainer

from ...utils import containers_equal


@pytest.mark.parametrize("compression", ["gzip", "lzf"])
@pytest.mark.parametrize("container_fixture", ["empty_container", "filled_container", "complex_container"])
def test_serialize_deserialize(
    container_fixture: str, request: pytest.FixtureRequest, compression: Literal["gzip", "lzf"]
):
    """Test serialization and deserialization of DataContainer.

    Args:
        container_fixture: Name of the fixture to test
        request: Pytest fixture request object to get the fixture
        compression: Compression method to use for serialization

    Tests:
        1. Serialization to HDF5 bytes
        2. Deserialization back to DataContainer
        3. Equality comparison of original and deserialized containers
    """
    # Get the container from the fixture
    original_container = request.getfixturevalue(container_fixture)  # type: DataContainer

    # Serialize the DataContainer
    hdf5_bytes = original_container.serialize_to_hdf5(compression=compression)

    # Check if the serialized data is a bytes object and is not empty
    assert isinstance(hdf5_bytes, io.BytesIO)
    assert hdf5_bytes.getbuffer().nbytes > 0

    # Deserialize into a new container
    deserialized_container = DataContainer(hdf5_bytes)

    # Check if the deserialized container is equal to the original container
    assert deserialized_container == original_container


@pytest.mark.parametrize("compression", ["gzip", "lzf"])
@pytest.mark.parametrize("container_fixture", ["empty_container", "filled_container", "complex_container"])
def test_serialize_file_operations(
    container_fixture: str, request: pytest.FixtureRequest, compression: Literal["gzip", "lzf"], tmp_path
):
    """Test save_to_hdf5 and load_from_hdf5 file operations."""
    # Create temporary file path
    file_path = tmp_path / "test.hdf5"

    # Get the container from the fixture
    original_container = request.getfixturevalue(container_fixture)  # type: DataContainer

    # Save container to file
    original_container.save_to_hdf5(str(file_path), compression=compression)

    # Load into new container
    loaded_container = DataContainer()
    loaded_container.load_from_hdf5(str(file_path))

    # Verify equality
    assert loaded_container == original_container


def test_serialize_empty_bytes():
    """Test that deserializing empty bytes raises appropriate error."""
    empty_bytes = io.BytesIO()
    with pytest.raises(ValueError, match="The given IOPathWrapper object is empty."):
        DataContainer(empty_bytes)


def test_serialize_invalid_hdf5():
    """Test that deserializing invalid HDF5 data raises appropriate error."""
    invalid_bytes = io.BytesIO(b"not an hdf5 file")
    with pytest.raises(ValueError, match="The given IOPathWrapper object does not contain a valid HDF5 file."):
        DataContainer(invalid_bytes)


@pytest.mark.parametrize("container_fixture", ["empty_container", "filled_container", "complex_container"])
def test_pickle_serialize_deserialize(container_fixture: str, request: pytest.FixtureRequest):
    """Test that DataContainer can be pickled and unpickled.

    Args:
        container_fixture: Name of the fixture to test
        request: Pytest fixture request object to get the fixture

    Tests:
        1. Pickling DataContainer to bytes
        2. Unpickling back to DataContainer
        3. Equality comparison of original and unpickled containers
    """
    # Get the container from the fixture
    original_container = request.getfixturevalue(container_fixture)  # type: DataContainer

    # Pickle the DataContainer
    pickled_bytes = pickle.dumps(original_container)

    # Check if the pickled data is not empty
    assert len(pickled_bytes) > 0

    # Unpickle into a new container
    unpickled_container = pickle.loads(pickled_bytes)

    # Check if the unpickled container is equal to the original container
    assert containers_equal(unpickled_container, original_container), "Unpickled container does not match original"
    assert isinstance(unpickled_container, type(original_container)), "Unpickled container type does not match original"


@pytest.mark.parametrize("container_fixture", ["empty_container", "filled_container", "complex_container"])
@pytest.mark.parametrize("protocol", [0, 2, 4, pickle.HIGHEST_PROTOCOL])
def test_pickle_protocols(container_fixture: str, request: pytest.FixtureRequest, protocol: int):
    """Test pickling with different pickle protocols.

    Args:
        container_fixture: Name of the fixture to test
        request: Pytest fixture request object to get the fixture
        protocol: Pickle protocol version to test

    Tests:
        Pickling and unpickling with various protocol versions
    """
    # Get the container from the fixture
    original_container = request.getfixturevalue(container_fixture)  # type: DataContainer

    # Pickle with specific protocol
    pickled_bytes = pickle.dumps(original_container, protocol=protocol)

    # Unpickle
    unpickled_container = pickle.loads(pickled_bytes)

    # Verify equality
    assert containers_equal(unpickled_container, original_container), "Unpickled container does not match original"
    assert isinstance(unpickled_container, type(original_container)), "Unpickled container type does not match original"


@pytest.mark.parametrize("container_fixture", ["empty_container", "filled_container", "complex_container"])
def test_pickle_file_operations(container_fixture: str, request: pytest.FixtureRequest, tmp_path):
    """Test pickle file save and load operations.

    Args:
        container_fixture: Name of the fixture to test
        request: Pytest fixture request object to get the fixture
        tmp_path: Pytest temporary directory fixture

    Tests:
        1. Saving DataContainer to pickle file
        2. Loading DataContainer from pickle file
        3. Equality comparison
    """
    # Create temporary file path
    file_path = tmp_path / "test_container.pkl"

    # Get the container from the fixture
    original_container = request.getfixturevalue(container_fixture)  # type: DataContainer

    # Save container to pickle file
    with open(file_path, "wb") as f:
        pickle.dump(original_container, f)

    # Load container from pickle file
    with open(file_path, "rb") as f:
        loaded_container = pickle.load(f)

    # Verify equality
    assert containers_equal(loaded_container, original_container), "Unpickled container does not match original"
    assert isinstance(loaded_container, type(original_container)), "Unpickled container type does not match original"


def test_pickle_roundtrip_preserves_state():
    """Test that pickle roundtrip preserves internal state and functionality."""
    # Create a container with some data
    container = DataContainer()
    container.add_group("/", "test_group")
    container.add_dataset("/test_group", "data", data=torch.tensor([1, 2, 3, 4, 5]))
    container.add_attribute("/test_group", "description", "test data")

    # Pickle and unpickle
    pickled_bytes = pickle.dumps(container)
    unpickled_container = pickle.loads(pickled_bytes)  # type: DataContainer

    # Test that functionality is preserved
    assert unpickled_container.get_dataset("/test_group/data").tolist() == [1, 2, 3, 4, 5]
    assert unpickled_container.get_attribute("/test_group", "description") == "test data"
    assert "/test_group" in unpickled_container.nodes.keys()
    assert "/test_group/data" in unpickled_container.nodes.keys()

    # Test that we can still modify the unpickled container
    unpickled_container.add_dataset("/test_group", "new_data", data=torch.tensor([6, 7, 8]))
    assert "/test_group/new_data" in unpickled_container.nodes.keys()
    assert unpickled_container.get_dataset("/test_group/new_data").tolist() == [6, 7, 8]
