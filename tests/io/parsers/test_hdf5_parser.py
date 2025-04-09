import io

import pytest
import torch

from pythermondt.data import DataContainer
from pythermondt.io.parsers import HDF5Parser


def test_hdf5_parser_empty_bytes():
    """Test that HDF5Parser raises appropriate error with empty BytesIO."""
    empty_bytes = io.BytesIO()
    with pytest.raises(ValueError, match="The given BytesIO object is empty"):
        HDF5Parser.parse(empty_bytes)


def test_hdf5_parser_invalid_bytes():
    """Test that HDF5Parser raises appropriate error with invalid HDF5 data."""
    invalid_bytes = io.BytesIO(b"not an hdf5 file")
    with pytest.raises(ValueError, match="The given BytesIO object does not contain a valid HDF5 file"):
        HDF5Parser.parse(invalid_bytes)


def test_hdf5_parser_valid_container():
    """Test that HDF5Parser correctly parses a valid HDF5 DataContainer."""
    # Create a sample container and serialize it
    original = DataContainer()
    original.add_group("/", "TestGroup")
    original.add_dataset("/TestGroup", "TestData", torch.tensor([1, 2, 3]))

    serialized = original.serialize_to_hdf5()

    # Parse the serialized data
    parsed = HDF5Parser.parse(serialized)

    # Check that the parsed container has the expected structure
    assert parsed._path_exists("/TestGroup")
    assert parsed._path_exists("/TestGroup/TestData")

    # Check data content matches
    assert parsed == original
