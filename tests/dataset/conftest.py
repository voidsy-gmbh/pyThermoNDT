import pytest

from pythermondt import LocalReader, ThermoDataset


@pytest.fixture
def sample_dataset_single_file(localreader_with_file: LocalReader):
    """Create a sample ThermoDataset for indexing tests."""
    return ThermoDataset(localreader_with_file)
