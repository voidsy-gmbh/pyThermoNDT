import pytest

from pythermondt import LocalReader, ThermoDataset


@pytest.fixture
def local_reader():
    """Fixture for a LocalReader with no files."""
    return LocalReader(pattern="./tests/assets/integration/simulation/", num_files=3)


@pytest.fixture
def sample_dataset_single_file(localreader_with_file: LocalReader):
    """Create a sample ThermoDataset for indexing tests."""
    return ThermoDataset(localreader_with_file)


@pytest.fixture
def sample_dataset_three_files(local_reader: LocalReader):
    """Create a sample ThermoDataset with multiple files."""
    return ThermoDataset(local_reader)
