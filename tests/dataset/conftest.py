import pytest

from pythermondt import LocalReader, ThermoDataset
from pythermondt.transforms import ThermoTransform


@pytest.fixture
def local_reader_three_files():
    """Fixture for a LocalReader with 3 files."""
    return LocalReader(pattern="./tests/assets/integration/simulation/", num_files=3)


@pytest.fixture
def sample_dataset_single_file(localreader_with_file: LocalReader):
    """Create a sample ThermoDataset for indexing tests."""
    return ThermoDataset(localreader_with_file)


@pytest.fixture
def sample_dataset_three_files(local_reader_three_files: LocalReader):
    """Create a sample ThermoDataset with multiple files."""
    return ThermoDataset(local_reader_three_files)


@pytest.fixture
def sample_dataset_simple_transform(local_reader: LocalReader, single_transform: type[ThermoTransform]):
    """Create a sample ThermoDataset with a simple transform."""
    transform = single_transform("base_level")
    return ThermoDataset(local_reader, transform=transform)
