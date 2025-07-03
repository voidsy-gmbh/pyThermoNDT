import pytest

from pythermondt import LocalReader, ThermoDataset
from pythermondt.data.datacontainer.core import DataContainer
from pythermondt.transforms import ThermoTransform


@pytest.fixture
def local_reader():
    """Fixture for a LocalReader with 3 files."""
    return LocalReader(pattern="./tests/assets/integration/simulation/", num_files=3)


@pytest.fixture
def sample_dataset_single_file(localreader_with_file: LocalReader):
    """Create a sample ThermoDataset for indexing tests."""
    return ThermoDataset(localreader_with_file)


@pytest.fixture
def sample_dataset_three_files(local_reader: LocalReader):
    """Create a sample ThermoDataset with multiple files."""
    return ThermoDataset(local_reader)


@pytest.fixture
def single_transform():
    """Create a simple ThermoTransform that adds an attribute."""

    class SimpleTransform(ThermoTransform):
        def forward(self, container: DataContainer) -> DataContainer:
            if "transformed" in container.get_all_attributes("/MetaData"):
                i = container.get_attribute("/MetaData", "transformed")
                i = i if isinstance(i, int) else 0
                container.update_attribute("/MetaData", "transformed", i + 1)
            else:
                container.add_attribute("/MetaData", "transformed", 1)
            return container

    return SimpleTransform()


@pytest.fixture
def sample_dataset_simple_transform(local_reader: LocalReader, single_transform: ThermoTransform):
    """Create a sample ThermoDataset with a simple transform."""
    return ThermoDataset(local_reader, transform=single_transform)
