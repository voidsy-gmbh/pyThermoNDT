from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from pythermondt.data import DataContainer, IndexedThermoDataset, ThermoDataset, random_split
from pythermondt.readers import LocalReader

from ..utils import containers_equal
from .utils import IntegrationTestCase, discover_test_cases

# Get all test cases
INTEGRATION_DIR = Path(__file__).parent / ".." / "assets" / "integration"
TEST_CASES = discover_test_cases(INTEGRATION_DIR)
TEST_IDS = [case.id for case in TEST_CASES]


@pytest.mark.parametrize("test_case", TEST_CASES, ids=TEST_IDS)
def test_local_reader_integration(test_case: IntegrationTestCase):
    """Test LocalReader integration using source files and expected outputs."""
    # Create readers
    source_reader = LocalReader(test_case.source_path)
    expected_reader = LocalReader(test_case.expected_path)

    # Compare all containers in the reader
    for i, (source_container, expected_container) in enumerate(zip(source_reader, expected_reader, strict=True)):
        # Compare containers
        assert containers_equal(expected_container, source_container), (
            f"Test case '{test_case.id}': {source_reader.files[i]} and {expected_reader.files[i]} are not equal"
        )


@pytest.mark.parametrize("test_case", TEST_CASES, ids=TEST_IDS)
def test_thermodataset_integration(test_case: IntegrationTestCase):
    """Test ThermoDataset integration using source files and expected outputs."""
    # Create readers
    source_reader = LocalReader(test_case.source_path)
    expected_reader = LocalReader(test_case.expected_path)

    # Create ThermoDataset objects
    source_dataset = ThermoDataset(source_reader)
    expected_dataset = ThermoDataset(expected_reader)

    # Compare all containers in the dataset
    for i, (source_container, expected_container) in enumerate(zip(source_dataset, expected_dataset, strict=True)):
        # Compare containers
        assert containers_equal(expected_container, source_container), (
            f"Test case '{test_case.id}': {source_dataset.files[i]} and {expected_dataset.files[i]} are not equal"
        )


@pytest.mark.parametrize("test_case", TEST_CASES, ids=TEST_IDS)
def test_indexed_thermodataset_integration(test_case: IntegrationTestCase):
    """Test IndexedThermoDataset integration."""
    # Create readers
    source_reader = LocalReader(test_case.source_path)
    expected_reader = LocalReader(test_case.expected_path)

    # Create ThermoDataset objects
    source_dataset = ThermoDataset(source_reader)
    expected_dataset = ThermoDataset(expected_reader)

    # Create IndexedThermoDataset objects
    s_indexed_dataset = IndexedThermoDataset(source_dataset, indices=range(int(len(source_dataset) / 2)))
    e_indexed_dataset = IndexedThermoDataset(expected_dataset, indices=range(int(len(expected_dataset) / 2)))

    # Compare all containers in the indexed dataset
    for i, (source_container, expected_container) in enumerate(zip(s_indexed_dataset, e_indexed_dataset, strict=True)):
        # Compare containers
        assert containers_equal(expected_container, source_container), (
            f"Test case '{test_case.id}': {s_indexed_dataset.files[i]} and {e_indexed_dataset.files[i]} are not equal"
        )


@pytest.mark.parametrize("test_case", TEST_CASES, ids=TEST_IDS)
@pytest.mark.parametrize("split_ratio", [[0.5, 0.5], [0.8, 0.2], [0.7, 0.3]])
def test_random_split_integration(test_case: IntegrationTestCase, split_ratio: list[float]):
    """Test random_split function."""
    # Create readers
    source_reader = LocalReader(test_case.source_path)
    expected_reader = LocalReader(test_case.expected_path)

    # Create ThermoDataset objects
    source_dataset = ThermoDataset(source_reader)
    expected_dataset = ThermoDataset(expected_reader)

    # Split the datasets randomly at halve (set generator for reproducibility)
    seed = 42
    source_train, source_test = random_split(source_dataset, split_ratio, generator=torch.manual_seed(seed))
    expected_train, expected_test = random_split(expected_dataset, split_ratio, generator=torch.manual_seed(seed))

    # Compare the train datasets
    for i, (source_container, expected_container) in enumerate(zip(source_train, expected_train, strict=True)):
        assert containers_equal(expected_container, source_container), (
            f"Test case '{test_case.id}': Train {i} not equal"
        )

    # Compare the test datasets
    for i, (source_container, expected_container) in enumerate(zip(source_test, expected_test, strict=True)):
        assert containers_equal(expected_container, source_container), f"Test case '{test_case.id}': Test {i} not equal"


@pytest.mark.parametrize("test_case", TEST_CASES, ids=TEST_IDS)
def test_pytorch_dataloader_integration(test_case: IntegrationTestCase):
    """Test PyTorch DataLoader integration."""

    # Custom collate function to stack all datasets in the DataContainer into a single tensor
    def collate_fn(batch: list[DataContainer]):
        # Get the dataset that appear in all container in the batch
        paths = sorted({path for container in batch for path in container.get_all_dataset_paths()})

        # Stack all datasets in the batch
        return [torch.stack([container.get_dataset(path) for container in batch]) for path in paths]

    # Create readers
    source_reader = LocalReader(test_case.source_path)
    expected_reader = LocalReader(test_case.expected_path)

    # Create ThermoDataset objects
    source_dataset = ThermoDataset(source_reader)
    expected_dataset = ThermoDataset(expected_reader)

    # Create DataLoader objects
    source_dataloader = DataLoader(source_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    expected_dataloader = DataLoader(expected_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Compare all batches
    for i, (source_batch, expected_batch) in enumerate(zip(source_dataloader, expected_dataloader, strict=True)):
        # Assert the batches are the same length
        assert len(source_batch) == len(expected_batch), f"Batch {i}: Different number of tensors"

        # Compare each tensor in the batch
        for j, (source_tensor, expected_tensor) in enumerate(zip(source_batch, expected_batch, strict=True)):
            assert torch.equal(source_tensor, expected_tensor), (
                f"Test case '{test_case.id}': Batch {i}, Tensor {j} not equal"
            )
