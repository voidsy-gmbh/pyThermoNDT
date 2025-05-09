from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from pythermondt.data import DataContainer, ThermoDataset
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
