"""Run the training data pipeline end to end once per parser folder."""

import pytest
import torch
from torch.utils.data import DataLoader

from pythermondt.data import container_diff
from pythermondt.dataset import IndexedThermoDataset, ThermoDataset, container_collate, random_split
from tests.support.integration_data import IntegrationData, collect_integration_data, folder_readers
from tests.utils import format_container_diff

pytestmark = pytest.mark.cloud

# A placeholder keeps the test visible when cloud tests are skipped or discovery failed.
FOLDERS = collect_integration_data().folders or ("unavailable",)

SPLIT_FRACTIONS = [0.5, 0.5]
SPLIT_SEED = 42
BATCH_FIELDS = ("/Data/Tdata", "/MetaData/DomainValues")


def _assert_same_containers(
    stage: str, source: ThermoDataset | IndexedThermoDataset, expected: ThermoDataset | IndexedThermoDataset
):
    for i, (source_container, expected_container) in enumerate(zip(source, expected, strict=True)):
        differences = list(container_diff(expected_container, source_container))
        assert not differences, (
            f"{stage}, item {i}:\ncontainer1 = {expected.files[i]} (expected)\n"
            f"container2 = {source.files[i]} (parsed)\n{format_container_diff(differences)}"
        )


@pytest.mark.parametrize("folder", FOLDERS)
def test_training_pipeline(integration_data: IntegrationData, folder: str):
    """Dataset, random split and DataLoader must yield the same data for sources and expected outputs."""
    assert integration_data.config is not None
    source_reader, expected_reader = folder_readers(integration_data.config, integration_data.pairs, folder)
    source_dataset, expected_dataset = ThermoDataset(source_reader), ThermoDataset(expected_reader)
    _assert_same_containers(f"{folder} pipeline, dataset", source_dataset, expected_dataset)

    # Equal seeds select the same indices on both sides, so subsets stay paired.
    source_splits = random_split(source_dataset, SPLIT_FRACTIONS, generator=torch.Generator().manual_seed(SPLIT_SEED))
    expected_splits = random_split(
        expected_dataset, SPLIT_FRACTIONS, generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    for split_name, source_split, expected_split in zip(("train", "test"), source_splits, expected_splits, strict=True):
        _assert_same_containers(f"{folder} pipeline, split {split_name}", source_split, expected_split)

        # Batch size 1: files of one folder may differ in shape and cannot be stacked.
        loaders = [
            DataLoader(split, batch_size=1, collate_fn=container_collate(*BATCH_FIELDS))
            for split in (source_split, expected_split)
        ]
        for i, (source_batch, expected_batch) in enumerate(zip(*loaders, strict=True)):
            for field, source_tensor, expected_tensor in zip(BATCH_FIELDS, source_batch, expected_batch, strict=True):
                assert torch.equal(source_tensor, expected_tensor), (
                    f"{folder} pipeline, dataloader (split {split_name}), batch {i}:\n"
                    f"{source_split.files[i]} vs {expected_split.files[i]}\n  - {field} differs"
                )
