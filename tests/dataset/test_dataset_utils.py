"""Tests for dataset random_split edge cases and container_collate."""

import pytest
import torch

from pythermondt import ThermoDataset
from pythermondt.dataset.utils import container_collate, random_split

from ..utils import make_container


def test_random_split_more_transforms_than_splits(sample_dataset_three_files: ThermoDataset):
    """Test that more transforms than splits raises ValueError."""
    with pytest.raises(ValueError, match=r"Number of transforms.*must match number of splits"):
        random_split(sample_dataset_three_files, [0.5, 0.5], transforms=[None, None, None])


def test_random_split_fewer_transforms_than_splits(sample_dataset_three_files: ThermoDataset):
    """Test that fewer transforms than splits raises ValueError."""
    with pytest.raises(ValueError, match=r"Number of transforms.*must match number of splits"):
        random_split(sample_dataset_three_files, [0.5, 0.5], transforms=[None])


def test_random_split_negative_fraction(sample_dataset_three_files: ThermoDataset):
    """Test that negative fractions raise ValueError."""
    with pytest.raises(ValueError, match="All values in lengths must be greater or equal to 0"):
        random_split(sample_dataset_three_files, [-0.5, 1.5])


def test_random_split_negative_absolute_length(sample_dataset_three_files: ThermoDataset):
    """Test that negative absolute lengths raise ValueError."""
    with pytest.raises(ValueError, match="All values in lengths must be greater or equal to 0"):
        random_split(sample_dataset_three_files, [-1, 4])


def test_random_split_zero_fraction_produces_warning(sample_dataset_three_files: ThermoDataset):
    """Test that a 0-length split emits a UserWarning."""
    with pytest.warns(UserWarning, match="Length of split at index 0 is 0"):
        splits = random_split(sample_dataset_three_files, [0.0, 1.0])
    assert len(splits[0]) == 0
    assert len(splits[1]) == 3


def test_random_split_remainder_distribution(sample_dataset_three_files: ThermoDataset):
    """Test that remainder items are distributed round-robin when fractions don't divide evenly."""
    # 3 files with [0.5, 0.5]: floor(1.5)=1, floor(1.5)=1 => remainder=1, distributed to first split
    splits = random_split(sample_dataset_three_files, [0.5, 0.5])
    assert len(splits[0]) + len(splits[1]) == 3
    assert len(splits[0]) == 2  # gets the remainder
    assert len(splits[1]) == 1


def test_random_split_absolute_lengths_sum_too_small(sample_dataset_three_files: ThermoDataset):
    """Test that absolute lengths summing to less than dataset length raises ValueError."""
    with pytest.raises(ValueError, match="does not match the length of the original dataset"):
        random_split(sample_dataset_three_files, [1, 1])


def test_random_split_absolute_lengths_sum_too_large(sample_dataset_three_files: ThermoDataset):
    """Test that absolute lengths summing to more than dataset length raises ValueError."""
    with pytest.raises(ValueError, match="does not match the length of the original dataset"):
        random_split(sample_dataset_three_files, [2, 2])


def test_container_collate_no_paths():
    """Test that container_collate with no paths raises ValueError."""
    with pytest.raises(ValueError, match="At least one path must be specified"):
        container_collate()


def test_container_collate_returns_callable():
    """Test that container_collate returns a callable."""
    fn = container_collate("/Data/Tdata")
    assert callable(fn)


def test_container_collate_empty_batch():
    """Test that collating an empty batch raises ValueError."""
    fn = container_collate("/Data/Tdata")
    with pytest.raises(ValueError, match="Empty batch"):
        fn([])


def test_container_collate_single_path():
    """Test collating containers with a single dataset path."""
    shape = (4, 4, 10)
    t1, t2 = torch.randn(shape), torch.randn(shape)
    batch = [
        make_container(("/Data", "Tdata", t1)),
        make_container(("/Data", "Tdata", t2)),
    ]
    fn = container_collate("/Data/Tdata")
    (result,) = fn(batch)
    assert result.shape == (2, *shape)
    assert torch.equal(result[0], t1)
    assert torch.equal(result[1], t2)


def test_container_collate_multiple_paths():
    """Test collating containers with multiple dataset paths."""
    tdata = torch.randn(4, 4, 10)
    mask = torch.ones(4, 4)
    batch = [make_container(("/Data", "Tdata", tdata), ("/GroundTruth", "DefectMask", mask))]
    fn = container_collate("/Data/Tdata", "/GroundTruth/DefectMask")
    result = fn(batch)
    assert len(result) == 2
    assert result[0].shape == (1, 4, 4, 10)
    assert result[1].shape == (1, 4, 4)


def test_container_collate_missing_path():
    """Test that a missing dataset path raises KeyError."""
    batch = [make_container(("/Data", "Tdata", torch.randn(2, 2)))]
    fn = container_collate("/Data/NonExistent")
    with pytest.raises(KeyError, match="not found in container"):
        fn(batch)


def test_container_collate_incompatible_shapes():
    """Test that incompatible tensor shapes raise RuntimeError."""
    batch = [
        make_container(("/Data", "Tdata", torch.randn(4, 4))),
        make_container(("/Data", "Tdata", torch.randn(3, 5))),
    ]
    fn = container_collate("/Data/Tdata")
    with pytest.raises(RuntimeError, match="Cannot stack tensors"):
        fn(batch)
