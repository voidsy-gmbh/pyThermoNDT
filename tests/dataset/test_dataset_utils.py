"""Tests for dataset random_split edge cases."""

import pytest

from pythermondt import ThermoDataset
from pythermondt.dataset.utils import random_split


def test_random_split_more_transforms_than_splits(sample_dataset_three_files: ThermoDataset):
    """Test that more transforms than splits raises ValueError."""
    with pytest.raises(ValueError, match="Number of transforms.*must match number of splits"):
        random_split(sample_dataset_three_files, [0.5, 0.5], transforms=[None, None, None])


def test_random_split_fewer_transforms_than_splits(sample_dataset_three_files: ThermoDataset):
    """Test that fewer transforms than splits raises ValueError."""
    with pytest.raises(ValueError, match="Number of transforms.*must match number of splits"):
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


def test_random_split_absolute_lengths_sum_too_small(sample_dataset_three_files: ThermoDataset):
    """Test that absolute lengths summing to less than dataset length raises ValueError."""
    with pytest.raises(ValueError, match="does not match the length of the original dataset"):
        random_split(sample_dataset_three_files, [1, 1])


def test_random_split_absolute_lengths_sum_too_large(sample_dataset_three_files: ThermoDataset):
    """Test that absolute lengths summing to more than dataset length raises ValueError."""
    with pytest.raises(ValueError, match="does not match the length of the original dataset"):
        random_split(sample_dataset_three_files, [2, 2])
