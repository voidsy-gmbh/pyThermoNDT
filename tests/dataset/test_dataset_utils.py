"""Tests for dataset random_split, container_collate, and derive."""

import pytest
import torch

from pythermondt import ThermoDataset
from pythermondt.dataset.utils import container_collate, derive, random_split

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
    """Test that a missing dataset path raises KeyError with field name."""
    batch = [make_container(("/Data", "Tdata", torch.randn(2, 2)))]
    fn = container_collate("/Data/NonExistent")
    with pytest.raises(KeyError, match="Field '/Data/NonExistent'"):
        fn(batch)


def test_container_collate_incompatible_shapes():
    """Test that incompatible tensor shapes raise RuntimeError with field name."""
    batch = [
        make_container(("/Data", "Tdata", torch.randn(4, 4))),
        make_container(("/Data", "Tdata", torch.randn(3, 5))),
    ]
    fn = container_collate("/Data/Tdata")
    with pytest.raises(RuntimeError, match="Cannot stack tensors for field '/Data/Tdata'"):
        fn(batch)


def test_container_collate_invalid_field_type():
    """Test that a non-str, non-DeriveField argument raises TypeError."""
    with pytest.raises(TypeError, match="Must be str or DeriveField"):
        container_collate(123)


def test_derive_tensor_output():
    """Test that derive with a tensor output is stacked correctly."""
    t1, t2 = torch.randn(4, 4, 10), torch.randn(4, 4, 10)
    batch = [
        make_container(("/Data", "Tdata", t1)),
        make_container(("/Data", "Tdata", t2)),
    ]
    fn = container_collate(derive("permuted", lambda c: c.get_dataset("/Data/Tdata").permute(2, 0, 1)))
    (result,) = fn(batch)
    assert result.shape == (2, 10, 4, 4)
    assert torch.equal(result[0], t1.permute(2, 0, 1))
    assert torch.equal(result[1], t2.permute(2, 0, 1))


def test_derive_bool_output():
    """Test that derive with a bool output is stacked into a BoolTensor."""
    t = torch.randn(4, 4, 10)
    batch = [
        make_container(("/Data", "Tdata", t), ("/GroundTruth", "DefectMask", torch.ones(4, 4))),
        make_container(("/Data", "Tdata", t)),
    ]
    fn = container_collate(derive("has_mask", lambda c: "/GroundTruth/DefectMask" in c.nodes))
    (result,) = fn(batch)
    assert result.dtype == torch.bool
    assert result.tolist() == [True, False]


def test_derive_scalar_output():
    """Test that derive with a scalar output is stacked correctly."""
    batch = [
        make_container(("/Data", "Tdata", torch.randn(4, 4, 8))),
        make_container(("/Data", "Tdata", torch.randn(4, 4, 12))),
    ]
    fn = container_collate(derive("num_frames", lambda c: float(c.get_dataset("/Data/Tdata").shape[-1])))
    (result,) = fn(batch)
    assert result.tolist() == [8.0, 12.0]


def test_derive_mixed_with_str_paths():
    """Test that str paths and derive fields are returned in declaration order."""
    t = torch.randn(4, 4, 10)
    batch = [make_container(("/Data", "Tdata", t))]
    fn = container_collate(
        "/Data/Tdata",
        derive("tdata_cnn", lambda c: c.get_dataset("/Data/Tdata").permute(2, 0, 1)),
        derive("num_frames", lambda c: float(c.get_dataset("/Data/Tdata").shape[-1])),
    )
    result = fn(batch)
    assert len(result) == 3
    assert result[0].shape == (1, 4, 4, 10)
    assert result[1].shape == (1, 10, 4, 4)
    assert result[2].tolist() == [10.0]


def test_derive_error_contains_field_name():
    """Test that non-KeyError from derive fns is wrapped as RuntimeError with field name."""
    batch = [make_container(("/Data", "Tdata", torch.randn(4, 4)))]

    def bad_fn(c):
        raise ValueError("bad input")

    fn = container_collate(derive("bad_field", bad_fn))
    with pytest.raises(RuntimeError, match="Error evaluating field 'bad_field'"):
        fn(batch)


def test_derive_key_error_preserves_type():
    """Test that KeyError from a derive fn keeps the KeyError type with field name."""
    batch = [make_container(("/Data", "Tdata", torch.randn(4, 4)))]

    def bad_fn(c):
        raise KeyError("missing thing")

    fn = container_collate(derive("bad_key", bad_fn))
    with pytest.raises(KeyError, match="Field 'bad_key'"):
        fn(batch)


def test_derive_unsupported_return_type():
    """Test that a derive fn returning a non-stackable type raises RuntimeError with field name."""
    batch = [make_container(("/Data", "Tdata", torch.randn(4, 4)))]
    fn = container_collate(derive("bad_type", lambda c: "not a tensor"))
    with pytest.raises(RuntimeError, match="Cannot stack tensors for field 'bad_type'"):
        fn(batch)


def test_derive_stack_mismatch_contains_field_name():
    """Test that stack shape mismatch in a derive field includes the field name."""
    batch = [
        make_container(("/Data", "Tdata", torch.randn(4, 4))),
        make_container(("/Data", "Tdata", torch.randn(3, 5))),
    ]
    fn = container_collate(derive("custom_field", lambda c: c.get_dataset("/Data/Tdata")))
    with pytest.raises(RuntimeError, match="field 'custom_field'"):
        fn(batch)
