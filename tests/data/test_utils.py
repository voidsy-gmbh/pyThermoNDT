import pytest
import torch

from pythermondt.data import DataContainer
from pythermondt.data.utils import container_collate


def test_factory_function_returns_callable():
    """Test that container_collate returns a callable function."""
    collate_fn = container_collate("/TestDataset")
    assert callable(collate_fn)


def test_factory_function_with_no_paths_raises_error():
    """Test that container_collate raises ValueError when no paths are provided."""
    with pytest.raises(ValueError, match="At least one path must be specified"):
        container_collate()


def test_single_path_single_container(empty_container, sample_tensor):
    """Test collating a single container with one dataset path."""
    empty_container.add_dataset("/", "data", sample_tensor)

    collate_fn = container_collate("/data")
    result = collate_fn([empty_container])

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape[0] == 1


def test_single_path_multiple_containers(sample_tensor, sample_tensor2):
    """Test collating multiple containers with one dataset path."""
    container1 = DataContainer()
    container1.add_dataset("/", "data", sample_tensor)

    container2 = DataContainer()
    container2.add_dataset("/", "data", sample_tensor2)

    collate_fn = container_collate("/data")
    result = collate_fn([container1, container2])

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape[0] == 2
    torch.testing.assert_close(result[0][0], sample_tensor)
    torch.testing.assert_close(result[0][1], sample_tensor2)


def test_multiple_paths_single_container(empty_container, sample_tensor, sample_eye_tensor):
    """Test collating a single container with multiple dataset paths."""
    empty_container.add_dataset("/", "data1", sample_tensor)
    empty_container.add_dataset("/", "data2", sample_eye_tensor)

    collate_fn = container_collate("/data1", "/data2")
    result = collate_fn([empty_container])

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == (1, 2, 2)
    assert result[1].shape == (1, 3, 3)


def test_nonexistent_path_raises_keyerror(empty_container, sample_tensor):
    """Test that accessing a non-existent path raises KeyError."""
    empty_container.add_dataset("/", "data", sample_tensor)

    collate_fn = container_collate("/nonexistent")

    with pytest.raises(KeyError, match="One or more dataset paths not found in container"):
        collate_fn([empty_container])


def test_mixed_valid_invalid_paths(empty_container, sample_tensor):
    """Test behavior when some paths exist and others don't."""
    empty_container.add_dataset("/", "data", sample_tensor)

    collate_fn = container_collate("/data", "/nonexistent")

    with pytest.raises(KeyError, match="One or more dataset paths not found in container"):
        collate_fn([empty_container])


def test_incompatible_tensor_shapes(sample_tensor, sample_eye_tensor):
    """Test that incompatible tensor shapes raise RuntimeError."""
    container1 = DataContainer()
    container1.add_dataset("/", "data", sample_tensor)  # 2x2

    container2 = DataContainer()
    container2.add_dataset("/", "data", sample_eye_tensor)  # 3x3

    collate_fn = container_collate("/data")

    with pytest.raises(RuntimeError, match="Cannot stack tensors for path '/data'"):
        collate_fn([container1, container2])


def test_empty_batch_raises_error():
    """Test that empty batch raises ValueError."""
    collate_fn = container_collate("/data")

    with pytest.raises(ValueError, match="Empty batch provided - cannot collate empty sequence"):
        collate_fn([])


def test_nested_path_access(filled_container):
    """Test accessing datasets with nested paths using filled_container fixture."""
    collate_fn = container_collate("/TestGroup/TestDataset1")
    result = collate_fn([filled_container])

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape[0] == 1


def test_scalar_tensors():
    """Test collating containers with scalar tensors."""
    container1 = DataContainer()
    container1.add_dataset("/", "scalar", torch.tensor(1.0))

    container2 = DataContainer()
    container2.add_dataset("/", "scalar", torch.tensor(2.0))

    collate_fn = container_collate("/scalar")
    result = collate_fn([container1, container2])

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (2,)
    torch.testing.assert_close(result[0], torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("batch_size", [5, 10])
def test_large_batches(sample_tensor, batch_size):
    """Test collating larger batches."""
    containers = []
    for i in range(batch_size):
        container = DataContainer()
        tensor_data = sample_tensor.float() + i * 0.1
        container.add_dataset("/", "data", tensor_data)
        containers.append(container)

    collate_fn = container_collate("/data")
    result = collate_fn(containers)

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (batch_size, 2, 2)


def test_collate_function_reusability(sample_tensor, sample_tensor2):
    """Test that the same collate function can be used multiple times."""
    container1 = DataContainer()
    container1.add_dataset("/", "data", sample_tensor)

    container2 = DataContainer()
    container2.add_dataset("/", "data", sample_tensor2)

    collate_fn = container_collate("/data")

    result1 = collate_fn([container1])
    result2 = collate_fn([container2])
    result3 = collate_fn([container1, container2])

    assert result1[0].shape == (1, 2, 2)
    assert result2[0].shape == (1, 2, 2)
    assert result3[0].shape == (2, 2, 2)


def test_path_ordering_preservation(sample_tensor, sample_tensor2, sample_eye_tensor):
    """Test that the order of paths in the result tuple matches input order."""
    container = DataContainer()
    container.add_dataset("/", "data1", sample_tensor)
    container.add_dataset("/", "data2", sample_tensor2)
    container.add_dataset("/", "data3", sample_eye_tensor)

    collate_fn1 = container_collate("/data1", "/data2", "/data3")
    collate_fn2 = container_collate("/data3", "/data1", "/data2")

    result1 = collate_fn1([container])
    result2 = collate_fn2([container])

    # Check that ordering is preserved
    torch.testing.assert_close(result1[0][0], sample_tensor)
    torch.testing.assert_close(result1[1][0], sample_tensor2)
    torch.testing.assert_close(result1[2][0], sample_eye_tensor)

    torch.testing.assert_close(result2[0][0], sample_eye_tensor)
    torch.testing.assert_close(result2[1][0], sample_tensor)
    torch.testing.assert_close(result2[2][0], sample_tensor2)
