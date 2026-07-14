"""Tests for BaseDataset behavior exercised through ThermoDataset and IndexedThermoDataset."""

import gc
import multiprocessing as mp
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

from pythermondt import LocalReader, ThermoDataset
from pythermondt.dataset import container_collate
from pythermondt.dataset.base_dataset import CacheMode
from pythermondt.transforms import ApplyLUT

# Contexts that pickle the dataset into workers (the path broken by lazy Manager state).
_MP_CONTEXTS = [ctx for ctx in ("spawn", "forkserver") if ctx in mp.get_all_start_methods()]


@pytest.mark.parametrize("idx", [0, 1])
def test_map_index_default_returns_identity(sample_dataset_three_files: ThermoDataset, idx: int):
    """Test that the default _map_index returns the index unchanged."""
    assert sample_dataset_three_files._map_index(idx) == idx


def test_memory_bytes_without_cache(sample_dataset_single_file: ThermoDataset):
    """Test memory_bytes returns a small positive value when no cache is built."""
    mem = sample_dataset_single_file.memory_bytes()
    assert isinstance(mem, int)
    assert mem > 0


def test_memory_bytes_with_cache(local_reader_three_files: LocalReader):
    """Test memory_bytes returns a larger value after building cache."""
    dataset = ThermoDataset(local_reader_three_files)
    mem_before = dataset.memory_bytes()
    dataset.build_cache(mode="immediate")
    mem_after = dataset.memory_bytes()
    assert mem_after > mem_before
    dataset.release_cache()


def test_print_memory_usage(local_reader_three_files: LocalReader, capsys):
    """Test print_memory_usage outputs expected information."""
    dataset = ThermoDataset(local_reader_three_files)
    dataset.build_cache(mode="immediate")
    dataset.print_memory_usage()

    captured = capsys.readouterr()
    assert "ThermoDataset Overview:" in captured.out
    assert "items in the cache" in captured.out
    assert "Total memory usage" in captured.out
    dataset.release_cache()


def test_build_cache_idempotent(local_reader_three_files: LocalReader):
    """Test that calling build_cache twice is a no-op the second time."""
    dataset = ThermoDataset(local_reader_three_files)
    dataset.build_cache(mode="immediate")
    assert dataset.cache_built

    # Second call should return early without error
    dataset.build_cache(mode="immediate")
    assert dataset.cache_built

    # Data should still be accessible
    _ = dataset[0]
    dataset.release_cache()


def test_build_cache_invalid_mode(local_reader_three_files: LocalReader):
    """Test that an invalid cache mode raises ValueError."""
    dataset = ThermoDataset(local_reader_three_files)
    with pytest.raises(ValueError, match="Invalid cache mode"):
        dataset.build_cache(mode="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["lazy", "immediate"])
@pytest.mark.parametrize("mp_context", _MP_CONTEXTS)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_build_cache_dataloader_multiprocess(
    local_reader_three_files: LocalReader,
    mode: CacheMode,
    mp_context: str,
    num_workers: int,
):
    """Regression test for #465: build_cache + DataLoader under pickling start methods."""
    dataset = ThermoDataset(
        local_reader_three_files,
        transform=ApplyLUT(target_dtype=torch.float32),
    )
    dataset.build_cache(mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context if num_workers > 0 else None,
        collate_fn=container_collate("/Data/Tdata"),
    )

    # Try and finally for cleanup to ensure cache is released even if the test fails.
    try:
        batch = next(iter(loader))
        assert len(batch) == 1
        assert batch[0].shape[0] == 2
    finally:
        # Drain workers before releasing the cache manager.
        del loader
        dataset.release_cache()


def test_release_cache_manager_shutdown_exception(local_reader_three_files: LocalReader):
    """Test that release_cache handles manager.shutdown() exceptions gracefully."""
    dataset = ThermoDataset(local_reader_three_files)
    dataset.build_cache(mode="lazy")

    # Replace the manager with a mock that raises on shutdown
    mock_manager = MagicMock()
    mock_manager.shutdown.side_effect = OSError("shutdown failed")
    # Access the name-mangled attribute
    object.__setattr__(dataset, "_BaseDataset__manager", mock_manager)

    # Should not raise
    dataset.release_cache()
    assert not dataset.cache_built
    mock_manager.shutdown.assert_called_once()


def test_release_cache_gc_collect(local_reader_three_files: LocalReader, monkeypatch):
    """Test that release_cache respects the gc_collect flag."""
    gc_called = []
    monkeypatch.setattr(gc, "collect", lambda: gc_called.append(True))

    # Test gc_collect=True (default)
    dataset1 = ThermoDataset(local_reader_three_files)
    dataset1.build_cache(mode="immediate")
    dataset1.release_cache(gc_collect=True)
    assert len(gc_called) == 1

    # Test gc_collect=False
    gc_called.clear()
    dataset2 = ThermoDataset(local_reader_three_files)
    dataset2.build_cache(mode="immediate")
    dataset2.release_cache(gc_collect=False)
    assert len(gc_called) == 0
