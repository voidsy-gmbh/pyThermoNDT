from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest
import torch

from pythermondt.data import DataContainer
from tests.utils import make_container


@dataclass(frozen=True)
class HDF5TestCorpus:
    """Immutable collection of serialized HDF5 test files."""

    files: Mapping[str, bytes]


@pytest.fixture(scope="session")
def hdf5_test_corpus() -> Callable[[int], HDF5TestCorpus]:
    """Return a lazily cached deterministic HDF5 corpus factory."""
    cache: dict[int, HDF5TestCorpus] = {}

    def make_corpus(num_files: int) -> HDF5TestCorpus:
        if num_files not in cache:
            files: dict[str, bytes] = {}
            for index in range(num_files):
                container = make_container(("/Data", f"t{index}", torch.full((2, 2), float(index))))
                container.add_attribute("/Data", "index", index)
                files[f"file_{index}.hdf5"] = container.serialize_to_hdf5().getvalue()
            cache[num_files] = HDF5TestCorpus(files)
        return cache[num_files]

    return make_corpus


@pytest.fixture
def test_container() -> DataContainer:
    """Return a small DataContainer for writer round-trip tests."""
    container = make_container(("/Data", "values", torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
    container.add_attribute("/Data", "description", "test data")
    return container
