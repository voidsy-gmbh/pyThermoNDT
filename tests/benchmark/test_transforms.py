import copy

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pythermondt import DataContainer

from .config import BENCHMARK_SPECS, READER_SPECS, BenchmarkSpec


def get_readers():
    """Get all readers for parameterization."""
    return [pytest.param(reader, id=reader.name) for reader in READER_SPECS]


def get_test_files_as_container():
    """Get all (index, container) combinations for parameterization."""
    for reader_spec in READER_SPECS:
        for idx, container in enumerate(reader_spec.reader):
            combo_id = f"{reader_spec.name}_{idx}"
            yield pytest.param((idx, reader_spec.name, container), id=combo_id)


@pytest.mark.parametrize("data_config", get_test_files_as_container())
@pytest.mark.parametrize("benchmark_config", BENCHMARK_SPECS, ids=lambda config: config.name)
def test_benchmark_transform(
    benchmark: BenchmarkFixture,
    benchmark_config: BenchmarkSpec,
    data_config: tuple[int, str, DataContainer],
):
    """Benchmark individual transforms across different readers and files."""
    idx, name, container = data_config

    # Apply setup transform if specified
    if benchmark_config.setup:
        container = benchmark_config.setup(container)

    def run_transform():
        """Run the transform on a fresh copy of the container."""
        container_copy = copy.deepcopy(container)
        benchmark_config.transform(container_copy)

    # Set up grouping and naming
    benchmark.group = benchmark_config.name
    benchmark.name = f"{benchmark_config.name}_{name}_{idx}"
    benchmark(run_transform)
