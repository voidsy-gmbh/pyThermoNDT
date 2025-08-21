import copy

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from .config import BENCHMARK_SPECS, READER_SPECS, BenchmarkSpec, ReaderSpec


def get_readers():
    """Get all readers for parameterization."""
    return [pytest.param(reader, id=reader.name) for reader in READER_SPECS]


def get_file_indices_for_readers():
    """Get all (reader, file_index) combinations for parameterization."""
    combinations = []
    for reader_spec in READER_SPECS:
        for file_index in range(len(reader_spec.reader)):
            combo_id = f"{reader_spec.name}_{file_index}"
            combinations.append(pytest.param((reader_spec, file_index), id=combo_id))
    return combinations


@pytest.mark.parametrize("reader_file_combo", get_file_indices_for_readers())
@pytest.mark.parametrize("benchmark_config", BENCHMARK_SPECS, ids=lambda config: config.name)
def test_benchmark_transform(
    benchmark: BenchmarkFixture,
    benchmark_config: BenchmarkSpec,
    reader_file_combo: tuple[ReaderSpec, int],
):
    """Benchmark individual transforms across different readers and files."""
    reader_spec, file_index = reader_file_combo

    # Setup: Load and prepare the container (excluded from benchmark timing)
    container = reader_spec.reader[file_index]
    if benchmark_config.setup:
        container = benchmark_config.setup(container)

    def run_transform():
        """Run the transform on a fresh copy of the container."""
        container_copy = copy.deepcopy(container)
        benchmark_config.transform(container_copy)

    # Set up grouping and naming
    benchmark.group = benchmark_config.name
    benchmark.name = f"{benchmark_config.name}_{reader_spec.name}_{file_index}"
    benchmark(run_transform)
