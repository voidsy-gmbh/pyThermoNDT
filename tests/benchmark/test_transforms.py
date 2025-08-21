import copy

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pythermondt import DataContainer
from pythermondt import transforms as T  # noqa: N812

from .utils import BenchmarkConfig, discover_perf_readers

# Define benchmark configurations as constants
BENCHMARK_CONFIGS = [
    BenchmarkConfig(
        name="ApplyLUT",
        transform=T.ApplyLUT(),
    ),
    BenchmarkConfig(
        name="MinMaxNormalize",
        setup=T.ApplyLUT(),
        transform=T.MinMaxNormalize(),
    ),
    BenchmarkConfig(
        name="SelectFrameRange",
        transform=T.SelectFrameRange(start=0, end=50),
    ),
    BenchmarkConfig(
        name="NonUniformSampling",
        setup=T.ApplyLUT(),
        transform=T.NonUniformSampling(10),
    ),
    BenchmarkConfig(
        name="SubtractFrame",
        setup=T.ApplyLUT(),
        transform=T.SubtractFrame(0),
    ),
    BenchmarkConfig(
        name="RemoveFlash",
        transform=T.RemoveFlash(),
    ),
]


@pytest.mark.parametrize("config", BENCHMARK_CONFIGS, ids=lambda config: config.name)
def test_benchmark_transform(benchmark: BenchmarkFixture, config: BenchmarkConfig):
    """Benchmark individual transforms."""
    readers = discover_perf_readers()

    if len(readers.small) == 0:
        pytest.skip("No test files found")

    # Pre-setup: Load and prepare the container once
    original_container = config.setup(readers.small[0]) if config.setup else readers.small[0]

    def run_transform() -> DataContainer:
        """Run the transform on a fresh copy of the container to avoid issues with mutation."""
        container_copy = copy.deepcopy(original_container)
        return config.transform(container_copy)

    # Benchmark the transform with automatic calibration
    benchmark.group = config.name
    benchmark(run_transform)
