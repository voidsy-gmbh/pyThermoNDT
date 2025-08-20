import copy

from pytest_benchmark.fixture import BenchmarkFixture

from pythermondt import DataContainer, LocalReader
from pythermondt import transforms as T  # noqa: N812


def test_benchmark_applylut(benchmark: BenchmarkFixture):
    reader = LocalReader(pattern="tests/assets/perf", recursive=True)
    transform = T.ApplyLUT()

    def run_transform(container: DataContainer):
        cp_container = copy.deepcopy(container)
        return transform(cp_container)

    benchmark(run_transform, reader[0])
