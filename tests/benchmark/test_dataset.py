"""Reader benchmarks on real files."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from .config import SOURCES
from .conftest import PERF_FILE_INDEX, make_reader


@pytest.mark.benchmark
@pytest.mark.parametrize("source", SOURCES)
def test_reader_load(benchmark: BenchmarkFixture, source: str):
    """Benchmark repeated loading and parsing of one sample."""
    reader = make_reader(source)
    reader.download()
    if len(reader) == 0:
        raise ValueError(f"No benchmark files found for source {source!r}.")

    benchmark.group = "reader"
    result = benchmark.pedantic(reader.__getitem__, args=(PERF_FILE_INDEX,), rounds=3, iterations=1)

    assert result.get_dataset("/Data/Tdata") is not None
