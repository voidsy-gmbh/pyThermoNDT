"""Dataset and reader benchmarks on real files.

Readers and datasets are built inside the test (never at collection time).
File I/O stays inside the timed region on purpose: this measures load cost.
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pythermondt import ThermoDataset
from pythermondt import transforms as T  # noqa: N812

from .config import SOURCES
from .conftest import PERF_FILE_INDEX, make_reader_or_skip


def _make_dataset(source: str, *, with_pipeline: bool) -> ThermoDataset:
    """Build a dataset over the benchmark files, optionally with an ingest pipeline."""
    dataset = ThermoDataset(
        make_reader_or_skip(source),
        transform=T.Compose([T.ApplyLUT(), T.SubtractFrame(0), T.MinMaxNormalize()]) if with_pipeline else None,
    )
    if len(dataset) == 0:
        raise ValueError(f"No benchmark files found for source {source!r}.")
    return dataset


@pytest.mark.benchmark
@pytest.mark.parametrize("source", SOURCES)
def test_dataset_getitem_raw(benchmark: BenchmarkFixture, source: str):
    """Benchmark cold file load via ``ThermoDataset.__getitem__``."""
    dataset = _make_dataset(source, with_pipeline=False)

    benchmark.group = "dataset"
    result = benchmark.pedantic(dataset.__getitem__, args=(PERF_FILE_INDEX,), rounds=3, warmup_rounds=1, iterations=1)

    assert result.get_dataset("/Data/Tdata") is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("source", SOURCES)
def test_dataset_getitem_pipeline(benchmark: BenchmarkFixture, source: str):
    """Benchmark file load plus a realistic ingest pipeline."""
    dataset = _make_dataset(source, with_pipeline=True)

    benchmark.group = "dataset"
    result = benchmark.pedantic(dataset.__getitem__, args=(PERF_FILE_INDEX,), rounds=3, warmup_rounds=1, iterations=1)

    assert result.get_dataset("/Data/Tdata") is not None
