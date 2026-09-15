"""End-to-end pipeline benchmarks on real data."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from .config import PIPELINES, SOURCES, BenchmarkSpec
from .conftest import fresh_copy, prepare_base_or_skip


@pytest.mark.benchmark
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("spec", PIPELINES, ids=lambda spec: spec.name)
def test_pipeline(benchmark: BenchmarkFixture, spec: BenchmarkSpec, source: str):
    """Benchmark a realistic Compose chain on a fresh real-data container per round."""
    base = prepare_base_or_skip(spec.setup, source)

    benchmark.group = f"pipeline_{spec.name}"
    result = benchmark.pedantic(
        spec.transform,
        setup=lambda: fresh_copy(base),
        rounds=5,
        warmup_rounds=1,
        iterations=1,
    )

    tdata = result.get_dataset("/Data/Tdata")
    assert tdata is not None and tdata.numel() > 0
