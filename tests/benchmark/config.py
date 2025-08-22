from collections.abc import Callable
from dataclasses import dataclass

import pytest

from pythermondt import DataContainer, LocalReader, S3Reader
from pythermondt import transforms as T  # noqa: N812
from pythermondt.readers import BaseReader


@dataclass
class BenchmarkSpec:
    """Configuration for a single transform benchmark."""

    name: str
    transform: Callable[[DataContainer], DataContainer]
    setup: Callable[[DataContainer], DataContainer] | None = None


@dataclass
class BenchmarkData:
    """Specification for a reader."""

    name: str
    reader: BaseReader
    marker: pytest.MarkDecorator


LOCAL_READER = [
    BenchmarkData(
        name="small", reader=LocalReader(pattern="tests/assets/perf/small", recursive=True), marker=pytest.mark.local
    ),
]

CLOUD_READER = [
    BenchmarkData(
        name="medium-fraunhofer",
        reader=S3Reader("ffg-bp", "benchmark_datasets/fraunhofer", download_files=True, num_files=1),
        marker=pytest.mark.cloud,
    ),
]

BENCHMARK_DATA = LOCAL_READER + CLOUD_READER

BENCHMARK_SPECS = [
    BenchmarkSpec(
        name="ApplyLUT",
        transform=T.ApplyLUT(),
    ),
    BenchmarkSpec(
        name="MinMaxNormalize",
        setup=T.ApplyLUT(),
        transform=T.MinMaxNormalize(),
    ),
    BenchmarkSpec(
        name="SelectFrameRange",
        transform=T.SelectFrameRange(start=0, end=50),
    ),
    BenchmarkSpec(
        name="NonUniformSampling",
        setup=T.ApplyLUT(),
        transform=T.NonUniformSampling(200),
    ),
    BenchmarkSpec(
        name="SubtractFrame",
        setup=T.ApplyLUT(),
        transform=T.SubtractFrame(0),
    ),
]
