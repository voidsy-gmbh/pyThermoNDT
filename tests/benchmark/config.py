from collections.abc import Callable
from dataclasses import dataclass

from pythermondt import DataContainer, LocalReader
from pythermondt import transforms as T  # noqa: N812
from pythermondt.readers import BaseReader


@dataclass
class BenchmarkSpec:
    """Configuration for a single transform benchmark."""

    name: str
    transform: Callable[[DataContainer], DataContainer]
    setup: Callable[[DataContainer], DataContainer] | None = None


@dataclass
class ReaderSpec:
    """Specification for a reader."""

    name: str
    reader: BaseReader


READER_SPECS = [
    ReaderSpec(name="small", reader=LocalReader(pattern="tests/assets/perf/small", recursive=True)),
]


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
    BenchmarkSpec(
        name="RemoveFlash",
        transform=T.RemoveFlash(),
    ),
]
