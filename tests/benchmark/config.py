"""Transform benchmark definitions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from pythermondt import transforms as T  # noqa: N812
from pythermondt.data import DataContainer


@dataclass
class BenchmarkSpec:
    """Single benchmark: what to run and its untimed prep chain."""

    name: str
    transform: Callable[[DataContainer], DataContainer]
    setup: Sequence[Callable[[DataContainer], DataContainer]] = field(default_factory=tuple)


TO_TIME: tuple[Callable[[DataContainer], DataContainer], ...] = (T.ApplyLUT(),)
TO_FREQ: tuple[Callable[[DataContainer], DataContainer], ...] = (T.ApplyLUT(), T.PulsePhaseThermography())

SOURCES = [
    pytest.param("small", marks=pytest.mark.local),
    pytest.param("fraunhofer", marks=pytest.mark.cloud),
]

TRANSFORMS: list[BenchmarkSpec] = [
    BenchmarkSpec("ApplyLUT", T.ApplyLUT()),
    BenchmarkSpec("MinMaxNormalize", T.MinMaxNormalize(), TO_TIME),
    BenchmarkSpec("ZScoreNormalize", T.ZScoreNormalize(), TO_TIME),
    BenchmarkSpec("SubtractFrame", T.SubtractFrame(0), TO_TIME),
    BenchmarkSpec("NonUniformSampling", T.NonUniformSampling(100), TO_TIME),
    BenchmarkSpec("PulsePhaseThermography", T.PulsePhaseThermography(), TO_TIME),
    BenchmarkSpec("ExtractPhase", T.ExtractPhase(), TO_FREQ),
]
