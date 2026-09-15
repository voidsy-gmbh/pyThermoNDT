"""What we measure: one list of transforms, one list of pipelines.

How we measure lives in ``conftest.py``. Each spec names an optional prep
chain (``TO_TIME``, ``TO_FREQ``) that runs once before the timer starts.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest
import torch

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

BENCHMARK_SPECS: list[BenchmarkSpec] = [
    BenchmarkSpec("ApplyLUT", T.ApplyLUT()),
    BenchmarkSpec("MinMaxNormalize", T.MinMaxNormalize(), TO_TIME),
    BenchmarkSpec("MaxNormalize", T.MaxNormalize(), TO_TIME),
    BenchmarkSpec("ZScoreNormalize", T.ZScoreNormalize(), TO_TIME),
    BenchmarkSpec("SubtractFrame", T.SubtractFrame(0), TO_TIME),
    BenchmarkSpec("SelectFrameRange", T.SelectFrameRange(start=0, end=50), TO_TIME),
    BenchmarkSpec("SelectFrames", T.SelectFrames(list(range(0, 100, 10))), TO_TIME),
    BenchmarkSpec("CropFrames", T.CropFrames(height=16, width=16), TO_TIME),
    BenchmarkSpec("CastTo", T.CastTo("/Data/Tdata", torch.float32), TO_TIME),
    BenchmarkSpec("RemoveFlash", T.RemoveFlash(method="excitation_signal"), TO_TIME),
    BenchmarkSpec("NonUniformSampling", T.NonUniformSampling(100), TO_TIME),
    BenchmarkSpec("PulsePhaseThermography", T.PulsePhaseThermography(), TO_TIME),
    BenchmarkSpec("ExtractAmplitude", T.ExtractAmplitude(), TO_FREQ),
    BenchmarkSpec("ExtractPhase", T.ExtractPhase(), TO_FREQ),
    BenchmarkSpec("GaussianNoise", T.GaussianNoise(std=25e-3), TO_TIME),
    BenchmarkSpec("AdaptiveGaussianNoise", T.AdaptiveGaussianNoise(std_range=(0.0, 25e-3)), TO_TIME),
    BenchmarkSpec("RandomFlip", T.RandomFlip(), TO_TIME),
]

PIPELINES: list[BenchmarkSpec] = [
    BenchmarkSpec("ingest", T.Compose([T.ApplyLUT(), T.SubtractFrame(0), T.MinMaxNormalize()])),
    BenchmarkSpec(
        "preprocess",
        T.Compose([T.SubtractFrame(0), T.RemoveFlash(method="excitation_signal"), T.MinMaxNormalize()]),
        TO_TIME,
    ),
    BenchmarkSpec("frequency", T.Compose([T.PulsePhaseThermography(), T.ExtractPhase()]), TO_TIME),
]
