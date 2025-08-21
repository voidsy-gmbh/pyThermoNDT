"""Utilities for benchmark tests."""

from collections.abc import Callable
from dataclasses import dataclass

from pythermondt import DataContainer, LocalReader


@dataclass
class ReaderConfig:
    """Configuration for test data readers."""

    small: LocalReader
    all: LocalReader


@dataclass
class BenchmarkConfig:
    """Configuration for a single transform benchmark."""

    name: str
    transform: Callable[[DataContainer], DataContainer]
    setup: Callable[[DataContainer], DataContainer] | None = None


def discover_perf_readers() -> ReaderConfig:
    """Discover and create readers for performance test data."""
    return ReaderConfig(
        small=LocalReader(pattern="tests/assets/perf/small", recursive=True),
        all=LocalReader(pattern="tests/assets/perf", recursive=True),
    )
