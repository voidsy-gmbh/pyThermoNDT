"""How we measure: load real files once, prep them, hand a fresh copy to the timer.

Two sources: ``small`` (in-repo cube, always available) and ``fraunhofer``
(S3 cube, larger, skipped without credentials). Sources load lazily inside
the test, never at collection time.
"""

import copy
from collections.abc import Callable, Sequence
from functools import lru_cache

import pytest
from botocore.exceptions import ClientError, NoCredentialsError, SSOTokenLoadError, TokenRetrievalError

from pythermondt import LocalReader, S3Reader
from pythermondt.data import DataContainer
from pythermondt.readers import BaseReader

PERF_PATTERN = "tests/assets/perf/small"
PERF_FILE_INDEX = 0


def make_reader(source: str) -> BaseReader:
    """Create the reader for a benchmark source (no I/O yet)."""
    if source == "small":
        return LocalReader(pattern=PERF_PATTERN, recursive=True)
    if source == "fraunhofer":
        return S3Reader("ffg-bp", "benchmark_datasets/fraunhofer", download_files=True, num_files=3)
    raise ValueError(f"Unknown benchmark source: {source}.")


def make_reader_or_skip(source: str) -> BaseReader:
    """Create and download a reader, skipping when cloud data is unavailable."""
    try:
        reader = make_reader(source)
        reader.download()
        return reader
    except (NoCredentialsError, ClientError, SSOTokenLoadError, TokenRetrievalError) as e:
        pytest.skip(f"Benchmark source {source!r} unavailable: {e}")
        raise  # Unreachable: pytest.skip raises, this satisfies the type checker.


@lru_cache(maxsize=4)
def load_raw_container(source: str = "small", index: int = PERF_FILE_INDEX) -> DataContainer:
    """Load one raw container (cached per session)."""
    reader = make_reader(source)
    reader.download()
    if index < 0 or index >= len(reader.file_uris):
        raise ValueError(f"File index {index} out of range [0, {len(reader.file_uris) - 1}].")
    return reader[index]


def prepare_base_or_skip(
    setup: Sequence[Callable[[DataContainer], DataContainer]],
    source: str = "small",
    index: int = PERF_FILE_INDEX,
) -> DataContainer:
    """Prep the base container once (untimed), skipping when cloud data is unavailable."""
    try:
        base = copy.deepcopy(load_raw_container(source, index))
    except (NoCredentialsError, ClientError, SSOTokenLoadError, TokenRetrievalError) as e:
        pytest.skip(f"Benchmark source {source!r} unavailable: {e}")
        raise  # Unreachable: pytest.skip raises, this satisfies the type checker.
    for transform in setup:
        base = transform(base)
    return base


def fresh_copy(container: DataContainer) -> tuple[tuple[DataContainer], dict]:
    """Return a ``pedantic`` setup callable producing a fresh container per round.

    ``pedantic`` setup must return ``(args, kwargs)``. The copy runs outside
    the timed region, so only the target itself is measured.
    """
    return ((copy.deepcopy(container),), {})
