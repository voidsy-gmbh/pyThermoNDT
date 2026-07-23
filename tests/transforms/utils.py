"""Helpers for transform tests."""

from collections.abc import Callable

from pythermondt.data import DataContainer, ThermoContainer

ThermoSequenceFactory = Callable[..., ThermoContainer]


def has_dataset(container: DataContainer, path: str) -> bool:
    """Return whether ``path`` is present as a dataset node."""
    return path in container.get_all_dataset_paths()
