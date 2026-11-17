from collections.abc import Iterable

import torch

from pythermondt.data import DataContainer


def format_container_diff(differences: Iterable[str]) -> str:
    """Format container differences as an indented list without printing."""
    return "\n".join(f"  - {difference}" for difference in differences)


def make_container(*datasets: tuple[str, str, torch.Tensor]) -> DataContainer:
    """Helper to build a DataContainer with datasets at specified paths.

    Automatically creates parent groups if they don't exist.
    """
    c = DataContainer()
    created_groups: set[str] = set()
    for path, name, data in datasets:
        if path not in created_groups:
            # path like "/Data" -> add_group("/", "Data")
            parent, group_name = path.rsplit("/", 1)
            c.add_group(parent or "/", group_name)
            created_groups.add(path)
        c.add_dataset(path, name, data)
    return c
