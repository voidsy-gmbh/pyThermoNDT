import os

import torch

from pythermondt.data import DataContainer
from pythermondt.io.parsers import find_parser_for_extension
from pythermondt.readers import LocalReader


def update_expected_outputs(source_folder: str, file_extension: str):
    """Update all expected outputs based on source files.

    Args:
        source_folder (str): Path to folder containing source files
        file_extension (str): File extension of the source files (e.g., ".mat", ".hdf5")
    """
    source_reader = LocalReader(source_folder, parser=find_parser_for_extension(file_extension))

    print(f"\nUpdating expected outputs for {source_reader.files}")

    updated_files = []
    for source_path in source_reader.files:
        source_container = source_reader.read_file(source_path)
        head, tail = os.path.split(source_path)
        output_name = tail.replace(file_extension, ".hdf5")
        output_name = output_name.replace("source", "expected")
        output_path = os.path.join(head, output_name)
        source_container.save_to_hdf5(output_path)
        updated_files.append(output_path)

    print(f"\nUpdated expected outputs: {updated_files}")


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
