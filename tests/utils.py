import os

import torch
from deepdiff import DeepDiff

from pythermondt.data import DataContainer, Units
from pythermondt.data.datacontainer.node import AttributeNode, DataNode
from pythermondt.io.parsers import find_parser_for_extension
from pythermondt.readers import LocalReader


def containers_equal(container1: DataContainer, container2: DataContainer, print_diff=True) -> bool:
    """Compare two DataContainers and optionally print differences.

    Args:
        container1: First DataContainer
        container2: Second DataContainer
        print_diff: Whether to print differences to stdout

    Returns:
        True if containers are equal, False otherwise
    """
    equal = True
    differences = []

    # Compare structure (paths)
    paths1 = set(container1.nodes.keys())
    paths2 = set(container2.nodes.keys())

    if paths1 != paths2:
        equal = False
        missing_in_2 = paths1 - paths2
        missing_in_1 = paths2 - paths1

        if missing_in_2:
            differences.append(f"Paths in container1 but not in container2: {missing_in_2}")
        if missing_in_1:
            differences.append(f"Paths in container2 but not in container1: {missing_in_1}")

    # For common paths, compare nodes
    common_paths = paths1.intersection(paths2)
    for path in common_paths:
        node1 = container1.nodes[path]
        node2 = container2.nodes[path]

        # Compare basic node properties
        if node1.type != node2.type:
            equal = False
            differences.append(f"Node type mismatch at path {path}: {node1.type} vs {node2.type}")
            continue

        if node1.name != node2.name:
            equal = False
            differences.append(f"Node name mismatch at path {path}: {node1.name} vs {node2.name}")

        # Compare attributes if the node is of type AttributeNode
        if isinstance(node1, AttributeNode) and isinstance(node2, AttributeNode):
            # Compare attributes using DeepDiff
            attrs1 = dict(node1.attributes)
            attrs2 = dict(node2.attributes)
            diff = DeepDiff(attrs1, attrs2)

            # Handle diff output for different types
            if diff:
                equal = False
                diff_text = diff.pretty(prefix="\t")
                differences.append(f"Attribute mismatch at path {path}: \n{diff_text}")

        # Compare the data if the node is of type DataNode
        if isinstance(node1, DataNode) and isinstance(node2, DataNode):
            data1, data2 = node1.data, node2.data
            if not torch.equal(data1, data2):
                equal = False
                if data1.shape != data2.shape:
                    differences.append(f"Data shape mismatch at path {path}: {data1.shape} vs {data2.shape}")
                else:
                    differences.append(f"Data content mismatch at path {path}")

    # Print differences if requested and there are any
    if print_diff and differences:
        print("\nContainer differences:")
        for diff in differences:
            print(f"- {diff}")

    return equal


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
        source_container = source_reader.read(source_path)
        head, tail = os.path.split(source_path)
        output_name = tail.replace(file_extension, ".hdf5")
        output_name = output_name.replace("source", "expected")
        output_path = os.path.join(head, output_name)
        source_container.save_to_hdf5(output_path)
        updated_files.append(output_path)

    print(f"\nUpdated expected outputs: {updated_files}")


if __name__ == "__main__":
    # Example usage
    container1 = DataContainer()
    container2 = DataContainer()

    # Add nodes and data to the containers for testing
    # ...
    container1.add_dataset("/", "Dataset1", torch.tensor([[1, 2], [3, 4]]))
    container2.add_dataset("/", "Dataset1", torch.tensor([[1, 2], [7, 5]]))
    container1.add_unit("/Dataset1", Units.kelvin)
    container2.add_unit("/Dataset1", Units.celsius)
    container1.add_attribute("/Dataset1", "Attribute1", "Value1")

    # Compare the containers
    result = containers_equal(container1, container2)
    print(f"Containers are equal: {result}")
