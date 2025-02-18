import re
from typing import TypeGuard

from .node import BaseNode, DataNode, GroupNode, NodeType, RootNode


def is_rootnode(node: BaseNode) -> "TypeGuard[RootNode]":
    """Checks if the given node is a RootNode.

    Parameters:
        node (BaseNode): The node to check.

    Returns:
        bool: True if the node is a RootNode, False otherwise.
    """
    return isinstance(node, RootNode) and node.type == NodeType.ROOT


def is_groupnode(node: BaseNode) -> "TypeGuard[GroupNode]":
    """Checks if the given node is a GroupNode.

    Parameters:
        node (BaseNode): The node to check.

    Returns:
        bool: True if the node is a GroupNode, False otherwise.
    """
    return isinstance(node, GroupNode) and node.type == NodeType.GROUP


def is_datanode(node: BaseNode) -> "TypeGuard[DataNode]":
    """Checks if the given node is a DataNode.

    Parameters:
        node (BaseNode): The node to check.

    Returns:
        bool: True if the node is a DataNode, False otherwise.
    """
    return isinstance(node, DataNode) and node.type == NodeType.DATASET


def validate_path(path: str, name: str = "") -> str:
    """Validates and normalizes the given HDF5 path.

    Parameters:
        path (str): The path to validate and normalize.
        name (str, optional): The name of the group or dataset to add to the path. Defaults to "".

    Returns:
        A normalized valid HDF5 path.

    Raises:
        ValueError: If the path is not valid.
    """
    # Add name to the path
    if name and path[-1] != "/":
        path = path + "/" + name
    else:
        path = path + name

    # Normalize: strip leading/trailing whitespace and ensure starting with a slash
    normalized_path = path.strip()
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path

    # Check for double slashes or trailing slash ==> remove them
    if "//" in normalized_path:
        normalized_path = re.sub(r"/+", "/", normalized_path)

    if normalized_path != "/" and normalized_path.endswith("/"):
        normalized_path = normalized_path[:-1]

    # Validate using a regex pattern
    pattern = r"^/[a-zA-Z0-9_/-]+$"
    if not re.match(pattern, normalized_path):
        raise ValueError(f"Invalid path: {normalized_path} contains invalid characters.")

    return normalized_path


def validate_paths(paths: list[str]) -> tuple[str, ...]:
    """Validates and normalizes the given list of HDF5 paths.

    Parameters:
        paths (List[str]): The list of paths to validate and normalize.

    Returns:
        A list of normalized valid HDF5 paths.

    Raises:
        ValueError: If any of the paths is not valid.
    """
    return tuple(validate_path(path) for path in paths)


def split_path(path: str) -> tuple[str, str]:
    """Splits the given HDF5 path into the parent path and the name of the group or dataset.

    Parameters:
        path (str): The path to split.

    Returns:
        parent (str): The parent path.
        child (str): The name of the group or dataset.
    """
    parent, child = path.rsplit("/", 1)
    if path.count("/") == 1:
        return "/", child
    else:
        return parent, child


def generate_key(path: str, name: str) -> tuple[str, str, str]:
    """Generates a key for the given path and name. First the path is validated and normalized, then it is split into the
    parent path and the name of the group or dataset.

    Parameters:
        path (str): The path to generate the key for.
        name (str): The name of the group or dataset.

    Returns:
        key (str): The generated key to be saved in the dictionary.
        parent (str): The parent path.
        child (str): The name of the group or dataset.
    """
    key = validate_path(path, name)
    head, tail = split_path(key)
    return key, head, tail
