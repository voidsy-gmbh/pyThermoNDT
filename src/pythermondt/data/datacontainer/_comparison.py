from collections.abc import Iterator
from dataclasses import fields

import numpy as np
import torch

from ..units import Unit
from .base import NodeAccessor
from .node import AttributeNode, DataNode


def _iter_node_differences(
    nodes1: NodeAccessor,
    nodes2: NodeAccessor,
    *,
    ignore_attribute_nan_inequality: bool = False,
) -> Iterator[str]:
    """Compare nodes without depending on the concrete DataContainer class."""
    # Compare structure (paths)
    paths1 = set(nodes1.keys())
    paths2 = set(nodes2.keys())

    if paths1 != paths2:
        missing_in_2 = paths1 - paths2
        missing_in_1 = paths2 - paths1

        if missing_in_2:
            yield f"Paths in container1 but not in container2: {missing_in_2}"
        if missing_in_1:
            yield f"Paths in container2 but not in container1: {missing_in_1}"

    # For common paths, compare nodes
    common_paths = paths1.intersection(paths2)
    for path in common_paths:
        node1 = nodes1[path]
        node2 = nodes2[path]

        # Compare basic node properties
        if node1.type != node2.type:
            yield f"Node type mismatch at path {path}: {node1.type} vs {node2.type}"
            continue

        if node1.name != node2.name:
            yield f"Node name mismatch at path {path}: {node1.name} vs {node2.name}"

        # Compare attributes if the node is of type AttributeNode
        if isinstance(node1, AttributeNode) and isinstance(node2, AttributeNode):
            attrs1 = dict(node1.attributes)
            attrs2 = dict(node2.attributes)
            yield from _iter_attribute_differences(attrs1, attrs2, path, ignore_attribute_nan_inequality)

        # Compare the data if the node is of type DataNode
        if isinstance(node1, DataNode) and isinstance(node2, DataNode):
            data1, data2 = node1.data, node2.data
            if not torch.equal(data1, data2):
                if data1.shape != data2.shape:
                    yield f"Data shape mismatch at path {path}: {data1.shape} vs {data2.shape}"
                else:
                    yield f"Data content mismatch at path {path}"


def _iter_attribute_differences(  # pylint: disable=too-many-branches
    value1: object, value2: object, path: str, ignore_nan_inequality: bool
) -> Iterator[str]:
    """Compare attribute values, including nested metadata."""
    # Preserve Python equality for shared objects, including attribute NaNs.
    if value1 is value2:
        return

    # Defensive handling: AttributeTypes excludes np.ndarray, but arrays can occur at runtime or after HDF5 loading.
    # Compare arrays element-wise because their equality result is not a single boolean.
    if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
        yield from _iter_array_differences(value1, value2, path, ignore_nan_inequality)
        return

    # Use Python equality before comparing nested values.
    try:
        if value1 == value2:
            return
    except ValueError:
        # Mixed array attributes also need explicit comparison.
        if not isinstance(value1, np.ndarray) and not isinstance(value2, np.ndarray):
            # Dictionaries and sequences can contain arrays that need recursive comparison.
            if not isinstance(value1, (dict, list, tuple)) or not isinstance(value2, (dict, list, tuple)):
                raise

    # Ignore independently created attribute NaNs only when requested.
    if (
        ignore_nan_inequality
        and isinstance(value1, (float, complex, np.inexact))
        and isinstance(value2, (float, complex, np.inexact))
    ):
        if np.isnan(value1) and np.isnan(value2):
            return

    if isinstance(value1, dict) and isinstance(value2, dict):
        # Compare dictionary structure (keys).
        keys1 = set(value1.keys())
        keys2 = set(value2.keys())

        for key in keys1 - keys2:
            yield f"Attribute missing in container2 at {path}[{key!r}]: {value1[key]!r}"
        for key in keys2 - keys1:
            yield f"Attribute missing in container1 at {path}[{key!r}]: {value2[key]!r}"

        # For common keys, compare values.
        for key in keys1.intersection(keys2):
            item1, item2 = value1[key], value2[key]
            yield from _iter_attribute_differences(item1, item2, f"{path}[{key!r}]", ignore_nan_inequality)

    elif (isinstance(value1, list) and isinstance(value2, list)) or (
        isinstance(value1, tuple) and isinstance(value2, tuple)
    ):
        # Compare common sequence elements by index.
        common_length = min(len(value1), len(value2))
        for index in range(common_length):
            item1, item2 = value1[index], value2[index]
            yield from _iter_attribute_differences(item1, item2, f"{path}[{index}]", ignore_nan_inequality)

        # Report elements beyond the common length.
        for index in range(common_length, len(value1)):
            yield f"Attribute missing in container2 at {path}[{index}]: {value1[index]!r}"
        for index in range(common_length, len(value2)):
            yield f"Attribute missing in container1 at {path}[{index}]: {value2[index]!r}"

    elif isinstance(value1, Unit) and isinstance(value2, Unit) and type(value1) is type(value2):
        # Compare the fields that define the unit.
        for field in fields(value1):
            unit_value1 = getattr(value1, field.name)
            unit_value2 = getattr(value2, field.name)
            yield from _iter_attribute_differences(
                unit_value1, unit_value2, f"{path}.{field.name}", ignore_nan_inequality
            )

    else:
        # Compare remaining attribute types and values.
        if type(value1) is not type(value2):
            yield (
                f"Attribute type/value mismatch at {path}: "
                f"{type(value1).__name__} {value1!r} vs {type(value2).__name__} {value2!r}"
            )
        else:
            yield f"Attribute value mismatch at {path}: {value1!r} vs {value2!r}"


def _iter_array_differences(
    array1: np.ndarray, array2: np.ndarray, path: str, ignore_nan_inequality: bool
) -> Iterator[str]:
    """Compare array attributes and report only differing elements."""
    # Compare array shapes.
    if array1.shape != array2.shape:
        yield f"Attribute shape mismatch at {path}: {array1.shape} vs {array2.shape}"
        return

    # Object arrays can contain nested attributes that need recursive comparison.
    if array1.dtype.hasobject or array2.dtype.hasobject:
        for index in np.ndindex(array1.shape):
            value1, value2 = array1[index], array2[index]
            index_path = "".join(f"[{i}]" for i in index)
            yield from _iter_attribute_differences(value1, value2, path + index_path, ignore_nan_inequality)
        return

    # Compare array values in bulk.
    try:
        equal_mask = np.equal(array1, array2)
    except (TypeError, ValueError):
        yield f"Attribute value mismatch at {path}: {array1!r} vs {array2!r}"
        return

    # Ignore matching NaNs in floating-point and complex arrays when requested.
    if ignore_nan_inequality:
        is_inexact1 = np.issubdtype(array1.dtype, np.inexact)
        is_inexact2 = np.issubdtype(array2.dtype, np.inexact)
        if is_inexact1 and is_inexact2:
            nan_mask = np.isnan(array1) & np.isnan(array2)
            equal_mask |= nan_mask

    # Report only the mismatching elements, without looping over the full array.
    mismatch_indices = np.flatnonzero(~equal_mask)
    for flat_index in mismatch_indices:
        index = np.unravel_index(flat_index, array1.shape)
        value1, value2 = array1[index], array2[index]
        index_path = "".join(f"[{i}]" for i in index)
        yield f"Attribute value mismatch at {path}{index_path}: {value1!r} vs {value2!r}"
