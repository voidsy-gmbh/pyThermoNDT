from collections.abc import Iterator

from .datacontainer import DataContainer
from .datacontainer._comparison import _iter_node_differences


def containers_equal(
    container1: DataContainer,
    container2: DataContainer,
    *,
    ignore_attribute_nan_inequality: bool = False,
) -> bool:
    """Compare two DataContainers for equality.

    Containers are equal if they have:
    1. The same node paths, names, and types
    2. Equal shapes and values in all datasets
    3. Equal attributes for every group and dataset

    Stops at the first difference.
    Attributes use Python equality, including numeric equivalence and shared-object identity.
    NumPy-array attributes compare by shape and value without a dtype restriction.

    Datasets always use ``torch.equal``: shapes and values must match, but dtypes need not match.
    Dataset NaNs remain unequal, even when ``ignore_attribute_nan_inequality=True``.

    **Note**: Dataset values are compared without numerical tolerance. Containers with the same initial data
    can differ after stochastic transforms (e.g. ``GaussianNoise``) if those transforms produce different values.

    Args:
        container1 (DataContainer): First container.
        container2 (DataContainer): Second container.
        ignore_attribute_nan_inequality (bool, optional): If True, treat independently created NaNs in attributes
            as equal, including nested values and NumPy arrays. Does not affect dataset tensors. Default is False.

    Returns:
        bool: True if the containers are equal, False otherwise.

    Raises:
        TypeError: Either argument is not a DataContainer.
    """
    diffs = container_diff(container1, container2, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality)
    return next(diffs, None) is None


def container_diff(
    container1: DataContainer,
    container2: DataContainer,
    *,
    ignore_attribute_nan_inequality: bool = False,
) -> Iterator[str]:
    """Return the differences between two DataContainers.

    Uses the same comparison rules as ``containers_equal``. Attribute reports include nested keys, indices,
    and differing values. Dataset reports include shape/content mismatches, not individual tensor elements.

    Datasets always use ``torch.equal``. Dataset NaNs remain unequal, even when
    ``ignore_attribute_nan_inequality=True``.

    Returns a lazy, single-pass iterator. Use ``list(...)`` or ``sorted(...)`` to collect the differences.

    Args:
        container1 (DataContainer): First container.
        container2 (DataContainer): Second container.
        ignore_attribute_nan_inequality (bool, optional): If True, treat independently created NaNs in attributes
            as equal, including nested values and NumPy arrays. Does not affect dataset tensors. Default is False.

    Yields:
        str: Differences in unspecified order. No differences means the containers are equal.

    Raises:
        TypeError: Either argument is not a DataContainer. Raised when iteration begins.
    """
    if not isinstance(container1, DataContainer):
        raise TypeError(f"container1 must be a DataContainer, got {type(container1).__name__}.")
    if not isinstance(container2, DataContainer):
        raise TypeError(f"container2 must be a DataContainer, got {type(container2).__name__}.")

    # Compare the nodes of both containers
    nodes1, nodes2 = container1.nodes, container2.nodes
    yield from _iter_node_differences(nodes1, nodes2, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality)
