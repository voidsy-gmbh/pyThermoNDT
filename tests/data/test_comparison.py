import pickle
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import Tensor

from pythermondt.data import DataContainer, ThermoContainer, container_diff, containers_equal
from pythermondt.data.units import celsius, kelvin


@pytest.fixture
def container_pair(sample_tensor: Tensor) -> tuple[DataContainer, DataContainer]:
    containers = DataContainer(), DataContainer()
    for container in containers:
        container.add_group("/", "MetaData")
        container.add_dataset("/", "Values", sample_tensor.clone())
    return containers


@pytest.mark.parametrize("ignore_attribute_nan_inequality", [False, True])
@pytest.mark.parametrize(
    "data1, data2",
    [
        (torch.tensor([1, 2]), torch.tensor([1, 2])),
        (torch.tensor([1, 2]), torch.tensor([1.0, 2.0])),
        (torch.tensor([1, 2]), torch.tensor([1, 3])),
        (torch.tensor([1, 2]), torch.tensor([[1, 2]])),
        (torch.tensor([1.0, float("nan")]), torch.tensor([1.0, float("nan")])),
    ],
    ids=["equal", "different-dtypes", "different-values", "different-shapes", "dataset-nans"],
)
def test_datasets_follow_torch_equal(
    container_pair: tuple[DataContainer, DataContainer],
    data1: Tensor,
    data2: Tensor,
    ignore_attribute_nan_inequality: bool,
):
    """Keep torch.equal behavior even when attribute NaN inequality is ignored."""
    left, right = container_pair
    left.update_dataset("/Values", data1)
    right.update_dataset("/Values", data2)
    expected = torch.equal(data1, data2)

    assert containers_equal(left, right, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality) is expected
    differences = list(container_diff(left, right, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality))
    assert (differences == []) is expected
    assert (left == right) is expected


@pytest.mark.parametrize(
    "value1, value2, expected",
    [
        (1, 1.0, True),
        (True, 1, True),
        (np.array(1), 1.0, True),
        ([1, {"value": 2}], [1.0, {"value": 2.0}], True),
        ((1, 2), (1.0, 2.0), True),
        ([1, 2], (1, 2), False),
        ([1, 2], [2, 1], False),
        ({"a": 1, "b": 2}, {"b": 2.0, "a": 1.0}, True),
        (kelvin, kelvin, True),
        (kelvin, celsius, False),
    ],
)
def test_attributes_keep_python_equality(
    container_pair: tuple[DataContainer, DataContainer], value1: Any, value2: Any, expected: bool
):
    """Preserve numeric equivalence, sequence types/order, and Unit equality."""
    left, right = container_pair
    left.add_attribute("/MetaData", "value", value1)
    right.add_attribute("/MetaData", "value", value2)

    assert containers_equal(left, right) is expected
    assert (list(container_diff(left, right)) == []) is expected
    assert (left == right) is expected


@pytest.mark.parametrize("path", ["/MetaData", "/Values"])
@pytest.mark.parametrize("ignore_attribute_nan_inequality", [False, True])
def test_attribute_nan_inequality_after_pickling(
    container_pair: tuple[DataContainer, DataContainer], path: str, ignore_attribute_nan_inequality: bool
):
    """Ignore independent attribute NaNs only when requested, on groups and datasets."""
    left, right = container_pair
    metadata = {"values": [float("nan"), (np.float32("nan"),)]}
    left.add_attribute(path, "metadata", metadata)
    # Pickling creates independent NaN objects, as lazy caching does.
    right.add_attribute(path, "metadata", pickle.loads(pickle.dumps(metadata)))

    equal = containers_equal(left, right, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality)
    differences = list(container_diff(left, right, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality))

    assert equal is ignore_attribute_nan_inequality
    assert (differences == []) is ignore_attribute_nan_inequality
    assert left != right
    if not ignore_attribute_nan_inequality:
        assert len(differences) == 2
        assert any(f"{path}['metadata']['values'][0]" in difference for difference in differences)
        assert any(f"{path}['metadata']['values'][1][0]" in difference for difference in differences)


def test_shared_attribute_nan_identity_is_preserved(container_pair: tuple[DataContainer, DataContainer]):
    """Retain Python's identity shortcut instead of rejecting shared attribute NaNs."""
    left, right = container_pair
    shared_nan = float("nan")
    left.add_attribute("/MetaData", "values", [shared_nan])
    right.add_attribute("/MetaData", "values", [shared_nan])

    assert containers_equal(left, right)
    assert list(container_diff(left, right)) == []
    assert left == right


@pytest.mark.parametrize("ignore_attribute_nan_inequality", [False, True])
@pytest.mark.parametrize(
    "value1, value2, expected",
    [
        (np.array([[1, 2]], dtype=np.int16), np.array([[1, 2]], dtype=np.float64), True),
        (np.array([]), np.array([], dtype=np.int32), True),
        (np.array(1), np.array(1.0), True),
        (np.array(1), np.array(2), False),
        (np.array([1, 2]), np.array([[1, 2]]), False),
        (np.array(["hot", "cold"]), np.array(["hot", "cold"]), True),
        (np.array([1, 2]), np.array(["1", "2"]), False),
        (np.array([{"a": 1}], dtype=object), np.array([{"a": 1.0}], dtype=object), True),
    ],
    ids=["different-dtypes", "empty", "scalar", "different-scalars", "shape", "strings", "incompatible", "objects"],
)
def test_numpy_array_attributes(
    container_pair: tuple[DataContainer, DataContainer],
    value1: np.ndarray,
    value2: np.ndarray,
    expected: bool,
    ignore_attribute_nan_inequality: bool,
):
    """Support array attributes without ambiguous truth tests or dtype restrictions."""
    left, right = container_pair
    # Exercise array handling inside nested metadata.
    left.add_attribute("/MetaData", "values", {"nested": [value1]})
    right.add_attribute("/MetaData", "values", {"nested": [value2]})

    assert containers_equal(left, right, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality) is expected
    differences = list(container_diff(left, right, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality))
    assert (differences == []) is expected
    assert (left == right) is expected


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize("reverse", [False, True], ids=["array-first", "array-second"])
@pytest.mark.parametrize(
    "value1, value2",
    [
        (np.array([1, 2]), [1, 2]),
        (np.array([1, 2]), (1, 2)),
        (np.array([1, 2]), 1),
    ],
    ids=["list", "tuple", "scalar"],
)
def test_mixed_numpy_attribute_types(
    container_pair: tuple[DataContainer, DataContainer], value1: Any, value2: Any, reverse: bool, nested: bool
):
    """Report mixed array attributes as mismatches without ambiguous truth tests."""
    left, right = container_pair
    if reverse:
        value1, value2 = value2, value1
    if nested:
        value1, value2 = {"nested": [value1]}, {"nested": [value2]}
    left.add_attribute("/MetaData", "value", value1)
    right.add_attribute("/MetaData", "value", value2)

    differences = list(container_diff(left, right))
    attribute_path = "/MetaData['value']['nested'][0]" if nested else "/MetaData['value']"
    assert len(differences) == 1
    assert differences[0].startswith(f"Attribute type/value mismatch at {attribute_path}: ")
    assert not containers_equal(left, right)
    assert left != right


@pytest.mark.parametrize("ignore_attribute_nan_inequality", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128, object])
def test_numpy_array_attribute_nans(
    container_pair: tuple[DataContainer, DataContainer], dtype: Any, ignore_attribute_nan_inequality: bool
):
    """Apply the attribute option to both numeric and object-array NaNs."""
    left, right = container_pair
    left.add_attribute("/MetaData", "values", np.array([1.0, float("nan")], dtype=dtype))  # type: ignore[arg-type]
    right.add_attribute("/MetaData", "values", np.array([1.0, float("nan")], dtype=dtype))  # type: ignore[arg-type]

    equal = containers_equal(left, right, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality)
    differences = list(container_diff(left, right, ignore_attribute_nan_inequality=ignore_attribute_nan_inequality))

    assert equal is ignore_attribute_nan_inequality
    assert (differences == []) is ignore_attribute_nan_inequality
    assert left != right


def test_numpy_attribute_hdf5_round_trip(container_pair: tuple[DataContainer, DataContainer]):
    """Fix container equality for array attributes that already survive HDF5 serialization."""
    left, _ = container_pair
    left.add_attribute("/MetaData", "values", np.array([1, 2]))  # type: ignore[arg-type]
    restored = DataContainer(left.serialize_to_hdf5())

    assert containers_equal(left, restored)
    assert list(container_diff(left, restored)) == []
    assert left == restored


def test_report_details_and_silent_output(container_pair: tuple[DataContainer, DataContainer], capsys):
    """Report nested locations, missing values, types, units, and data without printing."""
    left, right = container_pair
    left.add_attribute("/MetaData", "config", {"values": [1, 2], "removed": "cold", "unit": kelvin, "kind": "1"})
    right.add_attribute("/MetaData", "config", {"values": [1, 3, 4], "added": "hot", "unit": celsius, "kind": 1})
    right.update_dataset("/Values", torch.zeros_like(right.get_dataset("/Values")))

    # Check report locations and values without relying on iteration order.
    differences = list(container_diff(left, right))
    report = "\n".join(differences)
    assert "/MetaData['config']['values'][1]: 2 vs 3" in report
    assert "Attribute missing in container1 at /MetaData['config']['values'][2]: 4" in report
    assert "Attribute missing in container2 at /MetaData['config']['removed']: 'cold'" in report
    assert "Attribute missing in container1 at /MetaData['config']['added']: 'hot'" in report
    assert "/MetaData['config']['unit'].name: 'kelvin' vs 'celsius'" in report
    assert "/MetaData['config']['unit'].symbol: 'K' vs '°C'" in report
    assert "/MetaData['config']['kind']: str '1' vs int 1" in report
    assert "Data content mismatch at path /Values" in differences
    assert not containers_equal(left, right)
    assert capsys.readouterr().out == ""


def test_numpy_report_element_locations(container_pair: tuple[DataContainer, DataContainer]):
    """Keep detailed array-attribute reports while leaving tensor data summarized."""
    left, right = container_pair
    left.add_attribute("/MetaData", "values", np.array([[1, 2], [3, 4]]))  # type: ignore[arg-type]
    right.add_attribute("/MetaData", "values", np.array([[1, 5], [6, 4]]))  # type: ignore[arg-type]

    differences = list(container_diff(left, right))
    assert len(differences) == 2
    assert "/MetaData['values'][0][1]" in differences[0]
    assert "/MetaData['values'][1][0]" in differences[1]
    assert "2" in differences[0] and "5" in differences[0]
    assert "3" in differences[1] and "6" in differences[1]


def test_missing_paths_in_both_containers(container_pair: tuple[DataContainer, DataContainer]):
    """Report paths missing from each container."""
    left, right = container_pair
    left.add_group("/", "LeftOnly")
    left.add_group("/", "AnotherLeftOnly")
    right.add_group("/", "RightOnly")

    differences = list(container_diff(left, right))
    assert len(differences) == 2
    assert differences[0].startswith("Paths in container1 but not in container2: ")
    assert "/LeftOnly" in differences[0] and "/AnotherLeftOnly" in differences[0]
    assert differences[1] == "Paths in container2 but not in container1: {'/RightOnly'}"
    assert not containers_equal(left, right)
    assert left != right


@pytest.mark.parametrize("mismatch", ["name", "type", "shape"])
def test_node_and_shape_reports(container_pair: tuple[DataContainer, DataContainer], mismatch: str):
    """Report node name/type mismatches and dataset shape mismatches."""
    left, right = container_pair
    if mismatch == "name":
        right.nodes["/Values"].name = "Renamed"
        expected = "Node name mismatch at path /Values: Values vs Renamed"
    elif mismatch == "type":
        right.remove_dataset("/Values")
        right.add_group("/", "Values")
        expected = "Node type mismatch at path /Values: NodeType.DATASET vs NodeType.GROUP"
    else:
        right.update_dataset("/Values", torch.ones(3))
        expected = "Data shape mismatch at path /Values: torch.Size([2, 2]) vs torch.Size([3])"

    assert list(container_diff(left, right)) == [expected]
    assert not containers_equal(left, right)
    assert left != right


def test_boolean_comparison_stops_early(container_pair: tuple[DataContainer, DataContainer], monkeypatch):
    """Avoid subsequent tensor comparisons when a boolean result is already known."""
    left, right = container_pair
    # Give both nodes an attribute mismatch so path order cannot trigger a tensor comparison first.
    for path in ("/MetaData", "/Values"):
        left.add_attribute(path, "value", 1)
        right.add_attribute(path, "value", 2)
    right.update_dataset("/Values", torch.zeros_like(right.get_dataset("/Values")))
    tensor_equal = Mock(wraps=torch.equal)
    monkeypatch.setattr(torch, "equal", tensor_equal)

    assert not containers_equal(left, right)
    assert left != right
    tensor_equal.assert_not_called()

    # A full report must include the dataset mismatch as well.
    assert len(list(container_diff(left, right))) == 3
    tensor_equal.assert_called_once()


def test_container_diff_streams_differences(container_pair: tuple[DataContainer, DataContainer], monkeypatch):
    """Yield structural differences before comparing datasets, then resume the same iterator."""
    left, right = container_pair
    left.add_group("/", "LeftOnly")
    right.update_dataset("/Values", torch.zeros_like(right.get_dataset("/Values")))
    tensor_equal = Mock(wraps=torch.equal)
    monkeypatch.setattr(torch, "equal", tensor_equal)

    differences = container_diff(left, right)
    tensor_equal.assert_not_called()

    assert next(differences) == "Paths in container1 but not in container2: {'/LeftOnly'}"
    tensor_equal.assert_not_called()

    # Resume the same iterator, then check that it is exhausted.
    assert list(differences) == ["Data content mismatch at path /Values"]
    tensor_equal.assert_called_once()
    assert list(differences) == []


@pytest.mark.parametrize("compare", [containers_equal, container_diff])
@pytest.mark.parametrize("position", [0, 1])
def test_invalid_arguments(
    container_pair: tuple[DataContainer, DataContainer], compare: Callable[..., bool | Iterator[str]], position: int
):
    """Reject non-container arguments in either position."""
    arguments: list[Any] = list(container_pair)
    arguments[position] = "not-a-container"

    with pytest.raises(TypeError, match=f"container{position + 1} must be a DataContainer, got str"):
        result = compare(*arguments)
        if isinstance(result, Iterator):
            # Generator input validation runs when iteration starts.
            next(result)


def test_container_subclasses_are_supported():
    """Compare DataContainer subclasses using the same equality rules."""
    thermal = ThermoContainer()
    plain = DataContainer(thermal.serialize_to_hdf5())

    assert containers_equal(thermal, plain)
    assert list(container_diff(thermal, plain)) == []
    assert thermal == plain
