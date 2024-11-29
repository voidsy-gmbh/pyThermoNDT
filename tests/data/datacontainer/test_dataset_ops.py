import pytest
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from typing import Optional
from pythermondt.data import DataContainer
from pythermondt.data.datacontainer.node import DataNode

@pytest.fixture
def dataset_container(empty_container:DataContainer):
    empty_container.add_group("/", "testgroup")
    empty_container.add_group("/testgroup", "nestedgroup")
    return empty_container

# Test adding a dataset to the container
@pytest.mark.parametrize("data", [
    pytest.param("sample_tensor"),
    pytest.param("sample_ndarray"),
    pytest.param("sample_empty_tensor"),
    pytest.param("sample_empty_ndarray"),
    pytest.param(None)
])
@pytest.mark.parametrize("path, name", [
    ("/", "dataset0"), # add directly to root
    ("/testgroup", "dataset1"), # add to a group
    ("/testgroup/nestedgroup", "dataset2"), # add to a nested group
])
def test_add_dataset(dataset_container:DataContainer, data:str | None, path:str, name:str, request:pytest.FixtureRequest):
    # Request testdata from the fixtures
    test_data = request.getfixturevalue(data) if data is not None else None

    # Add a dataset
    key = f"{path}/{name}" if path != "/" else f"{path}{name}"
    dataset_container.add_dataset(path, name, test_data)

    # Assertions
    assert key in dataset_container.nodes.keys()
    assert isinstance(dataset_container.nodes[key], DataNode)

    # Assertions
    # Empty dataset ==> default value in container is a torch.empty(0) tensor
    if test_data is None:
        assert torch.equal(dataset_container.nodes[key].data, torch.empty(0)) #type: ignore
    
    # Ndarray
    elif isinstance(test_data, ndarray):
        assert torch.equal(dataset_container.nodes[key].data, torch.from_numpy(test_data)) #type: ignore
    
    # Tensor
    elif isinstance(test_data, Tensor):
        assert torch.equal(dataset_container.nodes[key].data, test_data) # type:ignore

    # Error case
    else:
        assert False

# Only run the tests in this file if it is run directly
if __name__ == '__main__':
    pytest.main(["-v", __file__])