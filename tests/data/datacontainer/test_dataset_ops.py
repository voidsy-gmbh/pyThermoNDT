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

# Test adding multiple datasets to the container
@pytest.mark.parametrize("datasets", [
    pytest.param({"dataset0": "sample_tensor", "dataset1": "sample_ndarray", "dataset2": "sample_empty_tensor"}),
    pytest.param({"dataset0": "sample_empty_ndarray", "dataset1": "sample_empty_tensor", "dataset2": None}),
    pytest.param({"dataset0": None, "dataset1": "sample_ndarray", "dataset2": "sample_empty_tensor"}),
    pytest.param({"dataset0": "sample_tensor", "dataset1": None, "dataset2": "sample_empty_ndarray"}),
])
@pytest.mark.parametrize("path", [
    ("/"), # add directly to root
    ("/testgroup"), # add to a group
    ("/testgroup/nestedgroup"), # add to a nested group
])
def test_add_datasets(dataset_container:DataContainer, datasets:dict[str, str | None], path:str, request:pytest.FixtureRequest):
    # Request testdata from the fixtures
    test_data = {key: request.getfixturevalue(value) if value is not None else None for key, value in datasets.items()}

    # Add multiple datasets
    dataset_container.add_datasets(path, **test_data)

    # Assertions
    for name, data in test_data.items():
        key = f"{path}/{name}" if path != "/" else f"{path}{name}"
        assert key in dataset_container.nodes.keys()
        assert isinstance(dataset_container.nodes[key], DataNode)

        # Empty dataset ==> default value in container is a torch.empty(0) tensor
        if data is None:
            assert torch.equal(dataset_container.nodes[key].data, torch.empty(0)) #type: ignore
        
        # Ndarray
        elif isinstance(data, ndarray):
            assert torch.equal(dataset_container.nodes[key].data, torch.from_numpy(data)) #type: ignore

        # Tensor
        elif isinstance(data, Tensor):
            assert torch.equal(dataset_container.nodes[key].data, data) # type:ignore
        
        # Error case
        else:
            assert False

# Test getting a dataset from the container
@pytest.mark.parametrize("data", [
    pytest.param("sample_tensor"),
    pytest.param("sample_ndarray"),
    pytest.param("sample_empty_tensor"),
    pytest.param("sample_empty_ndarray"),
])
@pytest.mark.parametrize("path, name", [
    ("/", "dataset0"), # get directly from root
    ("/testgroup", "dataset1"), # get from a group
    ("/testgroup/nestedgroup", "dataset2"), # get from a nested group
])
def test_get_dataset(dataset_container:DataContainer, data:str, path:str, name:str, request:pytest.FixtureRequest):
    # Request testdata from the fixtures
    test_data = request.getfixturevalue(data)

    # Add a dataset
    key = f"{path}/{name}" if path != "/" else f"{path}{name}"
    dataset_container.add_dataset(path, name, test_data)

    # Get the dataset
    retrieved_data = dataset_container.get_dataset(key)

    # Assertions
    # Empty dataset ==> default value in container is a torch.empty(0) tensor
    if test_data is None:
        assert torch.equal(retrieved_data, torch.empty(0)) #type: ignore
    
    # Ndarray
    elif isinstance(test_data, ndarray):
        assert torch.equal(retrieved_data, torch.from_numpy(test_data)) #type: ignore
    
    # Tensor
    elif isinstance(test_data, Tensor):
        assert torch.equal(retrieved_data, test_data) # type:ignore
    
    # Error case
    else:
        assert False

# Only run the tests in this file if it is run directly
if __name__ == '__main__':
    pytest.main(["-v", __file__])