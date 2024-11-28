import pytest
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from typing import Optional
from pythermondt.data import DataContainer
from pythermondt.data.datacontainer.node import DataNode

@pytest.fixture
def dataset_container(empty_container:DataContainer, sample_empty_tensor:Tensor):
    empty_container.add_group("/", "testgroup")
    empty_container.add_group("/testgroup", "nestedgroup")
    return empty_container



@pytest.mark.parametrize("path, name, data", [
    ("/", "dataset0", None), # add directly to root
    ("/testgroup", "dataset1", None), # add to a group
    ("/testgroup/nestedgroup", "dataset2", None), # add to an existing dataset
])
def test_add_dataset(dataset_container:DataContainer, data:Optional[Tensor | ndarray], path:str, name:str):
    # Add a dataset
    key = f"{path}/{name}" if path != "/" else f"{path}{name}"
    dataset_container.add_dataset(path, name, data)

    # Assertions
    # Empty dataset
    if data is None:
        assert key in dataset_container.nodes.keys()
        assert isinstance(dataset_container.nodes[key], DataNode)
        assert torch.equal(dataset_container.nodes[key].data, torch.empty(0)) #type: ignore
    
    # Ndarray
    elif isinstance(data, ndarray):
        assert key in dataset_container.nodes.keys()
        assert isinstance(dataset_container.nodes[key], DataNode)
        assert torch.equal(dataset_container.nodes[key].data, torch.from_numpy(data)) #type: ignore
    
    # Tensor
    elif isinstance(data, Tensor):
        assert key in dataset_container.nodes.keys()
        assert isinstance(dataset_container.nodes[key], DataNode)
        assert torch.equal(dataset_container.nodes[key].data, data) # type:ignore

    # Error case
    else:
        assert False

def test_add_dataset_with_ndarray(dataset_ops):
    path = "/group"
    name = "dataset2"
    data = np.array([1, 2, 3])
    
    dataset_ops.add_dataset(path, name, data)
    
    key = f"{path}/{name}"
    assert key in dataset_ops.nodes
    assert isinstance(dataset_ops.nodes[key], DataNode)
    assert torch.equal(dataset_ops.nodes[key].data, torch.from_numpy(data))

def test_add_dataset_without_data(dataset_ops):
    path = "/group"
    name = "dataset3"
    
    dataset_ops.add_dataset(path, name)
    
    key = f"{path}/{name}"
    assert key in dataset_ops.nodes
    assert isinstance(dataset_ops.nodes[key], DataNode)
    assert dataset_ops.nodes[key].data is None

def test_add_dataset_already_exists(dataset_ops):
    path = "/group"
    name = "dataset4"
    data = torch.tensor([1, 2, 3])
    
    dataset_ops.add_dataset(path, name, data)
    
    with pytest.raises(KeyError, match=f"Dataset with name: '{name}' at the path: '{path}' already exists."):
        dataset_ops.add_dataset(path, name, data)

# Only run the tests in this file if it is run directly
if __name__ == '__main__':
    pytest.main(["-v", __file__])