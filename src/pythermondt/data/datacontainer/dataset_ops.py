import torch
from typing import List, Dict, Optional
from torch import Tensor
from numpy import ndarray
from .base import BaseOps
from .node import DataNode
from .utils import generate_key

class DatasetOps(BaseOps):
    def add_dataset(self, path: str, name: str, data: Optional[Tensor | ndarray]= None):
        """Adds a single dataset to a specified path in the DataContainer.

        Parameters:
            path (str): The path to the parent group.
            name (str): The name of the dataset to add.
            data (Tensor, optional): The data to store in the dataset. If None, an empty dataset is created. 
        
        Raises:
            KeyError: If the parent group does not exist.
            KeyError: If the dataset already exists.
        """
        key, parent, child = generate_key(path, name)

        if self._is_datanode(key):
            raise KeyError(f"Dataset with name: '{child}' at the path: '{parent}' already exists.")
        
        # Convert the numpy array to a PyTorch tensor ==> internally, the data is stored as a PyTorch tensor
        if isinstance(data, ndarray):
            data = torch.from_numpy(data)
        
        if data is None:
            self.nodes[key] = DataNode(name)
        else:
            self.nodes[key] = DataNode(name, data)

    def get_datasets(self) -> List[str]:
        """Get a list of all datasets in the DataContainer.

        Returns:
            List[str]: A list of all datasets in the DataContainer.
        """
        return [node.name for node in self.nodes.values() if isinstance(node, DataNode)]
    
    def get_dataset(self, path: str) -> Tensor:
        """Get a single dataset from a specified path in the DataContainer.

        Parameters:
            path (str): The path to the dataset
        
        Returns:
            Tensor: The data stored in the dataset.
        
        Raises:
            KeyError: If the dataset does not exist.
            KeyError: If the node is not a dataset.
        """       
        return self.nodes(path, DataNode).data
    
    def remove_dataset(self, path: str):
        """Removes a single dataset from a specified path in the DataContainer.

        Parameters:
            path (str): The path to the dataset
        
        Raises:
            KeyError: If the dataset does not exist.
        """
        del self.nodes[path]

    def update_dataset(self, path: str, data: Tensor | ndarray):
        """Updates a single dataset at a specified path in the DataContainer.

        Parameters:
            path (str): The path to the dataset.
            data (Tensor): The new data to store in the dataset.
        
        Raises:
            KeyError: If the dataset does not exist.
        """
        # Convert the numpy array to a PyTorch tensor ==> internally, the data is stored as a PyTorch tensor
        if isinstance(data, ndarray):
            data = torch.from_numpy(data)

        self.nodes(path, DataNode).data = data

    def update_datasets(self, updates: Dict[str, Tensor]):
        """Updates multiple datasets in the DataContainer.

        Parameters:
            updates (Dict[str, Tensor]): A dictionary of paths and new data to store in the datasets.
        
        Raises:
            KeyError: If a dataset does not exist.
        """
        for path, data in updates.items():
            self.update_dataset(path, data)