from typing import List
from torch import Tensor
from .base import BaseOps
from .node import DataNode, NodeType
from .utils import generate_key, split_path

class DatasetOps(BaseOps):
    def add_dataset(self, path: str, name: str):
        """Adds a single dataset to a specified path in the DataContainer.

        Parameters:
            path (str): The path to the parent group.
            name (str): The name of the dataset to add.
        
        Raises:
            KeyError: If the parent group does not exist.
            KeyError: If the dataset already exists.
        """
        key, parent, child = generate_key(path, name)
      
        if not self._parent_exists(key):
            raise KeyError(f"Node at path: '{parent}' is not a group- or root node. Cannot add dataset, without parent group.")

        if self._dataset_exists(key):
            raise KeyError(f"Dataset with name: '{child}' at the path: '{parent}' already exists.")
        
        self._nodes[key] = DataNode(name)

    def get_datasets(self) -> List[str]:
        """Get a list of all datasets in the DataContainer.

        Returns:
            List[str]: A list of all datasets in the DataContainer.
        """
        return [node.name for node in self._nodes.values() if isinstance(node, DataNode)]
    
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
        parent, child = split_path(path)
      
        if not self._is_dataset(path):
            raise KeyError(f"Dataset with name: '{child}' at the path: '{parent}' does not exist.")
        
        return self._nodes[path].data
    
    def remove_dataset(self, path: str):
        """Removes a single dataset from a specified path in the DataContainer.

        Parameters:
            path (str): The path to the dataset
        
        Raises:
            KeyError: If the dataset does not exist.
        """
        parent, child = split_path(path)
      
        if not self._dataset_exists(path):
            raise KeyError(f"Dataset with name: '{child}' at the path: '{parent}' does not exist.")
        
        del self._nodes[path]