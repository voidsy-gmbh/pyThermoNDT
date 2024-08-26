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

        if self._is_datanode(key):
            raise KeyError(f"Dataset with name: '{child}' at the path: '{parent}' already exists.")
        
        self.nodes[key] = DataNode(name)

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