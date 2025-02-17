
import torch
from numpy import ndarray
from torch import Tensor

from .base import BaseOps
from .node import DataNode
from .utils import generate_key


class DatasetOps(BaseOps):
    def add_dataset(self, path: str, name: str, data: Tensor | ndarray | None= None):
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

        # Convert the numpy array to a PyTorch tensor ==> internally, the data is stored as a PyTorch tensor
        if isinstance(data, ndarray):
            data = torch.from_numpy(data)

        if data is None:
            self.nodes[key] = DataNode(name)
        else:
            self.nodes[key] = DataNode(name, data)

    def add_datasets(self, path: str, **datasets: Tensor | ndarray | None):
        """Adds multiple datasets to a specified path in the DataContainer.

        Parameters:
            path (str): The path to the parent group.
            **datasets (Dict[str, Optional[Tensor | ndarray]]): The datasets to add, with the key being the name of the dataset.
        
        Raises:
            KeyError: If the parent group does not exist.
            KeyError: If any of the datasets already exist.
        """
        for name, data in datasets.items():
            self.add_dataset(path, name, data)

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

    def get_datasets(self, *paths: str) -> tuple[Tensor,...]:
        """Get multiple datasets from specified paths in the DataContainer.

        Parameters:
            *paths (str): Variable number of paths to the datasets. Can be provided as separate arguments or unpacked from a list.
        
        Returns:
            Tuple[Tensor, ...]: The tensors stored in the datasets, in the same order as the input paths.
        
        Raises:
            KeyError: If a dataset does not exist.
            KeyError: If the node is not a dataset.
        """
        return tuple(self.get_dataset(path) for path in paths)

    def get_all_dataset_names(self) -> list[str]:
        """Get a list of all dataset names in the DataContainer.

        Returns:
            List[str]: A list of names of all datasets in the DataContainer, without their full paths.
        """
        return [node.name for node in self.nodes.values() if isinstance(node, DataNode)]


    def remove_dataset(self, path: str):
        """Removes a single dataset from a specified path in the DataContainer.

        Parameters:
            path (str): The path to the dataset
        
        Raises:
            KeyError: If the dataset does not exist.
        """
        # Remove the dataset only if it is a DataNode
        if not self._is_datanode(path):
            raise KeyError(f"Dataset at path: '{path}' does not exist or is not a dataset.")
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

    def update_datasets(self, *updates: tuple[str, Tensor | ndarray]):
        """Update multiple datasets in the DataContainer.

        Parameters:
            *updates (Tuple[str, Tensor | ndarray]): Variable number of (path, data) tuples. Can be provided as separate tuples or unpacked from a list.
        
        Raises:
            KeyError: If a dataset does not exist.
        """
        for path, data in updates:
            self.update_dataset(path, data)
