from .base_ops import BaseOps
from .node import DataNode
from .utils import generate_key

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