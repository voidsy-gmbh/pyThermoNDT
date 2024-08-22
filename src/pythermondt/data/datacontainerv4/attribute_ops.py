from .base_ops import BaseOps
from typing import Dict

class AttributeOps(BaseOps):
    def add_attribute(self, path: str, key: str, value: str | int | float | list | dict):
        """Adds an attribute to the specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            key (str): The key of the attribute.
            value (str | int | float | list | dict): The value of the attribute.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the attribute already exists.
        """
        if not self._group_exists(path) and not self._dataset_exists(path):
            raise KeyError(f"Group or dataset at path: '{path}' does not exist. Cannot add attribute to non-existing node.")
        
        self._nodes[path].add_attribute(key, value)

    def add_attributes(self, path: str, **attributes: Dict[str, str | int | float | list | dict]):
        """Adds multiple attributes to the specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            **attributes (Dict[str, str | int | float | list | dict]): The attributes to add.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If any of the attributes already exists.
        """
        if not self._group_exists(path) and not self._dataset_exists(path):
            raise KeyError(f"Group or dataset at path: '{path}' does not exist. Cannot add attributes to non-existing node.")
        
        self._nodes[path].add_attributes(**attributes)