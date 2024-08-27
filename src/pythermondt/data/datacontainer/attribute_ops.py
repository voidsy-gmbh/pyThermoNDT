from typing import Dict, ItemsView
from .base import BaseOps
from .node import DataNode, GroupNode, AttributeTypes

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
        
        self.nodes(path, DataNode, GroupNode).add_attribute(key, value)

    def add_attributes(self, path: str, **attributes: Dict[str, str | int | float | list | dict]):
        """Adds multiple attributes to the specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            **attributes (Dict[str, str | int | float | list | dict]): The attributes to add.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If any of the attributes already exists.
        """
        self.nodes(path, DataNode, GroupNode).add_attributes(**attributes)

    def get_attribute(self, path: str, key: str) -> str | int | float | list | dict:
        """Get a single attribute from a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            key (str): The key of the attribute.

        Returns:
            str | int | float | list | dict: The value of the attribute.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the attribute does not exist.
        """
        return self.nodes(path, DataNode, GroupNode).get_attribute(key)
    
    def get_attributes(self, path: str) -> ItemsView[str, str | int | float | list | dict]:
        """Get all attributes from a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.

        Returns:
            ItemsView[str, str | int | float | list | dict]: A view of all attributes in the group or dataset.

        Raises:
            KeyError: If the group or dataset does not exist.
        """
        return self.nodes(path, DataNode, GroupNode).attributes
    
    def remove_attribute(self, path: str, key: str):
        """Remove an attribute from a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            key (str): The key of the attribute.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the attribute does not exist.
        """
        self.nodes(path, DataNode, GroupNode).remove_attribute(key)

    def update_attribute(self, path: str, key: str, value: str | int | float | list | dict):
        """Update an attribute in a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            key (str): The key of the attribute.
            value (str | int | float | list | dict): The new value of the attribute.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the attribute does not exist.
        """
        self.nodes(path, DataNode, GroupNode).update_attribute(key, value)

    def update_attributes(self, path: str, **attributes: Dict[str, str | int | float | list | dict]):
        """Update multiple attributes in a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            **attributes (Dict[str, str | int | float | list | dict]): The new attributes.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If any of the attributes do not exist.
        """
        self.nodes(path, DataNode, GroupNode).update_attributes(**attributes)