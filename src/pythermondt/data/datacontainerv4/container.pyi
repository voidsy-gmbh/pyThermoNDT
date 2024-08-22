from typing import Any, Dict
from pythermondt.data.datacontainerv4.node import Node

class DataContainer:
    def __init__(self) -> None:
        """Initializes the DataContainer with predefined groups and datasets.
        """

    def __str__(self):
        """Return str(self)."""

    def __getattr__(self, name):
        """None"""

    def add_dataset(self, path: str, name: str):
        """Adds a single dataset to a specified path in the DataContainer.

Parameters:
    path (str): The path to the parent group.
    name (str): The name of the dataset to add.

Raises:
    KeyError: If the parent group does not exist.
    KeyError: If the dataset already exists."""

    def add_group(self, path: str, name: str):
        """Adds a single group to a specified path in the DataContainer.

Parameters:
    path (str): The path to the parent group.
    name (str): The name of the group to add.

Raises:
    KeyError: If the parent group does not exist.
    KeyError: If the group already exists."""

    def add_attribute(self, path: str, key: str, value: str | int | float | list | dict):
        """Adds an attribute to the specified group or dataset in the DataContainer.

Parameters:
    path (str): The path to the group or dataset.
    key (str): The key of the attribute.
    value (str | int | float | list | dict): The value of the attribute.

Raises:
    KeyError: If the group or dataset does not exist.
    KeyError: If the attribute already exists."""

    def add_attributes(self, path: str, **attributes: Dict[str, str | int | float | list | dict]):
        """Adds multiple attributes to the specified group or dataset in the DataContainer.

Parameters:
    path (str): The path to the group or dataset.
    **attributes (Dict[str, str | int | float | list | dict]): The attributes to add.

Raises:
    KeyError: If the group or dataset does not exist.
    KeyError: If any of the attributes already exists."""

