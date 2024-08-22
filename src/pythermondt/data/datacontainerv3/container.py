from typing import Dict, Tuple
from .node import Node, RootNode, GroupNode, DataNode, NodeType
from .utils import generate_key, split_path


class DataContainer:
    """
    Manages and serializes data into HDF5 format.

    This class provides structured handling of groups and datasets read with the reader classes. It allows for easy access to the data and attributes stored in the DataContainer.
    It also provieds functions for easy serialization and data visualization.
    """
    def __init__(self):
        """ Initializes the DataContainer with predefined groups and datasets.
        """
        self.__nodes: Dict[str, Node] = {}

        # Set root node
        self.__nodes["/"] = RootNode()

        # Set initial groups
        self.__add_group("/", "Data")
        self.__add_group("/", "GroundTruth")
        self.__add_group("/", "MetaData")

        # Set initial datasets
        self.__add_dataset("/Data", "Tdata")
        self.__add_dataset("/GroundTruth", "DefectMask")
        self.__add_dataset("/MetaData", "LookUpTable")
        self.__add_dataset("/MetaData", "ExcitationSignal")
        self.__add_dataset("/MetaData", "DomainValues")

    # Overwrite the __str__ method to provide a string representation of the DataContainer
    def __str__(self):
        returnstring = ""
        for path, node in self.__nodes.items():
            returnstring = returnstring + f"{path}: ({node.name}: {node.type})" + "\n"

        return returnstring
    
    # Functions to check existence of various datacontainer elements
    def __group_exists(self, key: str) -> bool:
        return key in self.__nodes
    
    def __parent_exists(self, key: str) -> bool:
        parent, _ = split_path(key)
        return parent in self.__nodes and (self.__nodes[parent].type == NodeType.GROUP or self.__nodes[parent].type == NodeType.ROOT)
    
    def __dataset_exists(self, key: str) -> bool:
        return key in self.__nodes and self.__nodes[key].type == NodeType.DATASET

    def __add_group(self, path: str, name: str):
        """Adds a single group names to a specified path in the DataContainer.

        Parameters:
            path (str): The path to the parent group.
            name (str): The name of the group to add.

        Raises:
            KeyError: If the parent group does not exist.
            KeyError: If the group already exists.
        """
        # Generate key
        key, parent, child = generate_key(path, name)

        if not self.__parent_exists(key):
            raise KeyError(f"Group at the path: '{parent}' does not exist. Cannot add group, without parent group.")

        if self.__group_exists(key):
            raise KeyError(f"Group with name: '{child}' at the path: '{parent}' already exists.")
        self.__nodes[key] = GroupNode(name)

    def __add_dataset(self, path: str, name):
        """Adds a single dataset to a specified path in the DataContainer.

        Parameters:
            path (str): The path to the parent group.
            name (str): The name of the dataset to add.
        
        Raises:
            KeyError: If the parent group does not exist.
            KeyError: If the dataset already exists.
        """
        # Generate key
        key, parent, child = generate_key(path, name)
      
        # Check if the parent node is a group or root node
        if not self.__parent_exists(key):
            raise KeyError(f"Node at path: '{parent}' is not a group- or root node. Cannot add dataset, without parent group.")

        # Check if the dataset exists
        if self.__dataset_exists(key):
            raise KeyError(f"Dataset with name: '{child}' at the path: '{parent}' already exists.")
        
        # Add the dataset
        self.__nodes[key] = DataNode(name)

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
        # Check if the group or dataset exists
        if not self.__group_exists(path) and not self.__dataset_exists(path):
            raise KeyError(f"Group or dataset at path: '{path}' does not exist. Cannot add attribute to non-existing node.")
        
        # Add the attribute
        self.__nodes[path].add_attribute(key, value)

    def add_attributes(self, path: str, **attributes: Dict[str, str | int | float | list | dict]):
        """Adds multiple attributes to the specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            **attributes (Dict[str, str | int | float | list | dict]): The attributes to add.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If any of the attributes already exists.
        """
        # Check if the group or dataset exists
        if not self.__group_exists(path) and not self.__dataset_exists(path):
            raise KeyError(f"Group or dataset at path: '{path}' does not exist. Cannot add attributes to non-existing node.")
        
        # Add the attributes
        self.__nodes[path].add_attributes(**attributes)