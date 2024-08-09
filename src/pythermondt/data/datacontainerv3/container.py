from typing import Dict, Tuple
from .node import Node, RootNode, GroupNode, DataNode, NodeType
from .utils import generate_key, split_path


class DataContainer:
    def __init__(self):
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

    def __str__(self):
        returnstring = ""
        for path, node in self.__nodes.items():
            returnstring = returnstring + f"{path}: ({node.type})" + "\n"

        return returnstring

    def __add_group(self, path: str, name):
        # Generate key
        key, parent, child = generate_key(path, name)

        if not self.__parent_exists(key):
            raise KeyError(f"Group at the path: '{parent}' does not exist. Cannot add group, without parent group.")

        if self.__group_exists(key):
            raise KeyError(f"Group with name: '{child}' at the path: '{parent}' already exists.")
        self.__nodes[key] = GroupNode(name)

    def __add_dataset(self, path: str, name):
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

    def __group_exists(self, key: str) -> bool:
        return key in self.__nodes
    
    def __parent_exists(self, key: str) -> bool:
        parent, _ = split_path(key)
        return parent in self.__nodes and (self.__nodes[parent].type == NodeType.GROUP or self.__nodes[parent].type == NodeType.ROOT)
    
    def __dataset_exists(self, key: str) -> bool:
        return key in self.__nodes and self.__nodes[key].type == NodeType.DATASET