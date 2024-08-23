from abc import ABC
from typing import Dict, TypeVar, overload, Optional, Type
from .utils import split_path
from .node import Node, NodeType, RootNode, GroupNode, DataNode

# Add typevars to make type hinting more readable
NodeTypes = RootNode | GroupNode | DataNode
T = TypeVar("T", RootNode, GroupNode, DataNode)

class DataContainerBase(ABC):
    def __init__(self):
        self.__nodes: Dict[str, NodeTypes] = {}

    @overload
    def __getnode(self, key: str) -> NodeTypes: ...

    @overload
    def __getnode(self, key: str, node_type: Type[T]) -> T: ...

    def __getnode(self, key: str, node_type: Optional[Type[T]] = None) -> NodeTypes | T:
        if key not in self.__nodes:
            parent, child = split_path(key)
            raise KeyError(f"Path '{key}' does not exist.")
        
        node = self.__nodes[key]
        if node_type is not None and not isinstance(node, node_type):
            raise TypeError(f"Node at path '{key}' is not of type {node_type.__name__}.")
        
        return node
    
    def __setnode(self, key: str, value: NodeTypes) -> None:       
        parent, _ = split_path(key)
        if parent != "/" and parent not in self.__nodes:
            raise KeyError(f"Parent path '{parent}' does not exist.")
        
        if parent != "/" and not isinstance(self.__nodes[parent], (RootNode, GroupNode)):
            raise TypeError(f"Parent node at '{parent}' must be a RootNode or GroupNode.")
        
        self.__nodes[key] = value
    
    @property
    def nodes(self):
        return self.__getnode
    
    @nodes.setter
    def nodes(self, value: tuple[str, NodeTypes]):
        if not hasattr(self, '_DataContainerBase__setnode'):
            raise AttributeError("Setting nodes directly is not allowed.")
        key, node = value
        self.__setnode(key, node)


class BaseOps(DataContainerBase):
    def _path_exists(self, key: str) -> bool:
        try:
            self.nodes(key)
            return True
        except KeyError:
            return False

    def _is_datanode(self, key: str) -> bool:
        try:
            self.nodes(key, DataNode)
            return True
        except (KeyError, TypeError):
            return False

    def _is_groupnode(self, key: str) -> bool:
        try:
            self.nodes(key, GroupNode)
            return True
        except (KeyError, TypeError):
            return False

    def _is_rootnode(self, key: str) -> bool:
        try:
            self.nodes(key, RootNode)
            return True
        except (KeyError, TypeError):
            return False

    def _parent_exists(self, key: str) -> bool:
        parent, _ = split_path(key)
        return self._is_groupnode(parent) or self._is_rootnode(parent)