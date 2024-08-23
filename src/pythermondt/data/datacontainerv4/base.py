from abc import ABC
from typing import Dict
from .utils import generate_key, split_path
from .node import Node, NodeType, RootNode

class DataContainerBase(ABC):
    def __init__(self):
        self._nodes: Dict[str, Node] = {}

class BaseOps(DataContainerBase):
    def _group_exists(self, key: str) -> bool:
        return key in self._nodes

    def _parent_exists(self, key: str) -> bool:
        parent, _ = split_path(key)
        return parent in self._nodes and (self._nodes[parent].type == NodeType.GROUP or self._nodes[parent].type == NodeType.ROOT)

    def _dataset_exists(self, key: str) -> bool:
        return key in self._nodes and self._nodes[key].type == NodeType.DATASET