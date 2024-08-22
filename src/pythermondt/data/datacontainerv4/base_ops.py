# base_ops.py
from typing import Dict
from .node import Node, NodeType
from .utils import generate_key, split_path

class BaseOps:
    def __init__(self, nodes: Dict[str, Node]):
        self._nodes = nodes

    def _group_exists(self, key: str) -> bool:
        return key in self._nodes

    def _parent_exists(self, key: str) -> bool:
        parent, _ = split_path(key)
        return parent in self._nodes and (self._nodes[parent].type == NodeType.GROUP or self._nodes[parent].type == NodeType.ROOT)

    def _dataset_exists(self, key: str) -> bool:
        return key in self._nodes and self._nodes[key].type == NodeType.DATASET
