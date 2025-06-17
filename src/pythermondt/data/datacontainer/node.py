from abc import ABC, abstractmethod
from collections.abc import ItemsView
from enum import Enum
from sys import getsizeof

import objsize
import torch
from torch import Tensor

from ..units import Unit

AttributeTypes = str | int | float | list | tuple | dict | Unit


class NodeType(Enum):
    ROOT = "root"
    DATASET = "dataset"
    GROUP = "group"


class BaseNode(ABC):
    @abstractmethod
    def __init__(self, name: str, type: NodeType) -> None:
        self.__name: str = name
        self.__type: NodeType = type

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @property
    def type(self) -> NodeType:
        return self.__type

    def memory_bytes(self) -> int:
        """Returns the memory size of the node in bytes."""
        return getsizeof(self) + getsizeof(self.__name) + getsizeof(self.__type)


class RootNode(BaseNode):
    def __init__(self):
        super().__init__("root", NodeType.ROOT)


class AttributeNode(BaseNode, ABC):
    @abstractmethod
    def __init__(self, name: str, type: NodeType) -> None:
        super().__init__(name, type)
        self.__attributes: dict[str, AttributeTypes] = {}

    @property
    def attributes(self) -> ItemsView[str, AttributeTypes]:
        return self.__attributes.items()

    def add_attribute(self, key: str, value: AttributeTypes):
        if key in self.__attributes.keys():
            raise KeyError(
                f"Attribute with key '{key}' in node '{self.name}' already exists. "
                "Use 'update_attribute' to update the value."
            )
        self.__attributes[key] = value

    def add_attributes(self, **attributes: AttributeTypes):
        for key, value in attributes.items():
            self.add_attribute(key, value)

    def get_attribute(self, key: str) -> AttributeTypes:
        if key not in self.__attributes.keys():
            raise KeyError(f"Attribute with key '{key}' in node '{self.name}' does not exist.")
        return self.__attributes[key]

    def remove_attribute(self, key: str) -> None:
        if key not in self.__attributes.keys():
            raise KeyError(f"Attribute with key '{key}' in node '{self.name}' does not exist.")
        del self.__attributes[key]

    def update_attribute(self, key: str, value: AttributeTypes) -> None:
        if key not in self.__attributes.keys():
            raise KeyError(
                f"Attribute with key '{key}' in node '{self.name}' does not exist. "
                f"Use 'add_attribute' to add a new attribute."
            )

        if type(value) is not type(self.__attributes[key]):
            raise TypeError(
                f"Attribute with key '{key}' in node '{self.name}' is of type '{type(self.__attributes[key])}'. "
                f"Cannot update attribute with value of type '{type(value)}'."
            )
        self.__attributes[key] = value

    def update_attributes(self, **attributes: AttributeTypes):
        for update_key, update_value in attributes.items():
            self.update_attribute(update_key, update_value)

    def clear_attributes(self) -> None:
        self.__attributes.clear()

    def memory_bytes(self) -> int:
        """Returns the memory size of the node in bytes."""
        return super().memory_bytes() + objsize.get_deep_size(self.__attributes)


class GroupNode(AttributeNode):
    def __init__(self, name: str):
        super().__init__(name, NodeType.GROUP)


class DataNode(AttributeNode):
    def __init__(self, name: str, data: Tensor | None = None):
        super().__init__(name, NodeType.DATASET)
        self.__data: Tensor = data if data is not None else torch.empty(0)

    @property
    def data(self) -> Tensor:
        return self.__data

    @data.setter
    def data(self, value: Tensor) -> None:
        self.__data = value

    @data.deleter
    def data(self) -> None:
        self.__data = torch.empty(0)

    def memory_bytes(self) -> int:
        """Returns the memory size of the node in bytes."""
        return super().memory_bytes() + getsizeof(self.__data) + self.data.element_size() * self.data.numel()
