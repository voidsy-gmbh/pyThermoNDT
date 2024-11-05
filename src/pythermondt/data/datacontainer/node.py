import torch
from typing import Dict, ItemsView
from enum import Enum
from torch import Tensor
from abc import ABC, abstractmethod
from ..units import UnitInfo

AttributeTypes = str | int | float | list | tuple | dict | UnitInfo

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
    
class RootNode(BaseNode):
    def __init__(self):
        super().__init__("root", NodeType.ROOT)

class AttributeNode(BaseNode, ABC):
    @abstractmethod
    def __init__(self, name: str, type: NodeType) -> None:
        super().__init__(name, type)
        self.__attributes: Dict[str, AttributeTypes] = {}
    
    @property
    def attributes(self) -> ItemsView[str, AttributeTypes]:
        return self.__attributes.items()
    
    def add_attribute(self, key: str, value: AttributeTypes):         
        if key in self.__attributes.keys():
            raise KeyError(f"Attribute with key '{key}' in node '{self.name}' already exists. Use 'update_attribute' to update the value.")
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
            raise KeyError(f"Attribute with key '{key}' in node '{self.name}' does not exist. Use 'add_attribute' to add a new attribute.")
        
        if type(self.__attributes[key]) != type(value):
            raise TypeError(f"Attribute with key '{key}' in node '{self.name}' is of type '{type(self.__attributes[key])}'. Cannot update attribute with value of type '{type(value)}'.")
        self.__attributes[key] = value

    def update_attributes(self, **attributes: AttributeTypes):
        for update_key, update_value in attributes.items():
            self.update_attribute(update_key, update_value)

    def clear_attributes(self) -> None:
        self.__attributes.clear()

class GroupNode(AttributeNode):
    def __init__(self, name: str):
        super().__init__(name, NodeType.GROUP)

class DataNode(AttributeNode):
    def __init__(self, name: str, data: Tensor = torch.empty(0)):
        super().__init__(name, NodeType.DATASET)
        self.__data: Tensor = data

    @property
    def data(self) -> Tensor:
        return self.__data

    @data.setter
    def data(self, value: Tensor) -> None:
        self.__data = value

    @data.deleter
    def data(self) -> None:
        self.__data = torch.empty(0)

# Test code
if __name__ == "__main__":
    root = RootNode()
    group = GroupNode("group1")
    data = DataNode("data1", torch.randn(5, 5))

    print(f"Root: {root.name}, {root.type}")
    print(f"Group: {group.name}, {group.type}")
    print(f"Data: {data.name}, {data.type}")

    group.add_attribute("attr1", 10)
    print(f"Group attribute: {group.get_attribute('attr1')}")

    data.add_attribute("shape", list(data.data.shape))
    print(f"Data attribute: {data.get_attribute('shape')}")

    # Check if BaseNode and AttributeNode are abstract
    for AbstractClass in (BaseNode, AttributeNode):
        try:
            test = AbstractClass("test", NodeType.ROOT)  # type: ignore
            print(f"Warning: Successfully instantiated {AbstractClass.__name__}")
        except TypeError as e:
            print(f"Cannot instantiate {AbstractClass.__name__}: {e}")

    print(f"Data tensor: {data.data}")
    data.data = torch.zeros(3, 3)
    print(f"Updated data tensor: {data.data}")