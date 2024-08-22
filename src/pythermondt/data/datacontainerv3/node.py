from typing import List, Dict, ItemsView
from enum import Enum
from torch import Tensor
from abc import ABC
import torch

class NodeType(Enum):
    ROOT = "root"
    DATASET = "dataset"
    GROUP = "group"

class Node(ABC):
    def __init__(self, name: str, type: NodeType) -> None:
        self._name = name
        self._type = type
        self._attributes: Dict[str, str | int | float | list | dict] = {}

    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def type(self) -> NodeType:
        return self._type
    
    @property
    def attributes(self) -> ItemsView[str, str | int | float | list | dict]:
        return self._attributes.items()
    
    def add_attribute(self, key: str, value: str | int | float | list | dict):
        if key in self._attributes.keys():
            raise KeyError(f"Attribute with key '{key}' in node '{self._name}' already exists. Use 'update_attribute' to update the value.")
        self._attributes[key] = value
    
    def add_attributes(self, **attributes: Dict[str, str | int | float | list | dict]):
        for key, value in attributes.items():
            self.add_attribute(key, value)

    def get_attribute(self, key: str) -> str | int | float | list | dict:
        if key not in self._attributes.keys():
            raise KeyError(f"Attribute with key '{key}' in node '{self._name}' does not exist.")
        return self._attributes[key]
    
    def remove_attribute(self, key: str) -> None:
        if key not in self._attributes.keys():
            raise KeyError(f"Attribute with key '{key}' in node '{self._name}' does not exist.")
        del self._attributes[key]

class GroupNode(Node):
    def __init__(self, name: str):
        super().__init__(name, NodeType.GROUP)
    
class RootNode(Node):
    def __init__(self):
        super().__init__("root", NodeType.ROOT)

class DataNode(Node):
    def __init__(self, name: str, data: Tensor=torch.empty(0)):
        super().__init__(name, NodeType.DATASET)
        self.__data = data

    @property
    def data(self) -> Tensor:
        return self.get_data()
    
    @data.setter
    def data(self, data: Tensor):
        self.set_data(data)

    @data.deleter
    def data(self):
        self.remove_data()

    def set_data(self, data: Tensor):
        self.__data = data

    def remove_data(self) -> bool:
        if self.__data is not torch.empty(0):
            self.__data = torch.empty(0)
            return True
        return False

    def get_data(self) -> Tensor:
        return self.__data
        

# Test
if __name__ == "__main__":
    test = Node("test", NodeType.ROOT)

    print(test.name)

    test.name = "test2"

    test.add_attribute("test", 1)

    print(test.name)
    print(test.type)
    print(test.attributes)  
    print(test.get_attribute("test1")) 