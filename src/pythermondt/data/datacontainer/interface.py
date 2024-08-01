from typing import Protocol, List, Dict, Tuple
from torch import Tensor
from numpy import ndarray

class _IDdataContainer(Protocol):
    """
    Interface for the DataContainer class. This interface defines the methods that the DataContainer class must implement.
    """
    _groups: List[str] = []
    _datasets: Dict[Tuple[str, str], Tensor | None] = {}
    _attributes: Dict[Tuple[str, str], Dict[str, str | int | float | list | dict]] = {}

    def __init__(self):
        raise NotImplementedError
     
    def get_groups(self):
        raise NotImplementedError
     
    def get_datasets(self, group_name: str='') -> List[str]:
        raise NotImplementedError
     
    def get_dataset(self, path: str) -> Tensor:
        raise NotImplementedError
    
    def get_attribute(self, path: str, attribute_name: str) -> str | int | float | list | dict:
        raise NotImplementedError
    
    def fill_dataset(self, path: str, data: Tensor | ndarray, **attributes):
        raise NotImplementedError

    def update_attributes(self, path: str, **attributes):
        raise NotImplementedError