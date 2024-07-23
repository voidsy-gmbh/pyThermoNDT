import torch.nn as nn
from ..data import DataContainer
from typing import Callable, TypeVar, cast, Any

C = TypeVar('C', bound=Callable[..., DataContainer])

def proxy(f: C) -> C:
    return cast(C, lambda self, *args, **kwargs: f(self, *args, **kwargs))

class Compose(nn.Module):
    def __init__(self, transforms: list):
        super(Compose, self).__init__()
        self.transforms = transforms
    
    def forward(self, container: DataContainer) -> DataContainer:
        for t in self.transforms:
            container = t(container)
        return container

    __call__: Callable[..., DataContainer] = proxy(forward)