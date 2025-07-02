import collections
from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from ..data import DataContainer
from ..transforms.utils import Compose, _BaseTransform, _flatten_transforms


class BaseDataset(Dataset, ABC):
    """Base class for all ThermoDatasets."""

    def __init__(self, parent: "BaseDataset | None" = None, transform: _BaseTransform | None = None):
        self.__parent = parent
        self.__transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """Return length of dataset - implemented by concrete classes."""
        pass

    @property
    @abstractmethod
    def files(self) -> list[str]:
        """Get the list of files associated with this dataset."""
        pass

    @property
    def parent(self) -> "BaseDataset | None":
        """Get parent dataset if available."""
        return self.__parent

    @property
    def transform(self) -> _BaseTransform | None:
        """Get the transform for this dataset."""
        return self.__transform

    def get_transform_chain(self) -> _BaseTransform:
        """Walk up graph to build the complete sequence transforms for this dataset and compose it in a single one."""
        transforms: collections.deque[_BaseTransform] = collections.deque()
        current: BaseDataset | None = self

        while current is not None:
            if current.transform:
                # Flatten the transforms
                flattened = _flatten_transforms(
                    current.transform.transforms if isinstance(current.transform, Compose) else [current.transform]
                )
                transforms.extendleft(reversed(flattened))
            current = current.parent

        return Compose(list(transforms))

    @abstractmethod
    def _load_raw_data(self, idx: int) -> DataContainer:
        """Load raw data at index - implemented by concrete classes."""
        pass
