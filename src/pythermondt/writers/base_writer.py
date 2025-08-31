from abc import ABC, abstractmethod

from ..data import DataContainer
from ..io.backends import BaseBackend


class BaseWriter(ABC):
    @abstractmethod
    def write(self, container: DataContainer, file_name: str):
        """Actual implementation of the writing a single DataContainer to the destination folder.

        Args:
            container (DataContainer): The DataContainer which should be written to the destination folder.
            destination_folder (str): The destination folder to write to.
            file_name (str): The name of the DataContainer.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _create_backend(self) -> BaseBackend:
        """Create a new backend instance.

        This method must be implemented by subclasses to create or
        recreate their backend when needed or after unpickling.
        """
        raise NotImplementedError("Subclasses must implement this method")
