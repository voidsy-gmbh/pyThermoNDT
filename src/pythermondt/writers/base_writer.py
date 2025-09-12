from abc import ABC, abstractmethod

from ..data import DataContainer
from ..data.datacontainer.serialization_ops import CompressionType
from ..io.backends import BaseBackend


class BaseWriter(ABC):
    def __init__(self):
        """Constructor for the BaseWriter class. Should be called by all subclasses."""
        # Internal state
        self.__backend: BaseBackend | None = None

    @property
    def backend(self) -> BaseBackend:
        """The backend that the writer uses to write the data."""
        if not self.__backend:
            self.__backend = self._create_backend()
        return self.__backend

    @abstractmethod
    def write(
        self,
        container: DataContainer,
        file_name: str,
        compression: CompressionType = "lzf",
        compression_opts: int | None = 4,
    ):
        """Actual implementation of the writing a single DataContainer to the destination folder.

        Args:
            container (DataContainer): The DataContainer which should be written to the destination folder.
            file_name (str): The name of the DataContainer.
            compression (CompressionType): The compression method to use for the HDF5 file.
                Default is "lzf" which is a fast compression method. For smaller files, "gzip" can be used at the cost
                of speed. Use "none" to disable compression for faster read/write operations, resulting in larger files.
            compression_opts (int): The compression level for gzip compression. Ignored if compression is not "gzip".
                Default is 4, which is a good balance between speed and compression ratio.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _create_backend(self) -> BaseBackend:
        """Create a new backend instance.

        This method must be implemented by subclasses to create or
        recreate their backend when needed or after unpickling.
        """
        raise NotImplementedError("Subclasses must implement this method")
