from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool

from ..config import settings
from ..data import DataContainer
from ..data.datacontainer.serialization_ops import CompressionType
from ..io.backends import BaseBackend
from ..readers.base_reader import BaseReader


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

    def process_parallel(
        self,
        reader: BaseReader,
        file_name_pattern: str = "{index}",
        compression: CompressionType = "lzf",
        compression_opts: int | None = 4,
        num_workers: int | None = None,
    ):
        """Process multiple DataContainers from a reader in parallel.

        Args:
            reader: Reader containing DataContainers to write
            file_name_pattern: Pattern for naming files. Use {index} for zero-padded index.
                Example: "data_{index}" produces "data_00000.hdf5", "data_00001.hdf5", etc.
            compression: Compression method for HDF5 files
            compression_opts: Compression level for gzip (ignored for other methods)
            num_workers: Number of workers. Defaults to global config setting.
        """
        from tqdm.auto import tqdm

        workers = max(num_workers, 1) if num_workers is not None else settings.num_workers
        n = len(reader)

        # Format string for zero-padded indices
        index_width = len(str(n))

        def write_single(idx: int):
            container = reader[idx]
            # Replace {index} with zero-padded index
            file_name = file_name_pattern.replace("{index}", str(idx).zfill(index_width))
            self.write(container, file_name, compression, compression_opts)

        # Use ThreadPool for writing in parallel ==> I/O bound task
        desc = f"{self.__class__.__name__} - Writing files with {workers} workers"
        if workers > 1:
            with ThreadPool(processes=workers) as pool:
                list(tqdm(pool.imap(write_single, range(n)), total=n, desc=desc, unit="files"))
        else:
            list(map(write_single, tqdm(range(n), desc=desc, unit="files")))
