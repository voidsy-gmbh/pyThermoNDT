import os
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from urllib.parse import unquote, urlparse

from tqdm.auto import tqdm

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

    @abstractmethod
    def _get_destination_path(self, file_name: str) -> str:
        """Build the destination path without changing the extension; prepare local folders if needed."""
        raise NotImplementedError("Subclasses must implement this method")

    def _copy_file(self, source_backend: BaseBackend, source_uri: str, file_name: str) -> None:
        """Transfer original bytes, using direct copies for local destinations."""
        destination_path = self._get_destination_path(file_name)
        if self.backend.remote_source:
            with source_backend.read_file(source_uri) as file_data:
                self.backend.write_file(file_data, destination_path)
        else:
            source_backend.copy(source_uri, destination_path)

    def process_parallel(
        self,
        reader: BaseReader,
        keep_file_names: bool = False,
        file_name_pattern: str = "{index}",
        compression: CompressionType = "lzf",
        compression_opts: int | None = 4,
        num_workers: int | None = None,
        raw_copy: bool = False,
    ) -> None:
        """Process multiple DataContainers from a reader in parallel.

        Args:
            reader: Reader containing DataContainers to write
            keep_file_names: Whether to keep the original file names from the reader. If True, the argument
                `file_name_pattern` will be ignored.
            file_name_pattern: Pattern for naming files. Use {index} for zero-padded index. If {index} is not present,
                it will be appended to the pattern with an underscore.
                Example: "data_{index}_name" produces "data_00000_name.hdf5", "data_00001_name.hdf5", etc.
            compression: Compression method for HDF5 files
            compression_opts: Compression level for gzip (ignored for other methods)
            num_workers: Number of workers. Defaults to global config setting.
            raw_copy: Copy original bytes without parsing or HDF5 serialization. Preserves source extensions
                and ignores compression arguments. Local destinations use direct file copies/downloads;
                remote destinations buffer each file in memory. Reader discovery filters still apply,
                but the reader's download cache is bypassed. Default is False.
        """
        # Snapshot raw-copy sources once so refreshed listings cannot change file/name pairs during transfer.
        file_uris = list(reader.file_uris) if raw_copy else []
        file_names = []
        if raw_copy:
            file_names = [os.path.basename(urlparse(uri).path) for uri in file_uris]
            if reader.backend.scheme == "file":
                file_names = [unquote(name) for name in file_names]

        # Determine length to format zero-padded indices
        n = len(file_uris) if raw_copy else len(reader)
        index_width = len(str(n))

        if "{index}" not in file_name_pattern:
            file_name_pattern += "_{index}"

        if keep_file_names and not raw_copy:
            file_names = reader.file_names  # Load names before starting workers

        def write_single(idx: int):
            if keep_file_names:
                source_name = file_names[idx] if raw_copy else reader.file_names[idx]
                file_name = source_name if raw_copy else os.path.splitext(source_name)[0]
                if not os.path.splitext(source_name)[0]:
                    raise ValueError(f"Invalid file name at index {idx}: '{source_name}'")
            else:
                # Replace {index} with zero-padded index
                file_name = file_name_pattern.replace("{index}", str(idx).zfill(index_width))
                if raw_copy:
                    file_name += os.path.splitext(file_names[idx])[1]

            if raw_copy:
                self._copy_file(reader.backend, file_uris[idx], file_name)
            else:
                self.write(reader[idx], file_name, compression, compression_opts)

        # Use ThreadPool for writing in parallel ==> I/O bound task
        num_workers = max(num_workers, 1) if num_workers is not None else settings.num_workers
        desc = f"{self.__class__.__name__} - Writing files with {num_workers} workers"
        if num_workers > 1:
            with ThreadPool(processes=num_workers) as pool:
                list(tqdm(pool.imap(write_single, range(n)), total=n, desc=desc, unit="files"))
        else:
            list(map(write_single, tqdm(range(n), desc=desc, unit="files")))
