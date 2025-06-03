import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..config import settings
from ..data import DataContainer
from ..io import BaseBackend, IOPathWrapper
from ..io.parsers import BaseParser, find_parser_for_extension, get_all_supported_extensions


class BaseReader(ABC):
    @abstractmethod
    def __init__(
        self,
        num_files: int | None = None,
        download_remote_files: bool = False,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
    ):
        """Initialize an instance of the BaseReader class.

        Parameters:
            num_files (int, optional): The number of files to read. If not specified, all files will be read.
                Default is None.
            download_remote_files (bool, optional): Whether to download remote files to local storage. Set this
                to True if frequent access to the same files is needed. Default is False to avoid unnecessary downloads.
            cache_files (bool, optional): Whether to cache the files list in memory. If set to False, changes to the
                detected files will be reflected at runtime. Default is True.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data. If not specified,
                the parser will be auto selected based on the file extension. Default is None.
        """
        # Assign private attributes
        self.__parser = parser
        self.__supported_extensions = tuple(parser.supported_extensions if parser else get_all_supported_extensions())
        self.__num_files = num_files
        self.__files: list[str] | None = None
        self.__cache_files = cache_files
        self.__download_remote_files = download_remote_files

        # Setup cache directory if remote files are downloaded
        if self.__download_remote_files:
            self._setup_cache_base_dir()

    @abstractmethod
    def _create_backend(self) -> BaseBackend:
        """Create a new backend instance.

        This method must be implemented by subclasses to create or
        recreate their backend when needed or after unpickling.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _get_reader_params(self) -> str:
        """Get a string representation of the reader parameters used to create the backend."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def remote_source(self) -> bool:
        """Return True if the reader is reading from a remote source, False otherwise."""
        return self.backend.remote_source

    @property
    def cache_files(self) -> bool:
        """Return True if the reader caches the files/file-paths, False otherwise."""
        return self.__cache_files

    @property
    def backend(self) -> BaseBackend:
        """The backend that the reader uses to read the data."""
        if not hasattr(self, "_BaseReader__backend"):
            self.__backend = self._create_backend()
        return self.__backend

    @property
    def parser(self) -> type[BaseParser] | None:
        """The parser that the reader uses to parse the data."""
        return self.__parser

    @property
    def num_files(self) -> int | None:
        """The number of files to read."""
        return self.__num_files

    @property
    def files(self) -> list[str]:
        """List of files that the reader is able to read."""
        # If caching is disabled return the file list from the backend
        if not self.__cache_files:
            remote_files = self.backend.get_file_list(extensions=self.__supported_extensions, num_files=self.num_files)
            return self._download_missing_files(remote_files)

        # If files have never been loaded, load them from the backend
        if self.__files is None:
            remote_files = self.backend.get_file_list(extensions=self.__supported_extensions, num_files=self.num_files)
            self.__files = self._download_missing_files(remote_files)

        # Return the cached files list
        return self.__files

    def __getstate__(self):
        """Prepare object for pickling by removing the backend."""
        state = self.__dict__.copy()
        # Remove backend reference - will be recreated when needed
        if "_BaseReader__backend" in state:
            del state["_BaseReader__backend"]
        # Clear files cache to force reloading
        state["_BaseReader__files_cache"] = None
        return state

    def __setstate__(self, state):
        """Restore object from pickled state."""
        # Just restore the state dictionary - backend will be created
        # lazily when first accessed
        self.__dict__.update(state)

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self._get_reader_params()}, num_files={self.num_files}, "
            f"download_remote_files={self.__download_remote_files}, cache_files={self.cache_files}, "
            f"parser={self.__parser.__name__ if self.__parser else None})"
        )

    def __getitem__(self, idx: int) -> DataContainer:
        if idx < 0 or idx >= len(self.files):
            raise IndexError(f"Index out of bounds. Must be in range [0, {len(self.files)}[")
        return self.read_file(self.files[idx])

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[DataContainer]:
        # Take a snapshot of the file list ==> to avoid undefined behavior when the file list changes during iteration
        # and caching is of
        file_paths = self.files

        for file in file_paths:
            yield self.read_file(file)

    def _get_manifest_path(self, cache_dir: str) -> str:
        """Get path to manifest file."""
        return os.path.join(cache_dir, "downloaded.json")

    def _load_manifest(self, cache_dir: str) -> dict[str, str]:
        """Load manifest: {remote_path: local_filename}."""
        manifest_path = self._get_manifest_path(cache_dir)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self, cache_dir: str, manifest: dict[str, str]):
        """Save manifest to disk."""
        manifest_path = self._get_manifest_path(cache_dir)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _download_missing_files(self, remote_files: list[str]) -> list[str]:
        """Download missing files upfront, return local paths."""
        if not self.remote_source or not self.__download_remote_files:
            return remote_files

        # Setup cache
        reader_id = f"{self.__class__.__name__}_{self._get_reader_params()}"
        cache_dir = self._setup_cache_dir(reader_id)
        manifest = self._load_manifest(cache_dir)

        # Find missing files
        missing_files = []
        local_paths = []

        for remote_path in remote_files:
            if remote_path in manifest:
                # Already downloaded
                local_filename = manifest[remote_path]
                local_path = os.path.join(cache_dir, local_filename)
                # Verify file still exists
                if os.path.exists(local_path):
                    local_paths.append(local_path)
                    continue

            # Need to download
            missing_files.append(remote_path)

        # Download missing files with progress
        if missing_files:
            from tqdm.auto import tqdm

            with tqdm(total=len(missing_files), desc="Downloading files") as pbar:
                for remote_path in missing_files:
                    # Generate local filename
                    local_filename = hashlib.md5(remote_path.encode()).hexdigest() + os.path.splitext(remote_path)[1]
                    local_path = os.path.join(cache_dir, local_filename)

                    # Download
                    self.backend.download_file(remote_path, local_path)

                    # Update manifest and results
                    manifest[remote_path] = local_filename
                    local_paths.append(local_path)
                    pbar.update(1)

            # Save updated manifest
            self._save_manifest(cache_dir, manifest)

        return local_paths

    def _setup_cache_base_dir(self):
        """Setup the base cache directory in the configured download directory."""
        # Create base cache dir in users home directory
        base_dir = os.path.join(settings.download_dir, ".pythermondt_cache")
        os.makedirs(base_dir, exist_ok=True)

        # Add standard cache markers
        # CACHEDIR.TAG
        tag_file = os.path.join(base_dir, "CACHEDIR.TAG")
        if not os.path.exists(tag_file):
            with open(tag_file, "w") as f:
                f.write("Signature: 8a477f597d28d172789f06886806bc55\n")
                f.write("# This file is a cache directory tag automatically created by pythermondt.\n")
                f.write("# For information about cache directory tags see https://bford.info/cachedir/\n")

        # Create .gitignore file to ignore cache files in git
        gitignore = os.path.join(base_dir, ".gitignore")
        if not os.path.exists(gitignore):
            with open(gitignore, "w") as f:
                f.write("# Automatically created by pythermondt\n")
                f.write("*\n")

    def _setup_cache_dir(self, reader_id: str) -> str:
        """Step up the cache directory for the reader instance.

        Parameters:
            reader_id (str): A unique identifier for the reader instance.

        Returns:
            str: The path to the cache directory.
        """
        # Create base cache dir in users home directory
        base_dir = os.path.join(settings.download_dir, ".pythermondt_cache")

        # Hash the reader ID for a consistent directory name
        dir_hash = hashlib.md5(reader_id.encode()).hexdigest()

        # Create full path
        cache_dir = os.path.join(base_dir, dir_hash, "raw")
        os.makedirs(cache_dir, exist_ok=True)

        return cache_dir

    def read_file(self, file_path: str) -> DataContainer:
        """Read a file from the specified path and return it as a DataContainer object.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            DataContainer: The data contained in the file, parsed and returned as a DataContainer object.

        Raises:
            ValueError: If the file type cannot be determined or if no parser is found for the file extension.
        """
        if self.remote_source and self.__download_remote_files:
            # If remote source and files are downloaded, use pre-downloaded local files
            file_data = IOPathWrapper(file_path)
        else:
            # Get raw file data from backend
            file_data = self.backend.read_file(file_path)

        # If parser was specified during initialization, use it
        if self.__parser is not None:
            return self.__parser.parse(file_data)

        # Otherwise, choose a parser based on file extension
        _, ext = os.path.splitext(file_path)
        if not ext:
            raise ValueError(f"Cannot determine file type for {file_path} - no extension found")

        # Find appropriate parser for this extension
        parser_cls = find_parser_for_extension(ext)
        if parser_cls is None:
            raise ValueError(f"No parser found for file extension: {ext}")

        # Parse the file with the selected parser
        return parser_cls.parse(file_data)
