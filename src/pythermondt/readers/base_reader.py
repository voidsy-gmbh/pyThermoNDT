import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import partial
from multiprocessing.pool import ThreadPool
from threading import Lock

from tqdm.auto import tqdm

from ..config import settings
from ..data import DataContainer
from ..io import BaseBackend, IOPathWrapper
from ..io.parsers import BaseParser, find_parser_for_extension, get_all_supported_extensions


class BaseReader(ABC):
    @abstractmethod
    def __init__(
        self,
        num_files: int | None = None,
        download_files: bool = False,
        cache_files: bool = True,
        parser: type[BaseParser] | None = None,
    ):
        """Initialize an instance of the BaseReader class.

        Parameters:
            num_files (int, optional): The number of files to read. If not specified, all files will be read.
                Default is None.
            download_files (bool, optional): Whether to automatically cache remote files locally during operations.
                When False, files are downloaded on-demand but not saved locally. Default is False.
            cache_files (bool, optional): Whether to cache the files list in memory. If set to False, changes to the
                detected files will be reflected at runtime. Default is True.
            parser (Type[BaseParser], optional): The parser that the reader uses to parse the data. If not specified,
                the parser will be auto selected based on the file extension. Default is None.
        """
        # Assign private attributes
        self.__parser = parser
        self.__num_files = num_files
        self.__cache_files = cache_files
        self.__download_files = download_files

        # Internal state
        self.__files: list[str] | None = None
        self.__supported_extensions = tuple(parser.supported_extensions if parser else get_all_supported_extensions())
        self.__manifest_path: str | None = None
        self.__manifest_lock = Lock()
        self.__reader_cache_dir: str | None = None

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
    def download_files(self) -> bool:
        """Return True if the reader downloads remote files, False otherwise."""
        return self.__download_files

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
    def manifest_path(self) -> str:
        """Path to the manifest file that stores downloaded files."""
        if self.__manifest_path is None or self.__reader_cache_dir is None:
            self.__reader_cache_dir, self.__manifest_path = self._setup_cache_dir()
        return self.__manifest_path

    @property
    def reader_cache_dir(self) -> str:
        """Path to the manifest file that stores downloaded files."""
        if self.__manifest_path is None or self.__reader_cache_dir is None:
            self.__reader_cache_dir, self.__manifest_path = self._setup_cache_dir()
        return self.__reader_cache_dir

    @property
    def files(self) -> list[str]:
        """List of files that the reader is able to read."""
        # If caching is disabled return the file list from the backend
        if not self.__cache_files:
            return self.backend.get_file_list(extensions=self.__supported_extensions, num_files=self.num_files)

        # If files have never been loaded, load them from the backend
        if self.__files is None:
            self.__files = self.backend.get_file_list(extensions=self.__supported_extensions, num_files=self.num_files)

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
            f"download_remote_files={self.__download_files}, cache_files={self.cache_files}, "
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

    def _load_manifest(self, manifest_path: str) -> dict[str, str]:
        """Load manifest from disk with thread safety."""
        with self.__manifest_lock:
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    return {}
            return {}

    def _save_manifest(self, manifest_path: str, manifest: dict[str, str]):
        """Save manifest to disk with thread safety."""
        with self.__manifest_lock:
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            # Atomic write using temp file
            temp_path = manifest_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(manifest, f, indent=2)
            os.replace(temp_path, manifest_path)

    def _setup_cache_dir(self) -> tuple[str, str]:
        """Setup the cache directory in the configured download directory for this reader.

        Returns:
            tuple[str, str]: A tuple containing reader cache directory and manifest path.
        """
        # Create base cache dir in users home directory
        base_dir = os.path.join(settings.download_dir, ".pythermondt_cache")
        reader_id = f"{self.__class__.__name__}_{self._get_reader_params()}"
        dir_hash = hashlib.md5(reader_id.encode()).hexdigest()
        reader_cache_dir = os.path.join(base_dir, dir_hash)
        manifest_path = os.path.join(reader_cache_dir, "downloaded.json")

        # Ensure directories exist
        os.makedirs(os.path.join(reader_cache_dir, "./raw"), exist_ok=True)

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
        return reader_cache_dir, manifest_path

    def _download_single_file(self, remote_path: str, manifest: dict[str, str]) -> tuple[str, str]:
        """Download a single file from the remote source and return its local path.

        Parameters:
            remote_path (str): The path to the file on the remote source.
            manifest (dict[str, str]): The manifest dictionary containing the current state of downloaded files.

        Returns:
            tuple[str, str]: A tuple containing the relative local path to the downloaded file and its remote path.
        """
        # Check if already cached and exists
        if remote_path in manifest:
            relative_path = manifest[remote_path]
            local_path = os.path.join(self.reader_cache_dir, relative_path)
            if os.path.exists(local_path):
                return remote_path, local_path

        # Download the file
        filename = hashlib.md5(remote_path.encode()).hexdigest() + os.path.splitext(remote_path)[1]
        relative_path = f"./raw/{filename}"
        local_path = os.path.join(self.reader_cache_dir, relative_path)
        self.backend.download_file(remote_path, local_path)
        return remote_path, relative_path

    def _ensure_file_cached(self, remote_path: str) -> str:
        """Ensure a file is cached locally, return local path.

        Parameters:
            remote_path (str): The path to the file on the remote source.

        Returns:
            str: The local path to the cached file.
        """
        # Load manifest
        manifest = self._load_manifest(self.manifest_path)

        # Download the file
        remote_path, relative_path = self._download_single_file(remote_path, manifest)

        # Update manifest
        manifest[remote_path] = relative_path
        self._save_manifest(self.manifest_path, manifest)

        return os.path.join(self.reader_cache_dir, relative_path)

    def download(self, file_paths: list[str] | None = None, num_workers: int | None = None) -> None:
        """Trigger the download of files from the remote source.

        This method will download the specified files from the remote source and cache them locally in the reader's
        cache directory. The download will only happen if the reader has a remote source and only files that are
        not already cached locally will be downloaded.

        **Note:** This works regardless of the `download_files` flag, set during reader initialization.

        Parameters:
            file_paths (list[str], optional): List of file paths to download. If None, all files that the reader is
                able to read will be downloaded. Default is None.
            num_workers (int, optional): Number of workers to use for downloading files. If None, the default number of
                workers of pyThermoNDT will be used. Default is None.
        """
        # If no remote source, do nothing
        if not self.remote_source:
            return

        # If file_paths is None, download all files that the reader is able to read
        paths_to_download = file_paths or self.files
        if not paths_to_download:
            return

        # Get cache info once (not per file) from the manifest file
        manifest = self._load_manifest(self.manifest_path)

        # Use sets for efficient bulk comparison
        requested_files = set(paths_to_download)
        cached_files = set(manifest.keys())

        # Find files that need downloading
        potentially_cached = requested_files & cached_files
        uncached_files = requested_files - cached_files

        # Check which "cached" files actually exist on disk
        missing_cached_files = set()
        for file_path in potentially_cached:
            relative_path = manifest[file_path]
            local_path = os.path.join(self.reader_cache_dir, relative_path)
            if not os.path.exists(local_path):
                missing_cached_files.add(file_path)

        # Combine files that need downloading
        to_download = uncached_files | missing_cached_files

        if not to_download:
            return  # Nothing to download

        # Download files with progress bar
        unit = "files"
        desc = f"{self.__class__.__name__} - Downloading files"
        num = len(to_download)
        workers = num_workers or settings.num_workers
        worker_fn = partial(self._download_single_file, manifest=manifest)
        if workers > 1:
            # Use ThreadPool for parallel downloads
            with ThreadPool(processes=workers) as pool:
                results = dict(tqdm(pool.imap_unordered(worker_fn, to_download), total=num, desc=desc, unit=unit))
        else:
            results = dict(tqdm(map(worker_fn, to_download), total=num, desc=desc, unit=unit))

        # Single manifest update after all downloads
        manifest.update(results)
        self._save_manifest(self.manifest_path, manifest)

    def read_file(self, file_path: str) -> DataContainer:
        """Read a file from the specified path and return it as a DataContainer object.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            DataContainer: The data contained in the file, parsed and returned as a DataContainer object.

        Raises:
            ValueError: If the file type cannot be determined or if no parser is found for the file extension.
        """
        if self.remote_source and self.__download_files:
            # If remote source and files are downloaded, use pre-downloaded local files
            local_path = self._ensure_file_cached(file_path)
            file_data = IOPathWrapper(local_path)
        else:
            # Read directly from backend
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
