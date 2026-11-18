"""Discover and cache the Azure integration data shared by all end-to-end tests.

Files under ``<TEST_PREFIX>/integration/<parser folder>/`` pair by case name: ``source<N>.<ext>`` with
``expected<N>.hdf5``. All readers use the integration root as prefix, so they share one download cache.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

from azure.core.exceptions import AzureError

from pythermondt.io import FileInfo
from pythermondt.readers import AzureBlobReader
from tests.support.remote_data import RemoteTestConfig, cloud_skip_reason, load_remote_config

# Subfolder of the test data prefix that holds the integration data.
INTEGRATION_PREFIX = "integration"


@dataclass(frozen=True)
class FilePair:
    """Source file and its expected HDF5 output in one parser folder.

    ``has_expected`` is False for a new source whose expected output does not exist yet.
    """

    folder: str
    source_uri: str
    expected_uri: str
    has_expected: bool = True

    @property
    def source_name(self) -> str:
        return _file_name(self.source_uri)

    @property
    def expected_name(self) -> str:
        return _file_name(self.expected_uri)

    @property
    def id(self) -> str:
        """Return the source path relative to the integration root, e.g. 'edevis/source3.di'."""
        return f"{self.folder}/{self.source_name}"


@dataclass(frozen=True, eq=False)
class IntegrationData:
    """Discovered file pairs with the shared reader, or a safe error message if discovery failed."""

    pairs: tuple[FilePair, ...] = ()
    config: RemoteTestConfig | None = None
    reader: AzureBlobReader | None = None
    error: str | None = None

    @property
    def complete_pairs(self) -> tuple[FilePair, ...]:
        return tuple(pair for pair in self.pairs if pair.has_expected)

    @property
    def folders(self) -> tuple[str, ...]:
        """Return parser folders with at least one complete pair."""
        return tuple(sorted({pair.folder for pair in self.complete_pairs}))


@cache
def collect_integration_data() -> IntegrationData:
    """Discover the integration data once per process for test collection.

    Returns empty data without network access when cloud tests are disabled. Discovery failures become an error
    message instead of a collection error, so offline tests still run. Exception text is omitted because it can
    contain credentials or signed URLs.
    """
    if cloud_skip_reason() is not None:
        return IntegrationData()
    try:
        config = load_remote_config()
        reader, pairs = load_file_pairs(config)
    except (AzureError, RuntimeError, ValueError, OSError) as error:
        return IntegrationData(
            error=f"Remote integration discovery failed ({type(error).__name__}). "
            "Check test data configuration, Azure access, and network connectivity."
        )
    if not pairs:
        return IntegrationData(error="No paired integration test files found in remote test data.")
    return IntegrationData(pairs=pairs, config=config, reader=reader)


def load_file_pairs(config: RemoteTestConfig) -> tuple[AzureBlobReader, tuple[FilePair, ...]]:
    """List the integration root once and return the shared reader with all paired files."""
    reader = _make_reader(config)
    root_uri = f"az://{config.container_name}/{_integration_prefix(config)}"
    return reader, pair_files((uri.removeprefix(root_uri) for uri in reader.file_uris), root_uri)


def pair_files(relative_paths: Iterable[str], root_uri: str) -> tuple[FilePair, ...]:
    """Pair source and expected files by case name.

    Args:
        relative_paths: File paths relative to the integration root, e.g. "edevis/source3.di".
        root_uri: URI of the integration root, ending with "/".

    Returns:
        One pair per source, sorted by id. Sources without expected output have ``has_expected=False``.
        Files outside direct parser folders and expected outputs without source are ignored.
    """
    pairs = []
    for folder, names in _group_by_folder(relative_paths).items():
        for name in names:
            if not name.startswith("source"):
                continue
            expected_name = f"expected{name.removeprefix('source').rsplit('.', 1)[0]}.hdf5"
            folder_uri = f"{root_uri}{folder}/"
            pairs.append(FilePair(folder, folder_uri + name, folder_uri + expected_name, expected_name in names))
    return tuple(sorted(pairs, key=lambda pair: pair.id))


def folder_readers(
    config: RemoteTestConfig, pairs: Iterable[FilePair], folder: str
) -> tuple[AzureBlobReader, AzureBlobReader]:
    """Create source and expected readers for the complete pairs of one parser folder.

    The readers reuse the shared download cache.

    Raises:
        RuntimeError: If the readers would not list the files in matching pair order.
    """
    folder_pairs = [pair for pair in pairs if pair.folder == folder and pair.has_expected]
    readers = []
    for uris in ([pair.source_uri for pair in folder_pairs], [pair.expected_uri for pair in folder_pairs]):
        reader = _make_reader(config, file_filter=_UriFilter(tuple(uris)))
        # Datasets iterate in listing order, so a different order would compare unrelated files.
        if reader.file_uris != uris:
            raise RuntimeError(f"Reader order for {folder!r} does not match pairs: expected {uris}.")
        readers.append(reader)
    return readers[0], readers[1]


class _UriFilter:
    """Picklable filter for exact blob URIs."""

    def __init__(self, uris: tuple[str, ...]) -> None:
        self.uris = uris

    def __call__(self, info: FileInfo) -> bool:
        return info.path in self.uris


def _make_reader(config: RemoteTestConfig, file_filter: _UriFilter | None = None) -> AzureBlobReader:
    # The cache key includes the prefix, so every reader uses the integration root.
    return AzureBlobReader(
        account_url=str(config.account_url),
        container_name=config.container_name,
        prefix=_integration_prefix(config),
        download_files=True,
        file_filter=file_filter,
    )


def _integration_prefix(config: RemoteTestConfig) -> str:
    return f"{config.prefix}/{INTEGRATION_PREFIX}/" if config.prefix else f"{INTEGRATION_PREFIX}/"


def _group_by_folder(relative_paths: Iterable[str]) -> dict[str, set[str]]:
    """Group file names by direct parser folder; ignore top-level and nested files."""
    grouped: dict[str, set[str]] = {}
    for path in relative_paths:
        folder, _, name = path.partition("/")
        if folder and name and "/" not in name:
            grouped.setdefault(folder, set()).add(name)
    return grouped


def _file_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]
