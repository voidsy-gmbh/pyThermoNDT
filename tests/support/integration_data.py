"""Discover and cache the Azure integration data shared by all end-to-end tests.

Files under ``<TEST_PREFIX>/integration/<parser folder>/`` keep their original names. Each source pairs with
``<source filename>.expected.hdf5`` in the same folder. All readers share one download cache.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from os.path import splitext

from azure.core.exceptions import AzureError

from pythermondt.dataset import IndexedThermoDataset, ThermoDataset
from pythermondt.io import FileInfo
from pythermondt.io.parsers import get_all_supported_extensions
from pythermondt.readers import AzureBlobReader
from tests.support.remote_data import RemoteTestConfig, cloud_skip_reason, load_remote_config

# Subfolder of the test data prefix that holds the integration data.
INTEGRATION_PREFIX = "integration"
EXPECTED_SUFFIX = ".expected.hdf5"


class PairingError(ValueError):
    """Unsafe integration file paths; the message is safe to show."""


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
        """Return the source path relative to the integration root, e.g. 'edevis/panel_run03.ThermoVis'."""
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
    except PairingError as error:
        return IntegrationData(error=str(error))
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
    """Pair sources with expected outputs named after the full source filename.

    Args:
        relative_paths: File paths relative to the integration root, e.g. "edevis/panel_run03.ThermoVis".
        root_uri: URI of the integration root, ending with "/".

    Returns:
        One pair per source, sorted by id. Sources without expected output have ``has_expected=False``.
        Files outside direct parser folders and expected outputs without source are ignored.

    Raises:
        PairingError: If an integration file path is unsafe as a local path.
    """
    supported_extensions = get_all_supported_extensions()
    pairs = []
    for folder, names in _group_by_folder(relative_paths).items():
        folder_uri = f"{root_uri}{folder}/"
        for source_name in names:
            if source_name.endswith(EXPECTED_SUFFIX) or splitext(source_name)[1] not in supported_extensions:
                continue
            expected_name = f"{source_name}{EXPECTED_SUFFIX}"
            pairs.append(FilePair(folder, folder_uri + source_name, folder_uri + expected_name, expected_name in names))
    return tuple(sorted(pairs, key=lambda pair: pair.id))


def folder_datasets(
    config: RemoteTestConfig, pairs: Iterable[FilePair], folder: str
) -> tuple[ThermoDataset, IndexedThermoDataset]:
    """Create aligned datasets for the complete pairs of one parser folder.

    Raises:
        RuntimeError: If a reader no longer lists a file from a complete pair.
    """
    expected_by_source = {
        pair.source_uri: pair.expected_uri for pair in pairs if pair.folder == folder and pair.has_expected
    }
    datasets = []
    for uris in (frozenset(expected_by_source), frozenset(expected_by_source.values())):
        reader = _make_reader(config, file_filter=_UriFilter(uris))
        if missing := uris.difference(reader.file_uris):
            raise RuntimeError(f"Reader for {folder!r} did not list paired files: {sorted(missing)}.")
        datasets.append(ThermoDataset(reader))

    source, expected = datasets
    # Appending a suffix can change listing order when one filename starts with another.
    expected_indices = {uri: index for index, uri in enumerate(expected.files)}
    return source, IndexedThermoDataset(expected, [expected_indices[expected_by_source[uri]] for uri in source.files])


class _UriFilter:
    """Picklable filter for exact blob URIs."""

    def __init__(self, uris: frozenset[str]) -> None:
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
    """Group file names by direct parser folder; ignore top-level and nested files.

    Raises:
        PairingError: If a folder or file name would leave its folder as a local path, e.g. "..".
    """
    grouped: dict[str, set[str]] = {}
    for path in relative_paths:
        folder, _, name = path.partition("/")
        if folder and name and "/" not in name:
            # Blob names may contain these, but tools use them as local path components.
            if {folder, name} & {".", ".."} or "\\" in path:
                raise PairingError(f"Unsafe integration file path {path!r}. Rename the blob.")
            grouped.setdefault(folder, set()).add(name)
    return grouped


def _file_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]
