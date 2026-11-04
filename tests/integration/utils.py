from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pythermondt.io import FileInfo
from pythermondt.readers import AzureBlobReader
from tests.support.remote_data import RemoteTestConfig, cloud_skip_reason, load_remote_config

# Subfolder of the test data prefix that holds the integration data.
INTEGRATION_PREFIX = "integration"

# Factory that creates the source and expected readers for one integration case.
CaseReaders = Callable[["IntegrationTestCase"], tuple[AzureBlobReader, AzureBlobReader]]


@dataclass(frozen=True)
class IntegrationTestCase:
    """Integration case pairing source and expected blobs from the remote test data.

    Attributes:
        type: Test type; subfolder name under the integration prefix (e.g. "simulation").
        name: Case number ("1", "2", ...) or "glob" for one case over all files of the type.
        source_files: Source file names of the case.
        expected_files: Expected output file names of the case.
    """

    type: str
    name: str
    source_files: tuple[str, ...] = ()
    expected_files: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """Return a test ID for readable pytest output."""
        return f"{self.type}_{self.name}"


def integration_test_cases() -> tuple[IntegrationTestCase, ...]:
    """Return the integration cases to parametrize.

    Discovers the cases from the remote test data when cloud tests can run. Otherwise
    returns a single placeholder case so the skipped tests stay visible and collection
    needs no network access.

    Raises:
        RuntimeError: If the remote test data is not reachable with the configured credentials.
    """
    if cloud_skip_reason() is not None:
        return (IntegrationTestCase(type="simulation", name="unavailable"),)
    return discover_test_cases(list_integration_files(load_remote_config()))


def discover_test_cases(relative_paths: Iterable[str]) -> tuple[IntegrationTestCase, ...]:
    """Discover integration cases from test data file names.

    Files under ``<type>/`` pair by case number: ``source<N>.<ext>`` with
    ``expected<N>.hdf5``. Sources without matching expected output are ignored. Each
    type also gets one "glob" case over all of its paired files.

    Args:
        relative_paths: File paths relative to the integration prefix (e.g. "simulation/source1.mat").

    Returns:
        One case per paired source file plus one "glob" case per type, sorted by id.
        Case file fields hold bare file names.
    """
    cases: list[IntegrationTestCase] = []
    for test_type, paths in _group_by_type(relative_paths).items():
        sources = sorted(_file_name(path) for path in paths if _file_name(path).startswith("source"))
        expected_names = {_file_name(path) for path in paths if _file_name(path).startswith("expected")}

        paired_sources: list[str] = []
        paired_expected: list[str] = []
        for source in sources:
            case_name = source.removeprefix("source").rsplit(".", 1)[0]
            expected_name = f"expected{case_name}.hdf5"
            if expected_name not in expected_names:
                continue
            cases.append(
                IntegrationTestCase(
                    type=test_type,
                    name=case_name,
                    source_files=(source,),
                    expected_files=(expected_name,),
                )
            )
            paired_sources.append(source)
            paired_expected.append(expected_name)

        if paired_sources:
            cases.append(
                IntegrationTestCase(
                    type=test_type,
                    name="glob",
                    source_files=tuple(paired_sources),
                    expected_files=tuple(sorted(paired_expected)),
                )
            )

    return tuple(sorted(cases, key=lambda case: case.id))


def list_integration_files(config: RemoteTestConfig) -> list[str]:
    """List the integration test data files from the remote container.

    Args:
        config: Connection settings of the remote test data container.

    Returns:
        File paths relative to the integration prefix (e.g. "simulation/source1.mat").
    """
    prefix = f"{config.prefix}/{INTEGRATION_PREFIX}/" if config.prefix else f"{INTEGRATION_PREFIX}/"
    reader = AzureBlobReader(
        account_url=config.account_url,
        connection_string=config.connection_string,
        container_name=config.container_name,
        prefix=prefix,
    )
    uri_prefix = f"az://{config.container_name}/{prefix}"
    return [info.path.removeprefix(uri_prefix) for info in reader.file_entries]


def make_case_readers(
    config: RemoteTestConfig, test_case: IntegrationTestCase
) -> tuple[AzureBlobReader, AzureBlobReader]:
    """Create the source and expected readers for one integration case.

    Both readers download and cache their blobs locally before parsing.

    Args:
        config: Connection settings of the remote test data container.
        test_case: The integration case to load.

    Returns:
        The source reader and the expected output reader of the case.

    Raises:
        RuntimeError: If the remote test data does not contain the case files.
    """
    prefix = (
        f"{config.prefix}/{INTEGRATION_PREFIX}/{test_case.type}/"
        if config.prefix
        else f"{INTEGRATION_PREFIX}/{test_case.type}/"
    )
    source_reader = _make_reader(config, prefix, test_case.source_files)
    expected_reader = _make_reader(config, prefix, test_case.expected_files)
    return source_reader, expected_reader


class _BlobPathFilter:
    """Picklable filter for exact blob paths, excluding nested folders."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        self.paths = paths

    def __call__(self, info: FileInfo) -> bool:
        return info.path in self.paths


def _make_reader(config: RemoteTestConfig, prefix: str, names: tuple[str, ...]) -> AzureBlobReader:
    reader = AzureBlobReader(
        account_url=config.account_url,
        connection_string=config.connection_string,
        container_name=config.container_name,
        prefix=prefix,
        download_files=True,
        file_filter=_BlobPathFilter(tuple(f"az://{config.container_name}/{prefix}{name}" for name in names)),
    )
    found = sorted(_file_name(file_path) for file_path in reader.files)
    if found != sorted(names):
        raise RuntimeError(
            f"Test data mismatch under az://{config.container_name}/{prefix}: expected {sorted(names)}, found {found}."
        )
    return reader


def _group_by_type(relative_paths: Iterable[str]) -> dict[str, list[str]]:
    """Group direct children of type folders; ignore top-level and nested files."""
    grouped: dict[str, list[str]] = {}
    for path in relative_paths:
        test_type, _, name = path.partition("/")
        if not test_type or not name or "/" in name:
            continue
        grouped.setdefault(test_type, []).append(path)
    return grouped


def _file_name(path: str) -> str:
    """Return the file name of a path or blob URI."""
    return path.rsplit("/", 1)[-1]
