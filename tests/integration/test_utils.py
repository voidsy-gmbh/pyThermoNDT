from unittest.mock import PropertyMock, patch

import pytest

from tests.support.remote_data import RemoteTestConfig
from tests.support.storage.azure import mocked_azure_blob_storage

from .utils import IntegrationTestCase, discover_test_cases, list_integration_files, make_case_readers


def test_discover_pairs_sources_and_expected_by_number():
    """Sources pair with their expected output by case number, plus one glob case per type."""
    cases = discover_test_cases(
        ["simulation/source2.mat", "simulation/expected1.hdf5", "simulation/source1.mat", "simulation/expected2.hdf5"]
    )

    assert cases == (
        IntegrationTestCase(
            type="simulation", name="1", source_files=("source1.mat",), expected_files=("expected1.hdf5",)
        ),
        IntegrationTestCase(
            type="simulation", name="2", source_files=("source2.mat",), expected_files=("expected2.hdf5",)
        ),
        IntegrationTestCase(
            type="simulation",
            name="glob",
            source_files=("source1.mat", "source2.mat"),
            expected_files=("expected1.hdf5", "expected2.hdf5"),
        ),
    )


def test_discover_skips_source_without_expected_output():
    """Sources without matching expected output are ignored."""
    cases = discover_test_cases(["simulation/source1.mat", "simulation/expected1.hdf5", "simulation/source2.mat"])

    assert [case.id for case in cases] == ["simulation_1", "simulation_glob"]
    assert cases[0].source_files == ("source1.mat",)


def test_discover_ignores_top_level_files_and_groups_by_type():
    """Top-level files are ignored; each type subfolder gets its own cases."""
    cases = discover_test_cases(
        [
            "readme.txt",
            "simulation/source1.mat",
            "simulation/expected1.hdf5",
            "edevis/source1.mat",
            "edevis/expected1.hdf5",
        ]
    )

    assert [case.id for case in cases] == ["edevis_1", "edevis_glob", "simulation_1", "simulation_glob"]


def test_discover_pairs_alphanumeric_case_numbers():
    """Case numbers may contain letters; pairing follows the source file name."""
    cases = discover_test_cases(["simulation/sourceA.mat", "simulation/expectedA.hdf5"])

    assert [case.id for case in cases] == ["simulation_A", "simulation_glob"]
    assert cases[0].expected_files == ("expectedA.hdf5",)


def test_discover_empty_listing_gives_no_cases():
    """An empty test data listing yields no cases."""
    assert discover_test_cases([]) == ()


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["simulation/source1.mat", "simulation/expected1.hdf5"],
    ],
)
def test_discover_cases_have_readable_ids(paths: list[str]):
    """Every discovered case exposes a stable id of type and name."""
    for case in discover_test_cases(paths):
        assert case.id == f"{case.type}_{case.name}"


def test_discover_ignores_nested_files():
    """Archive subfolders cannot create duplicate cases or supply expected files."""
    cases = discover_test_cases(
        [
            "simulation/source1.mat",
            "simulation/expected1.hdf5",
            "simulation/archive/source1.mat",
            "simulation/archive/expected1.hdf5",
            "simulation/source2.mat",
            "simulation/archive/expected2.hdf5",
        ]
    )

    assert [case.id for case in cases] == ["simulation_1", "simulation_glob"]
    assert cases[1].source_files == ("source1.mat",)
    assert cases[1].expected_files == ("expected1.hdf5",)


@pytest.mark.parametrize("prefix", ["", "test-data"])
def test_remote_listing_and_readers_respect_folder_boundaries(prefix: str):
    """Sibling types and nested archives cannot enter a source/expected reader pair."""
    config = RemoteTestConfig("https://test.blob.core.windows.net", "fake", "test-container", prefix)
    root = f"{prefix}/" if prefix else ""
    paths = [
        "integration/simulation/source1.mat",
        "integration/simulation/expected1.hdf5",
        "integration/simulation_large/source1.mat",
        "integration/simulation_large/expected1.hdf5",
        "integration/simulation/archive/source1.mat",
        "integration/simulation/archive/expected1.hdf5",
        "integration-archive/simulation/source1.mat",
        "integration-archive/simulation/expected1.hdf5",
    ]
    with mocked_azure_blob_storage() as storage:
        storage.create_container(config.container_name)
        for path in paths:
            storage.upload_blob(config.container_name, root + path, b"data")

        listed = list_integration_files(config)
        assert sorted(listed) == sorted(path.removeprefix("integration/") for path in paths[:6])
        cases = discover_test_cases(listed)
        assert [case.id for case in cases] == [
            "simulation_1",
            "simulation_glob",
            "simulation_large_1",
            "simulation_large_glob",
        ]
        for case in cases:
            source, expected = make_case_readers(config, case)
            uri_prefix = f"az://test-container/{root}integration/{case.type}/"
            assert source.files == [uri_prefix + "source1.mat"]
            assert expected.files == [uri_prefix + "expected1.hdf5"]


def test_case_readers_reject_duplicate_listing_entries():
    """Duplicate file entries must not be hidden by set-based validation."""
    config = RemoteTestConfig("https://test.blob.core.windows.net", "fake", "test-container", "")
    case = IntegrationTestCase("simulation", "1", ("source1.mat",), ("expected1.hdf5",))
    with patch(
        "pythermondt.readers.AzureBlobReader.files",
        new_callable=PropertyMock,
        return_value=["az://test-container/integration/simulation/source1.mat"] * 2,
    ):
        with pytest.raises(RuntimeError, match="Test data mismatch"):
            make_case_readers(config, case)
