import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from azure.core.exceptions import ClientAuthenticationError

from pythermondt.readers import AzureBlobReader
from tests.support.remote_data import RemoteTestConfig
from tests.support.storage.azure import mocked_azure_blob_storage

from .utils import (
    IntegrationTestCase,
    discover_test_cases,
    integration_test_cases,
    list_integration_files,
    make_case_readers,
)


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
    config = RemoteTestConfig(
        account_url="https://test.blob.core.windows.net",
        connection_string="fake",
        container_name="test-container",
        prefix=prefix,
    )
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
    with mocked_azure_blob_storage() as storage, patch.object(AzureBlobReader, "sync"):
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
    config = RemoteTestConfig(
        account_url="https://test.blob.core.windows.net",
        connection_string="fake",
        container_name="test-container",
        prefix="",
    )
    case = IntegrationTestCase("simulation", "1", ("source1.mat",), ("expected1.hdf5",))
    with patch(
        "pythermondt.readers.AzureBlobReader.files",
        new_callable=PropertyMock,
        return_value=["az://test-container/integration/simulation/source1.mat"] * 2,
    ):
        with pytest.raises(RuntimeError, match="Test data mismatch"):
            make_case_readers(config, case)


@pytest.mark.parametrize(
    "error", [ClientAuthenticationError("secret-token"), RuntimeError("secret-token"), OSError("secret-token")]
)
def test_discovery_failure_returns_safe_placeholder(error: Exception):
    """Cloud failures leave collection intact without exposing exception text."""
    with (
        patch("tests.integration.utils.cloud_skip_reason", return_value=None),
        patch("tests.integration.utils.load_remote_config", side_effect=error),
    ):
        cases = integration_test_cases()

    assert len(cases) == 1
    assert cases[0].discovery_error is not None
    assert type(error).__name__ in cases[0].discovery_error
    assert "secret-token" not in repr(cases)


@pytest.mark.parametrize("paths", [[], ["simulation/source1.mat"]])
def test_enabled_discovery_without_pairs_returns_failure(paths: list[str]):
    """An empty or unpaired remote corpus cannot silently skip integration coverage."""
    with (
        patch("tests.integration.utils.cloud_skip_reason", return_value=None),
        patch("tests.integration.utils.load_remote_config"),
        patch("tests.integration.utils.list_integration_files", return_value=paths),
    ):
        cases = integration_test_cases()

    assert len(cases) == 1
    assert cases[0].discovery_error == "No paired integration test files found in remote test data."


@pytest.mark.parametrize("scenario", ["error", "empty"])
def test_discovery_failure_does_not_block_offline_tests(tmp_path: Path, scenario: str):
    """A real pytest session fails cloud tests but still executes offline tests."""
    plugin = tmp_path / "discovery_failure_plugin.py"
    plugin.write_text(
        "def pytest_configure(config):\n"
        "    import tests.integration.utils as utils\n"
        "    utils.cloud_skip_reason = lambda: None\n"
        "    utils.load_remote_config = lambda: None\n"
        + (
            "    def fail(config):\n"
            "        raise RuntimeError('secret-token')\n"
            "    utils.list_integration_files = fail\n"
            if scenario == "error"
            else "    utils.list_integration_files = lambda config: []\n"
        ),
        encoding="utf-8",
    )
    offline_test = tmp_path / "test_offline_sentinel.py"
    offline_test.write_text("def test_offline_sentinel():\n    assert True\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        REMOTE_TESTS="1",
        GITHUB_EVENT_NAME="push",
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(Path.cwd()), env.get("PYTHONPATH", "")]),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "discovery_failure_plugin",
            "tests/integration/test_integration.py",
            str(offline_test),
            "--benchmark-skip",
            "-q",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert "7 failed, 1 passed" in output
    assert "secret-token" not in output
    assert "error during collection" not in output


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
