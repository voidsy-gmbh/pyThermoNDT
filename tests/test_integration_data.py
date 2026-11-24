import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from azure.core.exceptions import ClientAuthenticationError

import tests.support.integration_data as integration_data
from pythermondt.data import DataContainer
from tests.support.integration_data import FilePair, PairingError, folder_readers, load_file_pairs, pair_files
from tests.support.remote_data import RemoteTestConfig
from tests.support.storage.azure import mocked_azure_blob_storage

ROOT = "az://test-container/integration/"


def _config(prefix: str = "") -> RemoteTestConfig:
    return RemoteTestConfig(
        account_url="https://test.blob.core.windows.net", container_name="test-container", prefix=prefix
    )


def _measurement(value: int) -> bytes:
    container = DataContainer()
    container.add_group("/", "Data")
    container.add_dataset("/Data", "Tdata", torch.tensor([value]))
    return container.serialize_to_hdf5().getvalue()


def _pair(folder: str, name: str, has_expected: bool = True) -> FilePair:
    expected_uri = f"{ROOT}{folder}/expected{name.split('.')[0]}.hdf5"
    return FilePair(folder, f"{ROOT}{folder}/source{name}", expected_uri, has_expected)


def test_pair_files_by_case_name():
    """Sources pair with expected outputs by case name, including alphanumeric names."""
    paths = [
        "simulation/source2.mat",
        "simulation/expected1.hdf5",
        "simulation/source1.mat",
        "simulation/expected2.hdf5",
    ]
    paths += ["edevis/sourceA.di", "edevis/expectedA.hdf5"]

    pairs = pair_files(paths, ROOT)

    assert pairs == (_pair("edevis", "A.di"), _pair("simulation", "1.mat"), _pair("simulation", "2.mat"))
    assert [pair.id for pair in pairs] == ["edevis/sourceA.di", "simulation/source1.mat", "simulation/source2.mat"]


def test_pair_files_marks_sources_without_expected_output():
    """New sources stay visible; an expected output in a nested folder does not count."""
    paths = ["simulation/source1.mat", "simulation/archive/expected1.hdf5", "edevis/source2.di"]

    assert pair_files(paths, ROOT) == (
        _pair("edevis", "2.di", has_expected=False),
        _pair("simulation", "1.mat", has_expected=False),
    )


def test_pair_files_rejects_sources_sharing_a_case_name():
    """Two sources for one expected output would compare and upload the wrong file; name both."""
    paths = ["edevis/source1.di", "edevis/source1.OTvis", "edevis/expected1.hdf5"]

    with pytest.raises(PairingError, match=r"edevis: source1\.OTvis, source1\.di share expected1\.hdf5"):
        pair_files(paths, ROOT)


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["simulation/expected1.hdf5"],
        ["readme.txt", "source1.mat", "expected1.hdf5"],
        ["simulation/archive/source1.mat", "simulation/archive/expected1.hdf5"],
    ],
    ids=["empty", "expected-only", "top-level", "nested"],
)
def test_pair_files_ignores_misplaced_files(paths: list[str]):
    """Only sources that are direct children of parser folders create pairs."""
    assert pair_files(paths, ROOT) == ()


@pytest.mark.parametrize("prefix", ["", "test-data"])
def test_load_file_pairs_stays_inside_integration_root(prefix: str):
    """Only direct parser folders below the integration root form pairs; siblings and archives are ignored."""
    config = _config(prefix)
    root = f"{prefix}/integration/" if prefix else "integration/"
    paths = ["edevis/expected1.hdf5", "edevis/source1.di", "simulation/expected1.hdf5", "simulation/source1.mat"]
    ignored = ["simulation/archive/source2.mat", "simulation/archive/expected2.hdf5"]
    with mocked_azure_blob_storage() as storage:
        storage.create_container(config.container_name)
        for path in [root + path for path in paths + ignored] + [
            root.replace("integration/", "integration-old/") + paths[0]
        ]:
            storage.upload_blob(config.container_name, path, b"data")

        _, pairs = load_file_pairs(config)

    root_uri = f"az://test-container/{root}"
    assert pairs == (
        FilePair("edevis", root_uri + "edevis/source1.di", root_uri + "edevis/expected1.hdf5"),
        FilePair("simulation", root_uri + "simulation/source1.mat", root_uri + "simulation/expected1.hdf5"),
    )


def _upload_pairs(storage, folders: dict[str, list[str]]) -> tuple[FilePair, ...]:
    for folder, names in folders.items():
        for name in names:
            storage.upload_blob("test-container", f"integration/{folder}/source{name}", _measurement(1))
            storage.upload_blob(
                "test-container", f"integration/{folder}/expected{name.split('.')[0]}.hdf5", _measurement(1)
            )
    return load_file_pairs(_config())[1]


def test_folder_readers_list_one_folder_in_pair_order():
    """Readers contain only complete pairs of their folder, in matching order."""
    with mocked_azure_blob_storage() as storage:
        storage.create_container("test-container")
        storage.upload_blob("test-container", "integration/simulation/source3.mat", _measurement(1))
        pairs = _upload_pairs(storage, {"simulation": ["1.mat", "2.mat"], "simulation_large": ["1.mat"]})

        source_reader, expected_reader = folder_readers(_config(), pairs, "simulation")

        folder_pairs = [pair for pair in pairs if pair.folder == "simulation" and pair.has_expected]
        assert len(folder_pairs) == 2
        assert source_reader.file_uris == [pair.source_uri for pair in folder_pairs]
        assert expected_reader.file_uris == [pair.expected_uri for pair in folder_pairs]


def test_folder_readers_reject_pairs_in_different_order():
    """Readers list files sorted, so pairs in another order would compare unrelated files."""
    with mocked_azure_blob_storage() as storage:
        storage.create_container("test-container")
        pairs = _upload_pairs(storage, {"simulation": ["1.mat", "2.mat"]})

        with pytest.raises(RuntimeError, match="does not match pairs"):
            folder_readers(_config(), pairs[::-1], "simulation")


@pytest.mark.parametrize(
    "error", [ClientAuthenticationError("secret-token"), RuntimeError("secret-token"), OSError("secret-token")]
)
def test_discovery_failure_returns_safe_error(error: Exception):
    """Cloud failures become an error message without exception text."""
    integration_data.collect_integration_data.cache_clear()
    with (
        patch.object(integration_data, "cloud_skip_reason", return_value=None),
        patch.object(integration_data, "load_remote_config", side_effect=error),
    ):
        data = integration_data.collect_integration_data()
    integration_data.collect_integration_data.cache_clear()

    assert data.pairs == ()
    assert data.error is not None and type(error).__name__ in data.error
    assert "secret-token" not in data.error


def test_pairing_error_is_reported_with_file_names():
    """Pairing errors contain only blob names, so discovery reports them instead of a generic message."""
    integration_data.collect_integration_data.cache_clear()
    with (
        patch.object(integration_data, "cloud_skip_reason", return_value=None),
        patch.object(integration_data, "load_remote_config"),
        patch.object(integration_data, "load_file_pairs", side_effect=PairingError("edevis: a, b share c")),
    ):
        data = integration_data.collect_integration_data()
    integration_data.collect_integration_data.cache_clear()

    assert data.error == "edevis: a, b share c"


def test_discovery_without_pairs_returns_error():
    """An empty or unpaired remote corpus cannot silently skip integration coverage."""
    integration_data.collect_integration_data.cache_clear()
    with (
        patch.object(integration_data, "cloud_skip_reason", return_value=None),
        patch.object(integration_data, "load_remote_config"),
        patch.object(integration_data, "load_file_pairs", return_value=(None, ())),
    ):
        data = integration_data.collect_integration_data()
    integration_data.collect_integration_data.cache_clear()

    assert data.error == "No paired integration test files found in remote test data."


@pytest.mark.parametrize("scenario", ["error", "empty"])
def test_discovery_failure_does_not_block_offline_tests(tmp_path: Path, scenario: str):
    """A real pytest session fails cloud tests but still runs offline tests and collects without errors."""
    failure = "raise RuntimeError('secret-token')" if scenario == "error" else "return None, ()"
    plugin = tmp_path / "discovery_failure_plugin.py"
    plugin.write_text(
        "def pytest_configure(config):\n"
        "    import tests.support.integration_data as data\n"
        "    data.cloud_skip_reason = lambda: None\n"
        "    data.load_remote_config = lambda: None\n"
        "    def load(config):\n"
        f"        {failure}\n"
        "    data.load_file_pairs = load\n",
        encoding="utf-8",
    )
    offline_test = tmp_path / "test_offline_sentinel.py"
    offline_test.write_text("def test_offline_sentinel():\n    assert True\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        REMOTE_TESTS="1",
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(Path.cwd()), env.get("PYTHONPATH", "")]),
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "discovery_failure_plugin", "tests/integration", str(offline_test)]
        + ["--benchmark-skip", "-q", "-p", "no:cacheprovider"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert "1 passed, 2 errors" in output  # one placeholder per cloud test module
    assert "secret-token" not in output
    assert "error during collection" not in output
