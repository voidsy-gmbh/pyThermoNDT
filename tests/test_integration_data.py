import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from azure.core.exceptions import ClientAuthenticationError

import pythermondt.io.backends.azure_backend as azure_backend
import tests.support.integration_data as integration_data
from pythermondt import settings
from pythermondt.data import DataContainer
from tests.support.integration_data import FilePair, folder_readers, load_file_pairs, pair_files
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


def _pair(folder: str, name: str) -> FilePair:
    return FilePair(folder, f"{ROOT}{folder}/source{name}", f"{ROOT}{folder}/expected{name.split('.')[0]}.hdf5")


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


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["simulation/source1.mat"],
        ["readme.txt", "source1.mat", "expected1.hdf5"],
        ["simulation/archive/source1.mat", "simulation/archive/expected1.hdf5"],
        ["simulation/source1.mat", "simulation/archive/expected1.hdf5"],
    ],
    ids=["empty", "unpaired", "top-level", "nested", "nested-expected"],
)
def test_pair_files_ignores_incomplete_or_misplaced_files(paths: list[str]):
    """Only direct children of parser folders can form pairs."""
    assert pair_files(paths, ROOT) == ()


@pytest.mark.parametrize("prefix", ["", "test-data"])
def test_shared_reader_syncs_pairs_once_and_folder_readers_reuse_cache(tmp_path: Path, prefix: str):
    """One sync downloads only paired files; later reads use the cache until a blob changes."""
    config = _config(prefix)
    root = f"{prefix}/integration/" if prefix else "integration/"
    paired = [
        f"{folder}/{side}1.{ext}"
        for folder in ("edevis", "simulation")
        for side, ext in (("expected", "hdf5"), ("source", "hdf5"))
    ]
    ignored = [
        "simulation/archive/source1.hdf5",
        "simulation/source2.hdf5",
        "../integration-archive/edevis/source1.hdf5",
    ]
    with patch.object(settings, "download_dir", str(tmp_path)), mocked_azure_blob_storage() as storage:
        storage.create_container(config.container_name)
        for path in paired + ignored:
            storage.upload_blob(config.container_name, os.path.normpath(root + path), _measurement(1))

        reader, pairs = load_file_pairs(config)
        uris = [uri for pair in pairs for uri in (pair.source_uri, pair.expected_uri)]
        assert [pair.id for pair in pairs] == ["edevis/source1.hdf5", "simulation/source1.hdf5"]

        with patch.object(storage, "download_blob", wraps=storage.download_blob) as download:
            reader.sync(file_paths=uris)
            assert download.call_count == len(paired)

            for pair in pairs:
                assert reader.read_file(pair.source_uri).get_dataset("/Data/Tdata").item() == 1
                for folder_reader in folder_readers(config, pairs, pair.folder):
                    assert folder_reader[0].get_dataset("/Data/Tdata").item() == 1
            assert download.call_count == len(paired)

            storage.upload_blob(config.container_name, root + "edevis/source1.hdf5", _measurement(2))
            reader.sync(file_paths=uris)
            assert download.call_count == len(paired) + 1


def test_shared_reader_uses_default_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The shared reader uses the credential chain, even if an obsolete SAS environment value exists."""
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "obsolete-secret")
    with (
        patch.object(settings, "download_dir", str(tmp_path)),
        mocked_azure_blob_storage() as storage,
        patch.object(azure_backend, "DefaultAzureCredential") as credential,
        patch.object(azure_backend.BlobServiceClient, "from_connection_string") as from_connection_string,
    ):
        storage.create_container("test-container")
        load_file_pairs(_config())

    credential.assert_called_once()
    from_connection_string.assert_not_called()


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
    assert "1 passed" in output
    assert "secret-token" not in output
    assert "error during collection" not in output
