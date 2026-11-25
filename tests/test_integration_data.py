import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from azure.core.exceptions import ClientAuthenticationError

import tests.support.integration_data as integration_data
from pythermondt.config import settings
from pythermondt.data import DataContainer
from tests.support.integration_data import FilePair, PairingError, folder_datasets, load_file_pairs, pair_files
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


def test_pair_files_by_full_source_filename():
    """Original names stay visible and same-stem measurements get separate expected outputs."""
    paths = [
        "edevis/Panel A.run03.di.expected.hdf5",
        "edevis/Panel A.run03.ThermoVis",
        "edevis/Panel A.run03.di",
        "edevis/Panel A.run03.ThermoVis.expected.hdf5",
        "simulation/panel_run02.mat",
        "simulation/panel_run02.mat.expected.hdf5",
    ]

    pairs = pair_files(paths, ROOT)

    assert pairs == (
        FilePair(
            "edevis", ROOT + "edevis/Panel A.run03.ThermoVis", ROOT + "edevis/Panel A.run03.ThermoVis.expected.hdf5"
        ),
        FilePair("edevis", ROOT + "edevis/Panel A.run03.di", ROOT + "edevis/Panel A.run03.di.expected.hdf5"),
        FilePair("simulation", ROOT + "simulation/panel_run02.mat", ROOT + "simulation/panel_run02.mat.expected.hdf5"),
    )
    assert [pair.id for pair in pairs] == [
        "edevis/Panel A.run03.ThermoVis",
        "edevis/Panel A.run03.di",
        "simulation/panel_run02.mat",
    ]


def test_pair_files_does_not_pair_legacy_output():
    """Only the new suffix identifies an expected output; other HDF5 files are sources."""
    paths = ["simulation/source1.mat", "simulation/expected1.hdf5"]

    assert pair_files(paths, ROOT) == (
        FilePair(
            "simulation", ROOT + "simulation/expected1.hdf5", ROOT + "simulation/expected1.hdf5.expected.hdf5", False
        ),
        FilePair("simulation", ROOT + "simulation/source1.mat", ROOT + "simulation/source1.mat.expected.hdf5", False),
    )


def test_pair_files_marks_sources_without_expected_output():
    """New sources stay visible; an expected output in a nested folder does not count."""
    paths = ["simulation/panel_run03.mat", "simulation/archive/panel_run03.mat.expected.hdf5"]

    assert pair_files(paths, ROOT) == (
        FilePair(
            "simulation", ROOT + "simulation/panel_run03.mat", ROOT + "simulation/panel_run03.mat.expected.hdf5", False
        ),
    )


@pytest.mark.parametrize("path", ["../source1.mat", "./source1.mat", "edevis/..", "edevis/source1\\..\\x.mat"])
def test_pair_files_rejects_paths_that_escape_their_folder(path: str):
    """Folder and file names are reused as local paths, so names that leave their folder are rejected."""
    with pytest.raises(PairingError, match="Unsafe integration file path"):
        pair_files([path], ROOT)


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["simulation/panel_run03.mat.expected.hdf5"],
        ["panel_run03.mat", "panel_run03.mat.expected.hdf5"],
        ["simulation/archive/panel_run03.mat", "simulation/archive/panel_run03.mat.expected.hdf5"],
        ["simulation/source_notes.txt", "edevis/panel_run03.thermovis"],
    ],
    ids=["empty", "expected-only", "top-level", "nested", "unsupported"],
)
def test_pair_files_ignores_misplaced_files(paths: list[str]):
    """Only sources that are direct children of parser folders create pairs."""
    assert pair_files(paths, ROOT) == ()


@pytest.mark.parametrize("prefix", ["", "test-data"])
def test_load_file_pairs_stays_inside_integration_root(prefix: str):
    """Only direct parser folders below the integration root form pairs; siblings and archives are ignored."""
    config = _config(prefix)
    root = f"{prefix}/integration/" if prefix else "integration/"
    paths = [
        "edevis/panel_run03.ThermoVis.expected.hdf5",
        "edevis/panel_run03.ThermoVis",
        "simulation/panel_run02.mat.expected.hdf5",
        "simulation/panel_run02.mat",
    ]
    ignored = ["simulation/archive/panel_run03.mat", "simulation/archive/panel_run03.mat.expected.hdf5"]
    with mocked_azure_blob_storage() as storage:
        storage.create_container(config.container_name)
        for path in [root + path for path in paths + ignored] + [
            root.replace("integration/", "integration-old/") + "edevis/panel_run04.ThermoVis"
        ]:
            storage.upload_blob(config.container_name, path, b"data")

        _, pairs = load_file_pairs(config)

    root_uri = f"az://test-container/{root}"
    assert pairs == (
        FilePair("edevis", root_uri + paths[1], root_uri + paths[0]),
        FilePair("simulation", root_uri + paths[3], root_uri + paths[2]),
    )


def test_folder_datasets_align_complete_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The suffix reverses listing order for these names, but datasets still yield matching samples."""
    monkeypatch.setattr(settings, "download_dir", str(tmp_path))
    paths = {
        "hdf5/Run 03.h5": 3,
        "hdf5/Run 03.h5.expected.hdf5": 3,
        "hdf5/Run 03.h5-backup.h5": 4,
        "hdf5/Run 03.h5-backup.h5.expected.hdf5": 4,
        "hdf5/Run 05.h5": 5,
        "other/Run 06.h5": 6,
        "other/Run 06.h5.expected.hdf5": 6,
    }
    with mocked_azure_blob_storage() as storage:
        storage.create_container("test-container")
        for path, value in paths.items():
            storage.upload_blob("test-container", f"integration/{path}", _measurement(value))
        _, pairs = load_file_pairs(_config())

        source, expected = folder_datasets(_config(), pairs, "hdf5")

        assert source.files == [ROOT + "hdf5/Run 03.h5", ROOT + "hdf5/Run 03.h5-backup.h5"]
        assert expected.files == [
            ROOT + "hdf5/Run 03.h5.expected.hdf5",
            ROOT + "hdf5/Run 03.h5-backup.h5.expected.hdf5",
        ]
        for dataset in (source, expected):
            assert [item.get_dataset("/Data/Tdata").item() for item in dataset] == [3, 4]


def test_folder_datasets_reject_missing_files():
    """A baseline that disappears after discovery must not silently drop part of the dataset."""
    source = "simulation/panel_run03.mat"
    expected = "simulation/panel_run03.mat.expected.hdf5"
    pairs = (FilePair("simulation", ROOT + source, ROOT + expected),)
    with mocked_azure_blob_storage() as storage:
        storage.create_container("test-container")
        storage.upload_blob("test-container", f"integration/{source}", b"data")

        with pytest.raises(RuntimeError, match=r"did not list paired files: .*panel_run03\.mat\.expected\.hdf5"):
            folder_datasets(_config(), pairs, "simulation")


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
    message = "Unsafe integration file path '../panel_run03.mat'. Rename the blob."
    integration_data.collect_integration_data.cache_clear()
    with (
        patch.object(integration_data, "cloud_skip_reason", return_value=None),
        patch.object(integration_data, "load_remote_config"),
        patch.object(integration_data, "load_file_pairs", side_effect=PairingError(message)),
    ):
        data = integration_data.collect_integration_data()
    integration_data.collect_integration_data.cache_clear()

    assert data.error == message


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
