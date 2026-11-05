from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pythermondt import settings
from pythermondt.data import DataContainer
from pythermondt.io import AzureBlobBackend
from tests.support.remote_data import RemoteTestConfig
from tests.support.storage.azure import mocked_azure_blob_storage

from .utils import IntegrationTestCase, make_case_readers


@pytest.fixture(scope="session")
def remote_config() -> RemoteTestConfig:
    """Use mocked Azure storage for cache regression tests, even in live runs."""
    return RemoteTestConfig(
        account_url="https://test.blob.core.windows.net",
        connection_string="fake",
        container_name="test-container",
        prefix="test-data",
    )


def _measurement(value: int) -> bytes:
    container = DataContainer()
    container.add_group("/", "Data")
    container.add_dataset("/Data", "Tdata", torch.tensor([value]))
    return container.serialize_to_hdf5().getvalue()


def test_case_readers_refresh_changed_blobs_and_reuse_unchanged_cache(tmp_path: Path, remote_config: RemoteTestConfig):
    """Synchronize both source and reference blobs without re-downloading unchanged data."""
    case = IntegrationTestCase("simulation", "1", ("source1.hdf5",), ("expected1.hdf5",))
    paths = ["test-data/integration/simulation/" + name for name in ("source1.hdf5", "expected1.hdf5")]
    with (
        patch.object(settings, "download_dir", str(tmp_path / "persistent-cache")),
        mocked_azure_blob_storage() as storage,
    ):
        storage.create_container(remote_config.container_name)
        for path in paths:
            storage.upload_blob(remote_config.container_name, path, _measurement(1))
        for reader in make_case_readers(remote_config, case):
            assert reader[0].get_dataset("/Data/Tdata").item() == 1

        for path in paths:
            storage.upload_blob(remote_config.container_name, path, _measurement(2))

        with patch.object(storage, "download_blob", wraps=storage.download_blob) as download:
            for reader in make_case_readers(remote_config, case):
                assert reader[0].get_dataset("/Data/Tdata").item() == 2
            assert download.call_count == 2

            for reader in make_case_readers(remote_config, case):
                assert reader[0].get_dataset("/Data/Tdata").item() == 2
            assert download.call_count == 2


def test_case_readers_sync_only_selected_blobs(tmp_path: Path, remote_config: RemoteTestConfig):
    """A case must not synchronize unrelated entries from the shared reader cache."""
    prefix = "test-data/integration/simulation/"
    with patch.object(settings, "download_dir", str(tmp_path)), mocked_azure_blob_storage() as storage:
        storage.create_container(remote_config.container_name)
        for name in ("source1.hdf5", "expected1.hdf5", "source2.hdf5", "expected2.hdf5"):
            storage.upload_blob(remote_config.container_name, prefix + name, _measurement(1))

        other_case = IntegrationTestCase("simulation", "2", ("source2.hdf5",), ("expected2.hdf5",))
        for reader in make_case_readers(remote_config, other_case):
            reader[0]

        case = IntegrationTestCase("simulation", "1", ("source1.hdf5",), ("expected1.hdf5",))
        with patch.object(
            AzureBlobBackend, "get_file_identity", autospec=True, side_effect=AzureBlobBackend.get_file_identity
        ) as identity:
            make_case_readers(remote_config, case)

        assert {call.args[1] for call in identity.call_args_list} == {
            "az://test-container/" + prefix + "source1.hdf5",
            "az://test-container/" + prefix + "expected1.hdf5",
        }
