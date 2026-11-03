import pytest

from pythermondt.readers import AzureBlobReader
from tests.support.remote_data import RemoteTestConfig, load_remote_config

from .utils import CaseReaders, IntegrationTestCase, make_case_readers


@pytest.fixture(scope="session")
def remote_config() -> RemoteTestConfig:
    """Connection settings for the remote integration test data."""
    return load_remote_config()


@pytest.fixture
def case_readers(remote_config: RemoteTestConfig) -> CaseReaders:
    """Return a factory that creates the cached source and expected readers for one test case."""

    def _make(test_case: IntegrationTestCase) -> tuple[AzureBlobReader, AzureBlobReader]:
        return make_case_readers(remote_config, test_case)

    return _make
