import pytest

from pythermondt.readers import AzureBlobReader
from tests.support.remote_data import RemoteTestConfig, load_remote_config

from .utils import CaseReaders, IntegrationTestCase, make_case_readers


@pytest.fixture(scope="session")
def remote_config() -> RemoteTestConfig:
    """Connection settings for the remote integration test data."""
    return load_remote_config()


@pytest.fixture
def case_readers(request: pytest.FixtureRequest) -> CaseReaders:
    """Return a factory that creates the cached source and expected readers for one test case."""

    def _make(test_case: IntegrationTestCase) -> tuple[AzureBlobReader, AzureBlobReader]:
        if test_case.discovery_error is not None:
            pytest.fail(test_case.discovery_error, pytrace=False)
        remote_config = request.getfixturevalue("remote_config")
        return make_case_readers(remote_config, test_case)

    return _make
