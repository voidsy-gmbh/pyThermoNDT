import pytest

from tests.support.storage import FILE_SCENARIOS, TEST_FILES, StorageTestContext, mocked_azure_blob_storage


@pytest.fixture()
def azure_mock():
    """Create mocked Azure Blob Storage for specialized backend tests."""
    with mocked_azure_blob_storage() as storage:
        yield storage


@pytest.fixture(params=TEST_FILES.items(), ids=lambda item: item[0])
def test_file(request: pytest.FixtureRequest, storage_context: StorageTestContext) -> tuple[str, bytes]:
    """Prepare one test file and return its URI and content."""
    name, content = request.param
    return storage_context.prepare_file(name, content), content


@pytest.fixture(params=FILE_SCENARIOS.items(), ids=lambda item: item[0])
def test_files_scenario(request: pytest.FixtureRequest, storage_context: StorageTestContext) -> dict[str, str]:
    """Prepare a deterministic multi-file scenario."""
    _, files = request.param
    return storage_context.prepare_files(dict(sorted(files.items())))
