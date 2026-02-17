"""Azure Blob Storage-specific backend tests."""

from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import AzureError

from pythermondt.io import AzureBlobBackend, IOPathWrapper


@pytest.fixture
def azure_backend(azure_mock):
    """Create mocked Azure Blob backend for testing."""
    azure_mock.create_container("test-container")

    backend = AzureBlobBackend(
        account_url="https://test.blob.core.windows.net",
        container_name="test-container",
        prefix="test/",
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake==;EndpointSuffix=core.windows.net",
    )

    # Write a test file
    backend.write_file(IOPathWrapper(b"test content"), "test/sample.txt")

    yield backend
    backend.close()


@pytest.fixture
def azure_backend_with_directory(azure_backend, azure_mock):
    """Azure backend with a directory marker in storage."""
    # Create a "directory" marker (blob name ending with /)
    azure_mock.upload_blob("test-container", "test/subdir/", b"")

    # Create a file in that directory
    azure_backend.write_file(IOPathWrapper(b"file content"), "test/subdir/file.txt")

    return azure_backend


def test_init_with_default_credential(azure_mock):
    """Test initialization with DefaultAzureCredential when no connection_string or credential provided."""
    azure_mock.create_container("test-container")

    with (
        patch("pythermondt.io.backends.azure_backend.DefaultAzureCredential") as mock_default_cred,
        patch("pythermondt.io.backends.azure_backend.BlobServiceClient") as mock_client_class,
    ):
        mock_cred_instance = MagicMock()
        mock_default_cred.return_value = mock_cred_instance
        mock_client_class.return_value = MagicMock()

        backend = AzureBlobBackend(
            account_url="https://test.blob.core.windows.net",
            container_name="test-container",
        )

        mock_default_cred.assert_called_once()
        mock_client_class.assert_called_once_with("https://test.blob.core.windows.net", credential=mock_cred_instance)
        backend.close()


def test_container_name_property(azure_backend):
    """Test container_name property returns the configured container name."""
    assert azure_backend.container_name == "test-container"


def test_prefix_property(azure_backend):
    """Test prefix property returns the configured prefix with trailing slash stripped."""
    assert azure_backend.prefix == "test"


def test_write_file_azure_error(azure_backend):
    """Test that AzureError during upload is wrapped in RuntimeError."""
    data = IOPathWrapper(b"test content")

    with patch.object(azure_backend._AzureBlobBackend__client, "get_blob_client") as mock_get_blob_client:
        mock_blob = MagicMock()
        mock_blob.upload_blob.side_effect = AzureError("Upload failed")
        mock_get_blob_client.return_value = mock_blob

        with pytest.raises(RuntimeError, match="Failed to upload blob"):
            azure_backend.write_file(data, "test/new_file.txt")


def test_get_file_list_skips_directories(azure_backend_with_directory):
    """Test that get_file_list skips directory markers (blobs ending with /)."""
    files = azure_backend_with_directory.get_file_list()

    assert all(not f.endswith("/") for f in files)
    assert "az://test-container/test/subdir/" not in files
    assert "az://test-container/test/subdir/file.txt" in files


def test_parse_input_azure_uri(azure_backend):
    """Test parsing full Azure URI."""
    container, blob_name = azure_backend._parse_input("az://my-container/path/to/file.txt")
    assert container == "my-container"
    assert blob_name == "path/to/file.txt"


def test_parse_input_blob_name_only(azure_backend):
    """Test parsing blob name without URI uses default container."""
    container, blob_name = azure_backend._parse_input("path/to/file.txt")
    assert container == "test-container"
    assert blob_name == "path/to/file.txt"


def test_to_url(azure_backend):
    """Test URL construction."""
    url = azure_backend._to_url("my-container", "path/to/file.txt")
    assert url == "az://my-container/path/to/file.txt"
