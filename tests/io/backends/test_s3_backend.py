"""S3-specific backend tests."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from moto import mock_aws

from pythermondt.io import FileInfo, IOPathWrapper, S3Backend, S3ClientOptions


@pytest.fixture
def s3_backend(s3_client):
    """Create mocked S3 backend for testing."""
    with mock_aws():
        # Setup S3 bucket
        s3_client.create_bucket(Bucket="test-bucket")

        # Create backend
        backend = S3Backend(bucket="test-bucket", prefix="test/")

        # Write a test file
        backend.write_file(IOPathWrapper(b"test content"), "test/sample.txt")

        yield backend
        backend.close()


@pytest.fixture
def s3_backend_with_directory(s3_backend):
    """S3 backend with a directory structure."""
    # Create a "directory" marker (key ending with /)
    s3_backend._S3Backend__client.put_object(Bucket="test-bucket", Key="test/subdir/", Body=b"")

    # Create a file in that directory
    s3_backend.write_file(IOPathWrapper(b"file content"), "test/subdir/file.txt")

    return s3_backend


def test_read_file_unexpected_error(s3_backend):
    """Test that unexpected S3 errors are re-raised."""
    error_response = {
        "Error": {"Code": "InternalError", "Message": "We encountered an internal error"},
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }

    with patch.object(s3_backend._S3Backend__client, "head_object") as mock_head:
        mock_head.side_effect = ClientError(error_response, "HeadObject")

        with pytest.raises(ClientError) as exc:
            s3_backend.read_file("test/sample.txt")

        err = exc.value.response["Error"]
        assert err["Code"] == "InternalError"
        assert err["Message"] == "We encountered an internal error"


def test_write_file_upload_error(s3_backend):
    """Test that S3 upload errors are wrapped in RuntimeError."""
    data = IOPathWrapper(b"test content")

    error_response = {
        "Error": {"Code": "ServiceUnavailable", "Message": "Service is unavailable"},
        "ResponseMetadata": {"HTTPStatusCode": 503},
    }

    with patch.object(s3_backend._S3Backend__client, "upload_fileobj") as mock_upload:
        mock_upload.side_effect = ClientError(error_response, "PutObject")

        with pytest.raises(RuntimeError) as exc:
            s3_backend.write_file(data, "test/new_file.txt")

        assert "Failed to upload file to S3" in str(exc.value)


def test_init_with_client_options_builds_botocore_config():
    """Test S3 client options are converted into botocore config."""
    session = MagicMock()
    session.client.return_value = MagicMock()
    client_options = S3ClientOptions(connect_timeout=10, read_timeout=20, max_pool_connections=30)

    backend = S3Backend(bucket="test-bucket", prefix="test/", session=session, client_options=client_options)

    session.client.assert_called_once()
    assert session.client.call_args.args == ("s3",)
    config = session.client.call_args.kwargs["config"]
    assert isinstance(config, Config)
    assert config.connect_timeout == 10  # type: ignore
    assert config.read_timeout == 20  # type: ignore
    assert config.max_pool_connections == 30  # type: ignore
    backend.close()


def test_exists_unexpected_error(s3_backend):
    """Test that unexpected errors in exists() are re-raised."""
    error_response = {
        "Error": {"Code": "InternalError", "Message": "Internal error"},
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }

    with patch.object(s3_backend._S3Backend__client, "head_object") as mock_head:
        mock_head.side_effect = ClientError(error_response, "HeadObject")

        with pytest.raises(ClientError) as exc:
            s3_backend.exists("test/sample.txt")

        err = exc.value.response["Error"]
        assert err["Code"] == "InternalError"
        assert err["Message"] == "Internal error"


def test_get_file_size_unexpected_error(s3_backend):
    """Test that unexpected errors in get_file_size() are re-raised."""
    error_response = {
        "Error": {"Code": "InternalError", "Message": "Internal error"},
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }

    with patch.object(s3_backend._S3Backend__client, "head_object") as mock_head:
        mock_head.side_effect = ClientError(error_response, "HeadObject")

        with pytest.raises(ClientError) as exc:
            s3_backend.get_file_size("test/sample.txt")

        err = exc.value.response["Error"]
        assert err["Code"] == "InternalError"
        assert err["Message"] == "Internal error"


def test_get_file_identity_unexpected_error(s3_backend):
    """Test that unexpected errors in get_file_identity() are re-raised."""
    error_response = {
        "Error": {"Code": "InternalError", "Message": "Internal error"},
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }

    with patch.object(s3_backend._S3Backend__client, "head_object") as mock_head:
        mock_head.side_effect = ClientError(error_response, "HeadObject")

        with pytest.raises(ClientError) as exc:
            s3_backend.get_file_identity("test/sample.txt")

        err = exc.value.response["Error"]
        assert err["Code"] == "InternalError"
        assert err["Message"] == "Internal error"


def test_get_file_identity_none_etag(s3_backend):
    """Test that a None ETag raises RuntimeError."""
    with patch.object(s3_backend._S3Backend__client, "head_object") as mock_head:
        mock_head.return_value = {"ContentLength": 12}  # No ETag key

        with pytest.raises(RuntimeError, match="ETag unavailable"):
            s3_backend.get_file_identity("test/sample.txt")


def test_get_file_list_skips_directories(s3_backend_with_directory):
    """Test that get_file_list skips directory markers."""
    # Get file list
    files = s3_backend_with_directory.get_file_list()

    # Should only include actual files, not the directory marker
    assert all(not f.endswith("/") for f in files)
    assert "s3://test-bucket/test/subdir/" not in files
    assert "s3://test-bucket/test/subdir/file.txt" in files


def test_parse_input_s3_uri(s3_backend):
    """Test parsing full S3 URI."""
    bucket, key = s3_backend._parse_input("s3://my-bucket/path/to/file.txt")
    assert bucket == "my-bucket"
    assert key == "path/to/file.txt"


def test_parse_input_s3_uri_with_leading_slash(s3_backend):
    """Test parsing S3 URI strips leading slash from key."""
    bucket, key = s3_backend._parse_input("s3://my-bucket//path/to/file.txt")
    assert bucket == "my-bucket"
    assert key == "path/to/file.txt"


def test_parse_input_key_only(s3_backend):
    """Test parsing key without URI."""
    bucket, key = s3_backend._parse_input("path/to/file.txt")
    assert bucket == "test-bucket"
    assert key == "path/to/file.txt"


def test_to_url(s3_backend):
    """Test URL construction."""
    url = s3_backend._to_url("my-bucket", "path/to/file.txt")
    assert url == "s3://my-bucket/path/to/file.txt"


def test_is_not_found_error(s3_backend):
    """Test detection of not-found errors."""
    # Test all not-found error codes
    for code in ("404", "403", "NoSuchKey", "NoSuchBucket"):
        error_response = {"Error": {"Code": code, "Message": "Not found"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
        error = ClientError(error_response, "TestOperation")
        assert s3_backend._is_not_found_error(error) is True

    # Test non-not-found error
    error_response = {
        "Error": {"Code": "InternalError", "Message": "Error"},
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }
    error = ClientError(error_response, "TestOperation")
    assert s3_backend._is_not_found_error(error) is False


def test_scheme_property(s3_backend):
    """Test scheme property."""
    assert s3_backend.scheme == "s3"


def test_bucket_property(s3_backend):
    """Test bucket property."""
    assert s3_backend.bucket == "test-bucket"


def test_prefix_property(s3_backend):
    """Test prefix property."""
    assert s3_backend.prefix == "test/"


def test_remote_source_property(s3_backend):
    """Test remote_source property."""
    assert s3_backend.remote_source is True


def test_get_file_list_with_metadata_skips_directories(s3_backend_with_directory):
    """Test get_file_list_with_metadata skips directory markers and populates metadata."""
    result = s3_backend_with_directory.get_file_list_with_metadata()

    # Should skip directory marker, include actual file
    assert all(not f.path.endswith("/") for f in result)
    paths = [f.path for f in result]
    assert "s3://test-bucket/test/subdir/" not in paths
    assert "s3://test-bucket/test/subdir/file.txt" in paths

    # Verify metadata is populated from listing (no extra HEAD requests)
    for info in result:
        assert isinstance(info, FileInfo)
        assert info.last_modified.tzinfo is not None
        assert info.size >= 0
        assert isinstance(info.file_identity, str)
        assert info.file_identity != ""
