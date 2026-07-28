"""Tests for general backend interface compliance."""

from pathlib import Path

import pytest

from pythermondt.io import FileInfo, IOPathWrapper
from tests.support.storage import StorageTestContext


def test_scheme(storage_context: StorageTestContext):
    """Test every backend exposes a canonical scheme."""
    assert storage_context.backend.scheme in {"file", "s3", "az"}


def test_remote_source(storage_context: StorageTestContext):
    """Test remote_source agrees with the backend scheme."""
    assert storage_context.backend.remote_source == (storage_context.backend.scheme != "file")


def test_read_file(storage_context: StorageTestContext, test_file):
    """Test reading single file."""
    backend_instance = storage_context.backend
    file_path, expected_content = test_file

    result = backend_instance.read_file(file_path)
    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read() == expected_content


def test_read_file_not_exist(storage_context: StorageTestContext):
    """Test reading non-existent file raises FileNotFoundError."""
    backend_instance = storage_context.backend
    path = storage_context.canonical_path("non_existent_file.txt")
    with pytest.raises(FileNotFoundError, match="File not found:"):
        backend_instance.read_file(path)


def test_write_file(storage_context: StorageTestContext):
    """Test writing and reading back."""
    backend_instance = storage_context.backend
    file_path = storage_context.prepare_file("new_sample.txt", b"")

    # Write
    data = IOPathWrapper(b"new test content")
    backend_instance.write_file(data, file_path)

    # Read back
    result = backend_instance.read_file(file_path)
    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read() == b"new test content"


@pytest.mark.parametrize("exists", [True, False])
def test_exists(storage_context: StorageTestContext, exists):
    """Test file existence check."""
    backend_instance = storage_context.backend
    file_path = storage_context.canonical_path("test_exists.txt")

    # Create file if should exist
    if exists:
        data = IOPathWrapper(b"exists test")
        backend_instance.write_file(data, file_path)

    assert backend_instance.exists(file_path) == exists


def test_get_file_size_not_exist(storage_context: StorageTestContext):
    """Test getting file size for a non-existent file raises FileNotFoundError."""
    backend_instance = storage_context.backend
    path = storage_context.canonical_path("non_existent_file.txt")
    with pytest.raises(FileNotFoundError, match="File not found:"):
        backend_instance.get_file_size(path)


def test_get_file_size(storage_context: StorageTestContext, test_file):
    """Test getting file size."""
    backend_instance = storage_context.backend
    file_path, content = test_file

    size = backend_instance.get_file_size(file_path)
    assert size == len(content)


def test_get_file_identity(storage_context: StorageTestContext, test_file):
    """Test getting a backend-specific file identity."""
    backend_instance = storage_context.backend
    file_path, _ = test_file

    identity = backend_instance.get_file_identity(file_path)

    assert isinstance(identity, str)
    assert identity != ""


def test_get_file_identity_changes_after_content_update(storage_context: StorageTestContext, test_file):
    """Test identity changes when file content changes."""
    backend_instance = storage_context.backend
    file_path, _ = test_file

    identity_before = backend_instance.get_file_identity(file_path)

    backend_instance.write_file(IOPathWrapper(b"updated test content"), file_path)
    identity_after = backend_instance.get_file_identity(file_path)

    assert identity_before != identity_after


def test_get_file_identity_not_exist(storage_context: StorageTestContext):
    """Test getting identity for non-existent file raises FileNotFoundError."""
    backend_instance = storage_context.backend
    path = storage_context.canonical_path("non_existent_file.txt")

    with pytest.raises(FileNotFoundError, match="File not found:"):
        backend_instance.get_file_identity(path)


def test_get_file_list(storage_context: StorageTestContext, test_file):
    """Test listing a single file without any filters."""
    backend_instance = storage_context.backend
    file_path, _ = test_file
    file_list = backend_instance.get_file_list()

    assert len(file_list) == 1
    assert file_list[0].startswith(backend_instance.scheme + "://")
    assert set(file_list) == {file_path}


def test_get_file_list_all(storage_context: StorageTestContext, test_files_scenario):
    """Test listing all files."""
    backend_instance = storage_context.backend
    file_list = backend_instance.get_file_list()

    assert len(file_list) == len(test_files_scenario)
    assert all(f.startswith(backend_instance.scheme + "://") for f in file_list)
    assert set(file_list) == set(test_files_scenario.values())


def test_download_file(storage_context: StorageTestContext, tmp_path, test_file):
    """Test file download/copy."""
    backend_instance = storage_context.backend
    file_path, expected_content = test_file

    # Extract filename for destination
    filename = Path(file_path).name
    dest_path = str(tmp_path / f"downloaded_{filename}")

    if not backend_instance.remote_source:
        # Local backends don't support download
        with pytest.raises(NotImplementedError):
            backend_instance.download_file(file_path, dest_path)
    else:
        # Remote backends download to local filesystem
        backend_instance.download_file(file_path, dest_path)

        # Verify content
        with open(dest_path, "rb") as f:
            downloaded_content = f.read()
        assert downloaded_content == expected_content


def test_get_file_list_with_metadata_single(storage_context: StorageTestContext, test_file):
    """Test get_file_list_with_metadata returns correct FileInfo for a single file."""
    backend_instance = storage_context.backend
    file_path, expected_content = test_file

    result = backend_instance.get_file_list_with_metadata()
    assert len(result) == 1

    info = result[0]
    assert isinstance(info, FileInfo)
    assert info.path == file_path
    assert info.size == len(expected_content)
    assert info.last_modified.tzinfo is not None  # tz-aware UTC
    assert isinstance(info.file_identity, str)
    assert info.file_identity != ""


def test_get_file_list_with_metadata_all(storage_context: StorageTestContext, test_files_scenario):
    """Test get_file_list_with_metadata with all files."""
    backend_instance = storage_context.backend

    result = backend_instance.get_file_list_with_metadata()

    assert len(result) == len(test_files_scenario)
    assert all(f.path.startswith(backend_instance.scheme + "://") for f in result)
    assert all(f.last_modified.tzinfo is not None for f in result)
    assert all(isinstance(f.file_identity, str) and f.file_identity != "" for f in result)
    assert all(f.size >= 0 for f in result)


def test_get_file_list_with_metadata_paths_match_get_file_list(
    storage_context: StorageTestContext, test_files_scenario
):
    """Test that paths returned by both listing methods are identical."""
    backend_instance = storage_context.backend

    paths_fast = set(backend_instance.get_file_list())
    paths_meta = {info.path for info in backend_instance.get_file_list_with_metadata()}

    assert paths_fast == paths_meta
