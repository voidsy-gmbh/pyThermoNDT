"""Tests for general backend interface compliance."""

from pathlib import Path

import pytest

from pythermondt.io import IOPathWrapper


def test_scheme(backend_config):
    """Test scheme property matches config."""
    backend_instance, config = backend_config
    assert backend_instance.scheme == config.scheme


def test_remote_source(backend_config):
    """Test remote_source property matches config."""
    backend_instance, config = backend_config
    assert backend_instance.remote_source == config.is_remote


def test_read_file(backend_config, test_file):
    """Test reading single file."""
    backend_instance, _ = backend_config
    file_path, expected_content = test_file

    result = backend_instance.read_file(file_path)
    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read() == expected_content


def test_read_file_not_exist(backend_config, tmp_path):
    """Test reading non-existent file raises FileNotFoundError."""
    backend_instance, config = backend_config

    # Determine path based on backend type
    if config.is_remote:
        path = "non_existent_file.txt"
    else:
        path = str(tmp_path / "non_existent_file.txt")
    with pytest.raises(FileNotFoundError, match="File not found:"):
        backend_instance.read_file(path)


def test_write_file(backend_config, tmp_path):
    """Test writing and reading back."""
    backend_instance, config = backend_config

    # Determine path based on backend type
    if config.is_remote:
        file_path = "new_sample.txt"
    else:
        file_path = str(tmp_path / "new_sample.txt")

    # Write
    data = IOPathWrapper(b"new test content")
    backend_instance.write_file(data, file_path)

    # Read back
    result = backend_instance.read_file(file_path)
    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read() == b"new test content"


@pytest.mark.parametrize("exists", [True, False])
def test_exists(backend_config, tmp_path, exists):
    """Test file existence check."""
    backend_instance, config = backend_config

    if config.is_remote:
        file_path = "test_exists.txt"
    else:
        file_path = str(tmp_path / "test_exists.txt")

    # Create file if should exist
    if exists:
        data = IOPathWrapper(b"exists test")
        backend_instance.write_file(data, file_path)

    assert backend_instance.exists(file_path) == exists


def test_get_file_size_not_exist(backend_config, tmp_path):
    """Test getting file size for a non-existent file raises FileNotFoundError."""
    backend_instance, config = backend_config

    # Determine path based on backend type
    if config.is_remote:
        path = "non_existent_file.txt"
    else:
        path = str(tmp_path / "non_existent_file.txt")
    with pytest.raises(FileNotFoundError, match="File not found:"):
        backend_instance.get_file_size(path)


def test_get_file_size(backend_config, test_file):
    """Test getting file size."""
    backend_instance, _ = backend_config
    file_path, content = test_file

    size = backend_instance.get_file_size(file_path)
    assert size == len(content)


def test_get_file_identity(backend_config, test_file):
    """Test getting a backend-specific file identity."""
    backend_instance, _ = backend_config
    file_path, _ = test_file

    identity = backend_instance.get_file_identity(file_path)

    assert isinstance(identity, str)
    assert identity != ""


def test_get_file_identity_changes_after_content_update(backend_config, test_file):
    """Test identity changes when file content changes."""
    backend_instance, _ = backend_config
    file_path, _ = test_file

    identity_before = backend_instance.get_file_identity(file_path)

    backend_instance.write_file(IOPathWrapper(b"updated test content"), file_path)
    identity_after = backend_instance.get_file_identity(file_path)

    assert identity_before != identity_after


def test_get_file_identity_not_exist(backend_config, tmp_path):
    """Test getting identity for non-existent file raises FileNotFoundError."""
    backend_instance, config = backend_config

    if config.is_remote:
        path = "non_existent_file.txt"
    else:
        path = str(tmp_path / "non_existent_file.txt")

    with pytest.raises(FileNotFoundError, match="File not found:"):
        backend_instance.get_file_identity(path)


def test_get_file_list(backend_config, test_file):
    """Test listing a single file without any filters."""
    backend_instance, config = backend_config
    file_path, _ = test_file
    file_list = backend_instance.get_file_list()

    assert len(file_list) == 1
    assert file_list[0].startswith(config.scheme + "://")
    assert set(file_list) == {file_path}


def test_get_file_list_all(backend_config, test_files_scenario):
    """Test listing all files."""
    backend_instance, config = backend_config
    file_list = backend_instance.get_file_list()

    assert len(file_list) == len(test_files_scenario)
    assert all(f.startswith(config.scheme + "://") for f in file_list)
    assert set(file_list) == set(test_files_scenario.values())


def test_download_file(backend_config, tmp_path, test_file):
    """Test file download/copy."""
    backend_instance, config = backend_config
    file_path, expected_content = test_file

    # Extract filename for destination
    filename = Path(file_path).name
    dest_path = str(tmp_path / f"downloaded_{filename}")

    if not config.is_remote:
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
