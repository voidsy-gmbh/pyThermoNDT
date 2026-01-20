"""Tests for general backend interface compliance."""

import pytest

from pythermondt.io import IOPathWrapper


def test_remote_source(backend):
    backend, config = backend
    assert backend.remote_source == config.is_remote


def test_read_file(backend, test_file):
    backend, _ = backend
    name, content = test_file

    result = backend.read_file(name)
    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read() == content


def test_read_write_file(backend, tmp_path):
    backend, _ = backend

    # Write
    data = IOPathWrapper(b"new test content")
    file_path = str(tmp_path / "new_sample.txt")
    backend.write_file(data, file_path)

    # Read back
    result = backend.read_file(file_path)
    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read() == b"new test content"


@pytest.mark.parametrize("exists", [True, False])
def test_exists(backend, tmp_path, exists):
    backend, _ = backend

    # Write if exists
    data = IOPathWrapper(b"new test content")
    file_path = str(tmp_path / "new_sample.txt")
    if exists:
        backend.write_file(data, file_path)

    assert backend.exists(file_path) == exists


def test_get_file_list_single(backend, test_file):
    backend, _ = backend
    name, _ = test_file

    file_list = backend.get_file_list()
    assert [name] == file_list


def test_get_file_list_multi(backend, test_files_all):
    backend, _ = backend

    file_list = backend.get_file_list()
    assert set(file_list) == set(test_files_all.values())


def test_get_file_size(backend, test_file):
    backend, _ = backend
    name, content = test_file

    size = backend.get_file_size(name)
    assert size == len(content)


def test_download_file(backend, tmp_path, test_file):
    backend, config = backend
    name, content = test_file
    destination_path = str(tmp_path / f"downloaded_{name}")

    # Download not supported for local backends
    if not config.is_remote:
        with pytest.raises(NotImplementedError):
            backend.download_file(name, destination_path)
    # Else Download and verify content
    else:
        backend.download_file(name, destination_path)

        # Verify content
        with open(destination_path, "rb") as f:
            downloaded_content = f.read()
        assert downloaded_content == content
