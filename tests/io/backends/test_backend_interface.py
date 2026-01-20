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
