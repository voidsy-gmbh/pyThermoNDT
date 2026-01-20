from pythermondt.io import IOPathWrapper


def test_remote_source(configured_backend):
    backend, config = configured_backend
    assert backend.remote_source == config.is_remote


def test_read_file(configured_backend, test_files):
    backend, _ = configured_backend

    result = backend.read_file(test_files["sample.txt"])
    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read().decode() == "test content"
