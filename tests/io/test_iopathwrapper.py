"""Tests for IOPathWrapper edge cases."""

import os
from io import BytesIO

import pytest
from pytest import FixtureRequest, MonkeyPatch

from pythermondt.io.utils import IOPathWrapper


@pytest.fixture(params=[None, 12345, [1, 2, 3], {"key": "value"}, 3.14], ids=lambda x: f"invalid_value={x!r}")
def invalid_values(request: FixtureRequest):
    """Fixture that provides invalid values for testing."""
    return request.param


@pytest.fixture(params=["path", "bytes", "bytesio"])
def valid_values(request: FixtureRequest, tmp_path):
    match request.param:
        case "path":
            p = tmp_path / "test.txt"
            p.write_bytes(b"hello from file")
            return str(p)
        case "bytes":
            return b"hello from bytes"
        case "bytesio":
            return BytesIO(b"hello from BytesIO")


def test_valid_values(valid_values):
    """Test that valid values are accepted and file_obj returns expected content."""
    wrapper = IOPathWrapper(valid_values)
    result = wrapper.file_obj
    assert isinstance(result, BytesIO)
    match valid_values:
        case str():
            assert result.read() == b"hello from file"
        case b"hello from bytes":
            assert result.read() == b"hello from bytes"
        case BytesIO():
            assert result is valid_values
            assert result.read() == b"hello from BytesIO"


def test_unsupported_source_type(invalid_values):
    """Test that an integer source raises ValueError on file_obj access."""
    wrapper = IOPathWrapper(invalid_values)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported source type. Must be str, bytes, or BytesIO."):
        _ = wrapper.file_obj


@pytest.mark.parametrize(
    "error_type",
    [PermissionError, OSError, FileNotFoundError],
    ids=["PermissionError", "OSError", "FileNotFoundError"],
)
def test_close_swallows_errors(error_type: type[Exception], caplog):
    wrapper = IOPathWrapper(b"test content")
    temp_path = wrapper.file_path
    assert os.path.exists(temp_path)

    def boom(_path: str):
        raise error_type("mocked error")

    with MonkeyPatch.context() as mp:
        mp.setattr("pythermondt.io.utils.os.remove", boom)

        # should NOT raise
        with caplog.at_level("WARNING"):
            wrapper.close()

    # file still exists because removal failed (was mocked to raise)
    assert os.path.exists(temp_path)

    # optional: ensure it logged
    assert "Failed to remove temporary file" in caplog.text

    # cleanup (real remove)
    os.remove(temp_path)


def test_close_removes_temp_file():
    """Test that close() successfully removes the temporary file."""
    wrapper = IOPathWrapper(b"hello world")
    temp_path = wrapper.file_path
    assert os.path.exists(temp_path)

    wrapper.close()
    assert not os.path.exists(temp_path)
