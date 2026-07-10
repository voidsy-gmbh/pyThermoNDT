"""Tests for IOPathWrapper edge cases."""

import json
import os
from io import BytesIO
from re import escape

import pytest
from pytest import FixtureRequest, MonkeyPatch

from pythermondt.io.utils import IOPathWrapper


@pytest.fixture(params=[12345, [1, 2, 3], {"key": "value"}, 3.14], ids=lambda x: f"invalid_value={x!r}")
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
    with pytest.raises(ValueError, match=escape("Unsupported source type. Must be str, bytes, or BytesIO.")):
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


def test_default_constructor_creates_writable_buffer():
    """Default-constructed wrapper starts as an empty writable buffer."""
    wrapper = IOPathWrapper()
    assert wrapper.getvalue() == b""

    wrapper.write(b"hello")
    assert wrapper.getvalue() == b"hello"


def test_context_manager_calls_close():
    """Context manager cleans up temp files on normal exit."""
    wrapper = IOPathWrapper(b"content")
    temp_path = wrapper.file_path
    assert os.path.exists(temp_path)

    with wrapper as f:
        assert f is wrapper
        assert f.getvalue() == b"content"

    assert not os.path.exists(temp_path)


def test_write_with_str_encodes_to_bytes():
    """write() accepts str, encodes to UTF-8, and returns byte count."""
    wrapper = IOPathWrapper()
    n = wrapper.write("héllo")
    assert n == 6  # é is 2 bytes in UTF-8
    assert wrapper.getvalue() == "héllo".encode()


def test_write_returns_byte_count_for_bytes_input():
    """write() with bytes input returns the number of bytes written."""
    wrapper = IOPathWrapper()
    n = wrapper.write(b"abc")
    assert n == 3
    assert wrapper.getvalue() == b"abc"


def test_sequential_writes_accumulate():
    """Multiple write calls append without overwriting previous content."""
    wrapper = IOPathWrapper()
    wrapper.write(b"hello")
    wrapper.write(b" world")
    assert wrapper.getvalue() == b"hello world"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("bytes", b"hello from bytes"),
        ("bytesio", b"hello from BytesIO"),
        ("path", b"hello from file"),
    ],
    ids=["bytes", "bytesio", "path"],
)
def test_getvalue_on_source_types(source, expected, tmp_path):
    """getvalue() returns buffer content for path, bytes, and BytesIO sources."""
    src: str | bytes | BytesIO
    match source:
        case "path":
            p = tmp_path / "test.txt"
            p.write_bytes(expected)
            src = str(p)
        case "bytes":
            src = expected
        case "bytesio":
            src = BytesIO(expected)

    wrapper = IOPathWrapper(src)
    assert wrapper.getvalue() == expected


def test_json_dump_integration():
    """json.dump writes JSON to the wrapper and getvalue returns valid parseable JSON."""
    results = {"key": "value", "count": 42}

    with IOPathWrapper() as f:
        json.dump(results, f)
        content = f.getvalue()

    assert json.loads(content) == results


def test_context_manager_cleans_up_on_exception():
    """Context manager cleans up temp files even when an exception propagates."""
    wrapper = IOPathWrapper(b"content")
    temp_path = wrapper.file_path
    assert os.path.exists(temp_path)

    try:
        with wrapper:
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    assert not os.path.exists(temp_path)
