"""Specific tests for the LocalBackend class."""

import io
import os
from pathlib import Path

import pytest

from pythermondt.io.backends import LocalBackend
from pythermondt.io.utils import IOPathWrapper


def _create_test_file(tmp_path):
    """Helper to create test file and return path."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    return str(test_file)


@pytest.mark.parametrize(
    "setup_func",
    [
        _create_test_file,
        lambda tmp_path: str(tmp_path),
        lambda tmp_path: str(tmp_path / "*.txt"),
    ],
)
def test_init_valid_patterns(tmp_path, setup_func):
    """Test LocalBackend initialization with valid patterns."""
    pattern = setup_func(tmp_path)
    backend = LocalBackend(pattern)

    assert backend.pattern == pattern
    assert backend.remote_source is False


@pytest.mark.parametrize("pattern", [123, None, 3.14, ["invalid"], {"key": "value"}])
def test_init_invalid_pattern_type(pattern):
    """Test that invalid pattern types raise ValueError."""
    with pytest.raises(ValueError, match=f"Invalid pattern type: {type(pattern)}. Must be a string."):
        LocalBackend(pattern)


@pytest.mark.parametrize("pattern", ["[[ ]]"])
def test_init_invalid_pattern(pattern):
    """Test that invalid pattern strings raise ValueError."""
    with pytest.raises(ValueError, match="Invalid pattern:"):
        LocalBackend(pattern)


@pytest.mark.parametrize("pattern", ["", "nonexistent/path", "nonexistent/file.txt", "nonexistent/*.md"])
def test_init_nonexistent_pattern(pattern):
    """Test that non-existent paths are treated as valid but return empty file list."""
    backend = LocalBackend(pattern)
    assert backend.pattern == pattern
    assert len(backend.get_file_list()) == 0  # Should return empty list for non-existent path


def test_close_does_nothing(tmp_path):
    """Test that close method doesn't raise errors."""
    backend = LocalBackend(str(tmp_path))
    backend.close()  # Should not raise


def test_download_file_not_implemented(tmp_path: Path):
    """Test that download_file raises NotImplementedError."""
    backend = LocalBackend(str(tmp_path))

    with pytest.raises(NotImplementedError, match="Direct download is not supported"):
        backend.download_file("source.txt", "dest.txt")


def test_get_file_list_single_file(tmp_path: Path):
    """Test get_file_list with single file pattern."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    backend = LocalBackend(str(test_file))

    result = backend.get_file_list()
    expected = [str(test_file)]

    assert len(result) == 1
    assert result == expected


def test_get_file_list_directory(tmp_path: Path):
    """Test get_file_list with directory pattern."""
    files = ["test1.txt", "test2.py", "test3.txt"]
    paths = sorted(tmp_path / filename for filename in files)  # Should be sorted to match the backend's behavior

    for path in paths:
        path.write_text("content")

    backend = LocalBackend(str(tmp_path))

    result = backend.get_file_list()
    expected = [str(p) for p in paths]  # Convert path objects to strings for comparison

    assert len(result) == 3
    assert result == expected


def test_get_file_list_glob_pattern(tmp_path: Path):
    """Test get_file_list with glob pattern."""
    files = ["test1.txt", "test2.py", "other.txt"]
    paths = sorted(tmp_path / filename for filename in files)  # Should be sorted to match the backend's behavior

    for path in paths:
        path.write_text("content")

    pattern = str(tmp_path / "test*.txt")
    backend = LocalBackend(pattern)

    result = backend.get_file_list()
    expected = [str(tmp_path / "test1.txt")]  # Only the file matching the pattern should be returned

    assert len(result) == 1
    assert result == expected


@pytest.mark.parametrize(
    "extensions, expected_count",
    [
        ((".txt",), 2),
        ((".py",), 1),
        ((".txt", ".py"), 3),
        ((".md",), 0),
    ],
)
def test_get_file_list_extension_filter(tmp_path, extensions, expected_count):
    """Test get_file_list with extension filtering."""
    files = ["test1.txt", "test2.py", "test3.txt"]
    for filename in files:
        (tmp_path / filename).write_text("content")

    backend = LocalBackend(str(tmp_path))
    result = backend.get_file_list(extensions=extensions)

    assert len(result) == expected_count
    if expected_count > 0:
        assert all(f.endswith(extensions) for f in result)


def test_get_file_list_num_files_limit(tmp_path):
    """Test get_file_list with number limit."""
    files = ["test1.txt", "test2.txt", "test3.txt", "test4.txt"]
    for filename in files:
        (tmp_path / filename).write_text("content")

    backend = LocalBackend(str(tmp_path))
    result = backend.get_file_list(num_files=2)

    assert len(result) == 2


def test_get_file_list_empty_directory(tmp_path):
    """Test get_file_list with empty directory."""
    backend = LocalBackend(str(tmp_path))
    result = backend.get_file_list()

    assert len(result) == 0


def test_get_file_list_sorting(tmp_path):
    """Test that get_file_list returns sorted results."""
    files = ["zebra.txt", "alpha.txt", "beta.txt"]
    for filename in files:
        (tmp_path / filename).write_text("content")

    backend = LocalBackend(str(tmp_path))
    result = backend.get_file_list()

    # Check that results are sorted (regardless of path separator format)
    assert len(result) == 3
    assert result == sorted(result)
    # Check that all expected filenames are present
    result_names = [Path(f).name for f in result]
    assert result_names == ["alpha.txt", "beta.txt", "zebra.txt"]


def test_read_nonexistent_file(tmp_path):
    """Test reading non-existent file."""
    backend = LocalBackend(str(tmp_path))
    nonexistent_file = str(tmp_path / "nonexistent.txt")

    result = backend.read_file(nonexistent_file)
    assert isinstance(result, IOPathWrapper)

    with pytest.raises(FileNotFoundError):
        result.file_obj.read()


def test_unicode_filenames(tmp_path):
    """Test handling files with unicode characters."""
    backend = LocalBackend(str(tmp_path))
    unicode_filename = "test_ñáméé_файл.txt"
    unicode_file = tmp_path / unicode_filename
    test_content = "unicode test content"

    try:
        unicode_file.write_text(test_content, encoding="utf-8")
        result = backend.read_file(str(unicode_file))
        assert isinstance(result, IOPathWrapper)
        assert result.file_obj.read().decode("utf-8") == test_content
    except (UnicodeError, OSError):
        pytest.skip("Unicode filenames not supported on this system")


def test_special_characters_in_paths(tmp_path):
    """Test handling paths with special characters."""
    special_dir = tmp_path / "test dir with spaces & special chars"
    special_dir.mkdir()

    backend = LocalBackend(str(special_dir))
    test_file = special_dir / "test file.txt"
    test_file.write_text("content")

    files = backend.get_file_list()
    assert len(files) == 1
    assert "test file.txt" in files[0]


def test_full_workflow(tmp_path):
    """Test complete workflow: create files, list, read, write."""
    test_files = {
        "data1.txt": "content 1",
        "data2.txt": "content 2",
        "script.py": "print('hello')",
        "readme.md": "# README",
    }

    for filename, content in test_files.items():
        (tmp_path / filename).write_text(content)

    backend = LocalBackend(str(tmp_path))

    # Test listing
    all_files = backend.get_file_list()
    assert len(all_files) == 4

    txt_files = backend.get_file_list(extensions=(".txt",))
    assert len(txt_files) == 2

    # Test reading
    for file_path in txt_files:
        assert backend.exists(file_path)
        content_wrapper = backend.read_file(file_path)
        content = content_wrapper.file_obj.read().decode()
        filename = Path(file_path).name
        assert content == test_files[filename]

    # Test writing
    new_content = IOPathWrapper(io.BytesIO(b"new content"))
    new_file_path = str(tmp_path / "new_file.txt")

    backend.write_file(new_content, new_file_path)
    assert backend.exists(new_file_path)

    read_back = backend.read_file(new_file_path)
    assert read_back.file_obj.read() == b"new content"


def test_nested_directory_structure(tmp_path):
    """Test working with nested directories."""
    nested_dir = tmp_path / "level1" / "level2"
    nested_dir.mkdir(parents=True)

    (tmp_path / "root.txt").write_text("root content")
    (tmp_path / "level1" / "level1.txt").write_text("level1 content")
    (nested_dir / "level2.txt").write_text("level2 content")

    # Test with root directory - only gets direct files
    backend = LocalBackend(str(tmp_path))
    root_files = backend.get_file_list()
    assert len(root_files) == 1

    # Test with nested directory
    backend_nested = LocalBackend(str(nested_dir))
    nested_files = backend_nested.get_file_list()
    assert len(nested_files) == 1
    assert "level2.txt" in nested_files[0]


def test_concurrent_access(tmp_path):
    """Test multiple backend instances work simultaneously."""
    test_file = tmp_path / "shared.txt"
    test_file.write_text("shared content")

    backend1 = LocalBackend(str(tmp_path))
    backend2 = LocalBackend(str(test_file))

    files1 = backend1.get_file_list()
    files2 = backend2.get_file_list()

    assert len(files1) == 1
    assert len(files2) == 1

    content1 = backend1.read_file(str(test_file))
    content2 = backend2.read_file(str(test_file))

    assert content1.file_obj.read() == content2.file_obj.read()


def test_recursive_directory_traversal(tmp_path):
    """Test recursive directory traversal."""
    # Create nested structure
    nested = tmp_path / "level1" / "level2"
    nested.mkdir(parents=True)

    (tmp_path / "root.txt").write_text("content")
    (tmp_path / "level1" / "l1.txt").write_text("content")
    (nested / "l2.txt").write_text("content")

    # Test non-recursive
    backend = LocalBackend(str(tmp_path), recursive=False)
    files = backend.get_file_list()
    assert len(files) == 1  # Only root.txt

    # Test recursive
    backend_recursive = LocalBackend(str(tmp_path), recursive=True)
    files_recursive = backend_recursive.get_file_list()
    assert len(files_recursive) == 3  # All files


def test_recursive_glob_pattern(tmp_path):
    """Test recursive glob patterns."""
    nested = tmp_path / "subdir"
    nested.mkdir()

    (tmp_path / "test1.txt").write_text("content")
    (nested / "test2.txt").write_text("content")

    pattern = str(tmp_path / "**/*.txt")

    # Non-recursive should find only direct matches
    backend = LocalBackend(pattern, recursive=False)
    files = backend.get_file_list()
    assert len(files) == 1

    # Recursive should find all matches
    backend_recursive = LocalBackend(pattern, recursive=True)
    files_recursive = backend_recursive.get_file_list()
    assert len(files_recursive) == 2


def test_permission_denied_directory(tmp_path, monkeypatch):
    """Test handling of permission denied errors."""

    # Mock os.path.isdir to return True, but os.scandir to raise PermissionError
    def mock_scandir(path):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(os, "scandir", mock_scandir)
    monkeypatch.setattr(os.path, "isdir", lambda x: True)

    backend = LocalBackend(str(tmp_path))

    # Should handle permission errors gracefully
    with pytest.raises(PermissionError):
        backend.get_file_list()


def test_pattern_with_special_characters(tmp_path):
    """Test patterns with special filesystem characters."""
    special_chars = ["spaces in name", "file&with&ampersand", "file(with)parens"]

    for char_name in special_chars:
        test_file = tmp_path / f"test_{char_name}.txt"
        test_file.write_text("content")

    backend = LocalBackend(str(tmp_path))
    files = backend.get_file_list()
    assert len(files) == len(special_chars)


def test_get_file_size(tmp_path):
    """Test get_file_size returns correct file size."""
    backend = LocalBackend(str(tmp_path))

    # Test empty file
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    assert backend.get_file_size(str(empty_file)) == 0

    # Test file with content
    test_content = "Hello, World!"
    test_file = tmp_path / "test.txt"
    test_file.write_text(test_content)
    expected_size = len(test_content.encode("utf-8"))
    assert backend.get_file_size(str(test_file)) == expected_size

    # Test binary file
    binary_content = b"\x00\x01\x02\x03\x04\x05"
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(binary_content)
    assert backend.get_file_size(str(binary_file)) == len(binary_content)


def test_get_file_size_nonexistent_file(tmp_path):
    """Test get_file_size raises error for non-existent file."""
    backend = LocalBackend(str(tmp_path))
    nonexistent_file = str(tmp_path / "nonexistent.txt")

    with pytest.raises(FileNotFoundError):
        backend.get_file_size(nonexistent_file)


def test_get_file_size_large_file(tmp_path):
    """Test get_file_size with larger file."""
    backend = LocalBackend(str(tmp_path))

    # Create a file with known size (1KB)
    large_content = "A" * 1024
    large_file = tmp_path / "large.txt"
    large_file.write_text(large_content)

    assert backend.get_file_size(str(large_file)) == 1024


def test_get_file_size_directory_raises_error(tmp_path):
    """Test get_file_size raises error when given directory path."""
    backend = LocalBackend(str(tmp_path))

    # Create a subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    # Should raise an error when trying to get size of directory
    with pytest.raises(IsADirectoryError):
        backend.get_file_size(str(subdir))
