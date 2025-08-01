import io
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
    "source_type, setup_func",
    [
        ("file", _create_test_file),
        ("directory", lambda tmp_path: str(tmp_path)),
        ("glob_pattern", lambda tmp_path: str(tmp_path / "*.txt")),
    ],
)
def test_init_valid_patterns(tmp_path, source_type, setup_func):
    """Test LocalBackend initialization with valid patterns."""
    pattern = setup_func(tmp_path)
    backend = LocalBackend(pattern)

    assert backend.pattern == pattern.replace("\\", "/")
    assert backend.remote_source is False


def test_init_backslash_conversion(tmp_path):
    """Test that backslashes are converted to forward slashes."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    path_with_backslashes = str(test_file).replace("/", "\\")
    backend = LocalBackend(path_with_backslashes)

    assert "\\" not in backend.pattern
    assert "/" in backend.pattern


def test_init_invalid_integer():
    """Test that integer input raises AttributeError."""
    with pytest.raises(AttributeError, match="'int' object has no attribute 'replace'"):
        LocalBackend(123)  # type: ignore


def test_init_nonexistent_path_as_regex():
    """Test that non-existent paths are treated as regex patterns."""
    # This should not raise an error as it's treated as a valid regex
    backend = LocalBackend("/nonexistent/path")
    assert backend.pattern == "/nonexistent/path"


def test_read_file(tmp_path):
    """Test reading a file returns IOPathWrapper."""
    test_file = tmp_path / "test.txt"
    test_content = "test content"
    test_file.write_text(test_content)

    backend = LocalBackend(str(tmp_path))
    result = backend.read_file(str(test_file))

    assert isinstance(result, IOPathWrapper)
    assert result.file_obj.read().decode() == test_content


def test_write_file(tmp_path):
    """Test writing a file."""
    backend = LocalBackend(str(tmp_path))
    test_content = b"test content"
    test_file = tmp_path / "output.txt"

    wrapper = IOPathWrapper(io.BytesIO(test_content))
    backend.write_file(wrapper, str(test_file))

    assert test_file.exists()
    assert test_file.read_bytes() == test_content


@pytest.mark.parametrize("exists", [True, False])
def test_exists(tmp_path, exists):
    """Test exists method."""
    backend = LocalBackend(str(tmp_path))
    test_file = tmp_path / "test.txt"

    if exists:
        test_file.write_text("content")

    assert backend.exists(str(test_file)) == exists


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
