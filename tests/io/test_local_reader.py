"""Tests for LocalReader files, file_uris, and file_names properties."""

import pytest

from pythermondt.readers import LocalReader


@pytest.mark.parametrize(
    "filename, encoded_substring",
    [
        ("normal.h5", "normal.h5"),
        ("file with spaces.h5", "file%20with%20spaces.h5"),
        ("file#hash.h5", "file%23hash.h5"),
        ("file%percent.h5", "file%25percent.h5"),
        ("file&and.h5", "file%26and.h5"),
    ],
)
def test_file_uris_is_url_encoded(tmp_path, filename, encoded_substring):
    """File_uris returns URL-encoded paths for files with special characters."""
    (tmp_path / filename).write_text("content")
    reader = LocalReader(str(tmp_path))

    uris = reader.file_uris

    assert any(encoded_substring in u for u in uris)


@pytest.mark.parametrize(
    "filename",
    [
        "normal.h5",
        "file with spaces.h5",
        "file#hash.h5",
        "file%percent.h5",
        "file&and.h5",
    ],
)
def test_files_is_decoded(tmp_path, filename):
    """Files returns human-readable paths with special characters preserved."""
    (tmp_path / filename).write_text("content")
    reader = LocalReader(str(tmp_path))

    files = reader.files

    assert any(filename in f for f in files)


@pytest.mark.parametrize(
    "filename, expected_basename",
    [
        ("normal.h5", "normal.h5"),
        ("file with spaces.h5", "file with spaces.h5"),
        ("file#hash.h5", "file#hash.h5"),
        ("file%percent.h5", "file%percent.h5"),
        ("file&and.h5", "file&and.h5"),
    ],
)
def test_file_names_decodes_correctly(tmp_path, filename, expected_basename):
    """File_names extracts basenames with special characters preserved."""
    (tmp_path / filename).write_text("content")
    reader = LocalReader(str(tmp_path))

    names = reader.file_names

    assert expected_basename in names
    for name in names:
        assert "/" not in name


def test_files_and_file_uris_same_count(tmp_path):
    """Files and file_uris return the same number of entries."""
    for fname in ("a.h5", "b.h5", "c.h5"):
        (tmp_path / fname).write_text("content")
    reader = LocalReader(str(tmp_path))

    assert len(reader.files) == len(reader.file_uris)


def test_len_uses_files(tmp_path):
    """Reader length matches the number of files."""
    for fname in ("a.h5", "b.h5"):
        (tmp_path / fname).write_text("content")
    reader = LocalReader(str(tmp_path))

    assert len(reader) == len(reader.files)


def test_file_uris_usable_for_reading(tmp_path):
    """Encoded file_uris work with backend.read_file for files with spaces."""
    (tmp_path / "file with spaces.h5").write_text("test content")
    reader = LocalReader(str(tmp_path))

    for uri in reader.file_uris:
        wrapper = reader.backend.read_file(uri)
        assert wrapper.file_obj.read().decode("utf-8") == "test content"


def test_cache_files_false(tmp_path):
    """Files and file_uris work correctly when caching is disabled."""
    (tmp_path / "test.h5").write_text("content")
    reader = LocalReader(str(tmp_path), cache_files=False)

    files = reader.files
    uris = reader.file_uris

    assert "test.h5" in files[0]
    assert "test.h5" in uris[0]
    assert len(files) == len(uris)


def test_files_vs_file_uris_differ_for_special_chars(tmp_path):
    """Files (decoded) and file_uris (encoded) differ when paths have special chars."""
    (tmp_path / "file with spaces.h5").write_text("content")
    reader = LocalReader(str(tmp_path))

    assert reader.files[0] != reader.file_uris[0]
    assert " " in reader.files[0]
    assert "%20" in reader.file_uris[0]
