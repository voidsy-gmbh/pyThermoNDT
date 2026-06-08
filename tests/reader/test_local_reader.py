"""Tests for LocalReader-specific behavior including URI encoding and recursive discovery."""

from pathlib import Path

import pytest

from pythermondt.readers import LocalReader
from tests.support.storage import PlainTextParser


def test_local_reader_recursive_includes_nested_test_files(tmp_path: Path):
    """Test recursive LocalReader discovery includes nested supported files."""
    asset_dir = Path(__file__).parents[1] / "assets" / "reader"

    # Keep the test independent from the committed asset directory layout.
    for relative_path in ("sample1.test", "nested/sample3.test"):
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((asset_dir / relative_path).read_text())

    reader = LocalReader(pattern=str(tmp_path), recursive=True, parser=PlainTextParser)

    assert sorted(reader.file_names) == ["sample1.test", "sample3.test"]


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
