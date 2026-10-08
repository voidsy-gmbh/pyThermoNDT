from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pythermondt.data import DataContainer, container_diff
from pythermondt.io import AzureBlobBackend, LocalBackend, S3Backend
from pythermondt.io.parsers import HDF5Parser
from pythermondt.readers import LocalReader
from pythermondt.writers import AzureBlobWriter, LocalWriter, S3Writer
from tests.support.storage import StorageTestContext
from tests.support.storage.context import AWS_BUCKET, AZURE_ACCOUNT_URL, AZURE_CONNECTION_STRING, AZURE_CONTAINER
from tests.utils import format_container_diff
from tests.writers.conftest import HDF5TestCorpus


def test_write_round_trip(storage_context: StorageTestContext, test_container: DataContainer):
    """Write a DataContainer and verify it reads back identically."""
    writer = storage_context.make_writer()
    filename = "test_file"
    read_path = storage_context.canonical_path(f"{filename}.hdf5", destination=True)

    writer.write(test_container, filename)

    # Read back via the writer's own backend.
    data = writer.backend.read_file(read_path)
    read_back = DataContainer(data.file_obj)

    differences = list(container_diff(read_back, test_container))
    assert not differences, f"Written container does not match original:\n{format_container_diff(differences)}"


@pytest.mark.parametrize("filename", ["myfile", "myfile.hdf5"], ids=["no_ext", "with_ext"])
def test_write_extension(storage_context: StorageTestContext, test_container: DataContainer, filename: str):
    """Writer appends .hdf5 if missing and does not double-append."""
    writer = storage_context.make_writer()
    read_path = storage_context.canonical_path(
        filename if filename.endswith(".hdf5") else f"{filename}.hdf5", destination=True
    )

    writer.write(test_container, filename)

    # Verify the written file exists at the path with .hdf5.
    assert writer.backend.exists(read_path), f"File not found at {read_path}"


@pytest.mark.parametrize("keep_file_names", [False, True], ids=["numbered", "keep_names"])
@pytest.mark.parametrize(
    "file_name_pattern", [None, "data_{index}", "data"], ids=["default_pattern", "custom_pattern", "no_index"]
)
@pytest.mark.parametrize("num_workers", [0, 2, None], ids=["single_worker", "multiple_workers", "default_workers"])
@pytest.mark.parametrize("storage_context", [LocalBackend], indirect=True)
def test_process_parallel_local(
    keep_file_names: bool,
    file_name_pattern: str | None,
    num_workers: int | None,
    tmp_path: Path,
    storage_context: StorageTestContext,
    hdf5_test_corpus: Callable[[int], HDF5TestCorpus],
):
    """process_parallel writes all reader containers to a local destination in parallel."""
    # TODO: extend this test to more remote reader/writer combinations (e.g. local -> remote, remote -> remote, etc.)
    num_files = 3
    corpus = hdf5_test_corpus(num_files)
    storage_context.prepare_files(corpus.files)

    # 2. Create reader and writer
    reader = storage_context.make_reader(parser=HDF5Parser)
    writer = LocalWriter(str(tmp_path / "dest"))

    # 3. Write in parallel
    p = file_name_pattern or "{index}"
    writer.process_parallel(reader, num_workers=num_workers, keep_file_names=keep_file_names, file_name_pattern=p)

    # 4. Verify output
    dest_dir = tmp_path / "dest"
    dest_files = sorted(dest_dir.glob("*.hdf5"))
    assert len(dest_files) == num_files

    # Verify file naming
    if keep_file_names:
        expected_names = {f"file_{i}.hdf5" for i in range(num_files)}
    elif file_name_pattern is not None:
        expected_pattern = file_name_pattern if "{index}" in file_name_pattern else file_name_pattern + "_{index}"
        expected_names = {f"{expected_pattern.replace('{index}', str(i).zfill(1))}.hdf5" for i in range(num_files)}
    else:
        expected_names = {f"{i}.hdf5" for i in range(num_files)}
    actual_names = {f.name for f in dest_files}
    assert actual_names == expected_names, f"Expected {expected_names}, got {actual_names}"

    # Verify each file reads back correctly
    for dest_file in sorted(dest_files):
        read_back = DataContainer(BytesIO(dest_file.read_bytes()))
        # Find matching original by inspecting the index attribute
        index_attr = read_back.get_attribute("/Data", "index")
        assert isinstance(index_attr, int)
        original_idx = index_attr
        expected = DataContainer(BytesIO(corpus.files[f"file_{original_idx}.hdf5"]))
        differences = list(container_diff(read_back, expected))
        assert not differences, (
            f"Container at {dest_file.name} (index {original_idx}) does not match original:\n"
            f"{format_container_diff(differences)}"
        )


@pytest.mark.parametrize("num_files", [1, 10, 12], ids=["unit", "tens", "teens"])
@pytest.mark.parametrize("storage_context", [LocalBackend], indirect=True)
def test_process_parallel_zero_padding(
    num_files: int,
    tmp_path: Path,
    storage_context: StorageTestContext,
    hdf5_test_corpus: Callable[[int], HDF5TestCorpus],
):
    """process_parallel zero-pads indices based on len(str(total_files))."""
    corpus = hdf5_test_corpus(num_files)
    storage_context.prepare_files(corpus.files)
    reader = storage_context.make_reader(parser=HDF5Parser)
    writer = LocalWriter(str(tmp_path / "dest"))
    writer.process_parallel(reader, keep_file_names=False, file_name_pattern="data_{index}")

    dest_files = sorted((tmp_path / "dest").glob("*.hdf5"))
    assert len(dest_files) == num_files

    index_width = len(str(num_files))
    expected_names = {f"data_{str(i).zfill(index_width)}.hdf5" for i in range(num_files)}
    actual_names = {f.name for f in dest_files}
    assert actual_names == expected_names


def test_process_parallel_rejects_invalid_file_name(storage_context: StorageTestContext, test_container: DataContainer):
    """process_parallel rejects source files without a usable base name."""
    # Mock Reader with invalid file names
    reader = MagicMock()
    reader.__len__.return_value = 1
    reader.__getitem__.return_value = test_container
    reader.file_names = [""]
    writer = storage_context.make_writer()

    with pytest.raises(ValueError, match="Invalid file name at index 0:"):
        writer.process_parallel(reader, keep_file_names=True)


@pytest.mark.parametrize("keep_file_names", [False, True])
@pytest.mark.parametrize("file_name_pattern", ["{index}", "copy_{index}", "copy"])
@pytest.mark.parametrize("num_workers", [1, 2])
def test_process_parallel_raw_copy_to_local(
    storage_context: StorageTestContext,
    tmp_path: Path,
    keep_file_names: bool,
    file_name_pattern: str,
    num_workers: int,
):
    """Copy unchanged bytes and mixed extensions from every backend without parsing or buffering."""
    files = {"a space%20.mat": b"\x00\xffnot a MAT file", "b.h5": b"not an HDF5 file"}
    storage_context.prepare_files(files)
    reader = storage_context.make_reader(parser=None)
    writer = LocalWriter(str(tmp_path / "copies"))

    with (
        patch.object(reader, "read_file", side_effect=AssertionError("Must not parse")),
        patch.object(reader.backend, "read_file", side_effect=AssertionError("Must not buffer")),
    ):
        writer.process_parallel(
            reader,
            raw_copy=True,
            keep_file_names=keep_file_names,
            file_name_pattern=file_name_pattern,
            num_workers=num_workers,
            compression="gzip",
            compression_opts=9,
        )

    pattern = file_name_pattern if "{index}" in file_name_pattern else file_name_pattern + "_{index}"
    expected = {
        name if keep_file_names else pattern.replace("{index}", str(index)) + Path(name).suffix: content
        for index, (name, content) in enumerate(files.items())
    }
    assert {file.name: file.read_bytes() for file in (tmp_path / "copies").iterdir()} == expected


@pytest.mark.parametrize("storage_context", [S3Backend, AzureBlobBackend], indirect=True)
@pytest.mark.parametrize("local_source", [False, True])
def test_process_parallel_raw_copy_to_remote(storage_context: StorageTestContext, tmp_path: Path, local_source: bool):
    """Copy local and remote sources to remote destinations with a prefix and no format conversion."""
    name, content = "source %20.mat", b"\x00\xffunchanged"
    if local_source:
        source = tmp_path / name
        source.write_bytes(content)
        reader = LocalReader(str(source))
    else:
        storage_context.prepare_file(name, content)
        reader = storage_context.make_reader(parser=None)

    writer = (
        S3Writer(AWS_BUCKET, "copies")
        if isinstance(storage_context.backend, S3Backend)
        else AzureBlobWriter(AZURE_ACCOUNT_URL, AZURE_CONTAINER, "copies", connection_string=AZURE_CONNECTION_STRING)
    )
    try:
        with patch.object(reader, "read_file", side_effect=AssertionError("Must not parse")):
            writer.process_parallel(reader, raw_copy=True, keep_file_names=True, num_workers=2)
        destination = storage_context.canonical_path(f"copies/{name}")
        assert storage_context.backend.read_file(destination).file_obj.read() == content
    finally:
        writer.backend.close()
        reader.backend.close()


@pytest.mark.parametrize("keep_file_names", [False, True])
@pytest.mark.parametrize("storage_context", [LocalBackend], indirect=True)
def test_process_parallel_raw_copy_snapshots_listing(
    storage_context: StorageTestContext, tmp_path: Path, keep_file_names: bool
):
    """Use one discovery snapshot even when the reader refreshes its listing on every access."""
    source_uri = storage_context.prepare_file("original.mat", b"original")
    reader = storage_context.make_reader(parser=None, listing_ttl=0)
    writer = LocalWriter(str(tmp_path / "copies"))

    with patch.object(reader.backend, "get_file_list", side_effect=[[source_uri], []]) as listing:
        writer.process_parallel(reader, raw_copy=True, keep_file_names=keep_file_names, num_workers=2)
    listing.assert_called_once()
    name = "original.mat" if keep_file_names else "0.mat"
    assert (tmp_path / "copies" / name).read_bytes() == b"original"
