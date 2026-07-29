from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import pytest

from pythermondt.data import DataContainer
from pythermondt.io import LocalBackend
from pythermondt.io.parsers import HDF5Parser
from pythermondt.writers import LocalWriter
from tests.support.storage import StorageTestContext
from tests.utils import containers_equal
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

    assert containers_equal(read_back, test_container), "Written container does not match original"


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
@pytest.mark.parametrize("storage_context", [LocalBackend], indirect=True)
def test_process_parallel_local(
    keep_file_names: bool,
    file_name_pattern: str | None,
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
    writer.process_parallel(reader, keep_file_names=keep_file_names, file_name_pattern=p)

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
        assert containers_equal(read_back, expected), (
            f"Container at {dest_file.name} (index {original_idx}) does not match original"
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
