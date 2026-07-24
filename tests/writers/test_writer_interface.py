from io import BytesIO
from pathlib import Path

import pytest
import torch

from pythermondt.data import DataContainer
from pythermondt.io.parsers import HDF5Parser
from pythermondt.readers import LocalReader
from pythermondt.writers import LocalWriter
from tests.utils import containers_equal, make_container
from tests.writers.conftest import WriterTestContext


def test_write_round_trip(writer_config: WriterTestContext, test_container: DataContainer):
    """Write a DataContainer and verify it reads back identically."""
    writer = writer_config.writer
    filename = "test_file"
    read_path = writer_config.read_path(filename)

    writer.write(test_container, filename)

    # Read back via the writer's own backend.
    data = writer.backend.read_file(read_path)
    read_back = DataContainer(data.file_obj)

    assert containers_equal(read_back, test_container), "Written container does not match original"


@pytest.mark.parametrize("filename", ["myfile", "myfile.hdf5"], ids=["no_ext", "with_ext"])
def test_write_extension(writer_config: WriterTestContext, test_container: DataContainer, filename: str):
    """Writer appends .hdf5 if missing and does not double-append."""
    writer = writer_config.writer
    read_path = writer_config.read_path(filename)

    writer.write(test_container, filename)

    # Verify the written file exists at the path with .hdf5.
    assert writer.backend.exists(read_path), f"File not found at {read_path}"


@pytest.mark.parametrize("keep_file_names", [False, True], ids=["numbered", "keep_names"])
@pytest.mark.parametrize("file_name_pattern", [None, "data_{index}"], ids=["default_pattern", "custom_pattern"])
def test_process_parallel_local(keep_file_names: bool, file_name_pattern: str | None, tmp_path: Path):
    """process_parallel writes all reader containers to a local destination in parallel."""
    # TODO: extend this test to more remote reader/writer combinations (e.g. local -> remote, remote -> remote, etc.)
    # 1. Seed source files
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    num_files = 3
    containers = []
    for i in range(num_files):
        c = make_container(("/Data", f"t{i}", torch.rand(2, 2)))
        c.add_attribute("/Data", "index", i)
        c.save_to_hdf5(str(source_dir / f"file_{i}.hdf5"))
        containers.append(c)

    # 2. Create reader and writer
    reader = LocalReader(str(source_dir), parser=HDF5Parser)
    writer = LocalWriter(str(tmp_path / "dest"))

    # 3. Write in parallel
    if file_name_pattern is not None:
        writer.process_parallel(reader, keep_file_names=keep_file_names, file_name_pattern=file_name_pattern)
    else:
        writer.process_parallel(reader, keep_file_names=keep_file_names)

    # 4. Verify output
    dest_dir = tmp_path / "dest"
    dest_files = sorted(dest_dir.glob("*.hdf5"))
    assert len(dest_files) == num_files

    # Verify file naming
    if keep_file_names:
        expected_names = {f"file_{i}.hdf5" for i in range(num_files)}
    elif file_name_pattern is not None:
        expected_names = {f"{file_name_pattern.replace('{index}', str(i).zfill(1))}.hdf5" for i in range(num_files)}
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
        assert containers_equal(read_back, containers[original_idx]), (
            f"Container at {dest_file.name} (index {original_idx}) does not match original"
        )
