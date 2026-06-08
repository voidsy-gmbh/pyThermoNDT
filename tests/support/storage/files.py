from pathlib import Path

from pythermondt.io import AzureBlobBackend, BaseBackend, IOPathWrapper, LocalBackend, S3Backend

TEST_FILES = {
    "sample.txt": b"test content",
    "data.bin": b"\x00\x01\x02\x03",
    "large.tiff": b"fake thermal data" * 100,
}

FILE_SCENARIOS = {
    "mixed_types": {
        "sample.txt": b"test content",
        "data.bin": b"\x00\x01\x02\x03",
        "large.tiff": b"fake thermal data" * 100,
    },
    "single_type": {
        "thermal1.tiff": b"data1",
        "thermal2.tiff": b"data2",
        "thermal3.tiff": b"data3",
    },
    "many_files": {f"file{i:03d}.bin": b"x" * i for i in range(15)},
}


def prepare_file(backend: BaseBackend, name: str, content: bytes, tmp_path: Path) -> str:
    """Prepare a file for a backend and return its canonical URI."""
    if isinstance(backend, LocalBackend):
        file_path = tmp_path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        return file_path.as_uri()
    if isinstance(backend, AzureBlobBackend):
        backend.write_file(IOPathWrapper(content), name)
        return f"az://test-container/{name}"
    if isinstance(backend, S3Backend):
        backend.write_file(IOPathWrapper(content), name)
        return f"s3://test-bucket/{name}"
    raise NotImplementedError("Unsupported backend for file preparation.")
