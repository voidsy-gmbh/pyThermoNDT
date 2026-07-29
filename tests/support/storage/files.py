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
