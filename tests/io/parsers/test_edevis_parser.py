import io
import struct
import tarfile
import xml.etree.ElementTree as ET

import pytest
import torch

from pythermondt.io.parsers import EdevisParser
from pythermondt.io.utils import IOPathWrapper


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "CameraSerialNumber": "66900057",
            "IntegrationMode": "IntegrateThenReadout",
            "Filter": "Open",
            "NucTableEntry": "66900057_L1013_Open_ITR_Base.cnuc",
        },
        {"Filter": "Bandpass", "NucTableEntry": "Unkalibriert"},
    ],
    ids=["missing", "all-present", "partial"],
)
def test_edevis_parser_acquisition_metadata(metadata: dict[str, str]):
    """Preserve optional sequence metadata without changing thermal data or units."""
    root = ET.Element("Content")
    ET.SubElement(root, "FileInfo")
    sequence = ET.SubElement(root, "Sequence", id="0")
    sequence_info = ET.SubElement(sequence, "SequenceInfo")
    for name, value in {
        "Window": "0,0,2,1",
        "FrameCount": "1",
        "DataType": "2",
        "BitDepth": "16",
        **metadata,
    }.items():
        ET.SubElement(sequence_info, name).text = f" {value} "
    frame_info = ET.SubElement(sequence, "FrameInfo")
    frame = ET.SubElement(frame_info, "Frame", TarFileHeaderDataOffset="0")
    ET.SubElement(frame, "FrameIndex").text = "0"
    ET.SubElement(frame, "FrameTime").text = "0s"

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        # The first member starts at offset 0; each frame has a 28-byte binary header.
        for name, payload in [
            ("sequence0/f0.bin", bytes(28) + struct.pack("<HH", 100, 200)),
            ("content.xml", ET.tostring(root)),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))

    parsed = EdevisParser.parse(IOPathWrapper(archive))
    attributes = parsed.get_all_attributes("/MetaData")
    for name in ("CameraSerialNumber", "IntegrationMode", "Filter", "NucTableEntry"):
        if name in metadata:
            assert attributes[name] == metadata[name]
        else:
            assert name not in attributes
    assert torch.equal(parsed.get_dataset("/Data/Tdata"), torch.tensor([[[100], [200]]], dtype=torch.uint16))
    assert parsed.get_unit("/Data/Tdata").quantity == "arbitrary"
    assert parsed.get_unit("/MetaData/DomainValues").quantity == "time"
