import io
import logging
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from re import escape

import numpy as np
import pytest
import torch

from pythermondt.io.parsers import EdevisParser
from pythermondt.io.parsers.edevis_parser import extract_metadata_from_xml
from pythermondt.io.utils import IOPathWrapper

# Edevis stores each frame with a binary header before the pixel data; the parser infers its size.
FRAME_HEADER_SIZE = 28


def _build_edevis_file(
    frames: Sequence[np.ndarray],
    domain_values: Sequence[str],
    *,
    sequence_info: dict[str, str] | None = None,
    frame_indices: Sequence[int] | None = None,
    frame_names: Sequence[str] | None = None,
    lut_celsius: np.ndarray | None = None,
    remove: Sequence[str] = (),
    edit_content: Callable[[ET.Element], object] | None = None,
) -> IOPathWrapper:
    """Build a minimal single-sequence edevis TAR archive.

    Args:
        frames: Frames of shape (H, W); uint16 creates intensity data, complex64 creates complex data.
        domain_values: FrameTime (intensity) or FourierFrequency (complex) text per frame, e.g. "0.02s".
        sequence_info: Extra or overriding SequenceInfo fields.
        frame_indices: Camera FrameIndex per frame; defaults to 0, 1, ...
        frame_names: TAR member names; defaults to "sequence0/f<FrameIndex>.bin".
        lut_celsius: 65536 float32 temperatures in degrees Celsius, stored as calibration data.
        remove: XML element paths to delete from content.xml, e.g. "Sequence/FrameInfo".
        edit_content: Called with the content.xml root before it is written, for other changes.
    """
    height, width = frames[0].shape
    is_complex = np.iscomplexobj(frames[0])
    frame_indices = frame_indices or range(len(frames))
    frame_names = frame_names or [f"sequence0/f{index}.bin" for index in frame_indices]
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:

        def add(name: str, payload: bytes) -> int:
            # The XML references members by the offset of their TAR header.
            offset = tar.offset
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
            return offset

        frame_offsets = [
            add(name, bytes(FRAME_HEADER_SIZE) + frame.tobytes())
            for name, frame in zip(frame_names, frames, strict=True)
        ]
        lut_offset = (
            add("sequence0/calibration.bin", lut_celsius.astype("<f4").tobytes()) if lut_celsius is not None else None
        )

        root = ET.Element("Content")
        ET.SubElement(root, "FileInfo")
        sequence = ET.SubElement(root, "Sequence", id="0")
        info = ET.SubElement(sequence, "SequenceInfo")
        if lut_offset is not None:
            info.set("TarFileHeaderCalibrationOffset", str(lut_offset))
        fields = {
            "Window": f"0,0,{width},{height}",
            "FrameCount": str(len(frames)),
            "DataType": "13" if is_complex else "2",
            "BitDepth": str(frames[0].itemsize * 8),
            **(sequence_info or {}),
        }
        for name, value in fields.items():
            ET.SubElement(info, name).text = f" {value} "
        frame_info = ET.SubElement(sequence, "FrameInfo")
        for index, offset, domain_value in zip(frame_indices, frame_offsets, domain_values, strict=True):
            frame = ET.SubElement(frame_info, "Frame", TarFileHeaderDataOffset=str(offset))
            ET.SubElement(frame, "FrameIndex").text = str(index)
            ET.SubElement(frame, "FourierFrequency" if is_complex else "FrameTime").text = domain_value
        for path in remove:
            parent_path, _, tag = path.rpartition("/")
            parent = root.find(parent_path) if parent_path else root
            for element in parent.findall(tag):
                parent.remove(element)
        if edit_content is not None:
            edit_content(root)
        add("content.xml", ET.tostring(root))
    return IOPathWrapper(archive)


@pytest.fixture
def intensity_frames() -> list[np.ndarray]:
    """Two distinct 2 x 3 uint16 frames."""
    return [np.arange(6, dtype="<u2").reshape(2, 3) + offset for offset in (100, 200)]


def test_intensity_frames_and_time_domain(intensity_frames: list[np.ndarray]):
    """Intensity frames keep raw uint16 values in H x W x T order with frame times in seconds."""
    parsed = EdevisParser.parse(_build_edevis_file(intensity_frames, ["0s", "0.02s"]))

    tdata = parsed.get_dataset("/Data/Tdata")
    assert tdata.dtype == torch.uint16
    assert torch.equal(tdata, torch.from_numpy(np.stack(intensity_frames, axis=-1)))
    assert torch.allclose(parsed.get_dataset("/MetaData/DomainValues"), torch.tensor([0.0, 0.02]))
    assert parsed.get_unit("/Data/Tdata").quantity == "arbitrary"
    assert parsed.get_unit("/MetaData/DomainValues").name == "second"


def test_complex_frames_and_frequency_domain():
    """Complex results keep complex64 values with Fourier frequencies in hertz."""
    frames = [np.array([[1 + 2j, -3j]], dtype=np.complex64), np.array([[0.5, 4 - 1j]], dtype=np.complex64)]

    parsed = EdevisParser.parse(_build_edevis_file(frames, ["0.1Hz", "0.2Hz"]))

    tdata = parsed.get_dataset("/Data/Tdata")
    assert tdata.dtype == torch.complex64
    assert torch.equal(tdata, torch.from_numpy(np.stack(frames, axis=-1)))
    assert torch.allclose(parsed.get_dataset("/MetaData/DomainValues"), torch.tensor([0.1, 0.2]))
    assert parsed.get_unit("/MetaData/DomainValues").name == "hertz"


def test_lut_is_converted_to_kelvin(intensity_frames: list[np.ndarray]):
    """Edevis stores the 16-bit lookup table in degrees Celsius; the container stores kelvin."""
    lut_celsius = np.linspace(-20.0, 120.0, 2**16, dtype=np.float32)

    parsed = EdevisParser.parse(_build_edevis_file(intensity_frames, ["0s", "0.02s"], lut_celsius=lut_celsius))

    lut = parsed.get_dataset("/MetaData/LookUpTable")
    assert lut.shape == (2**16,)
    # float32 storage limits the precision to about 3e-5 K near 400 K.
    assert torch.allclose(lut.double(), torch.from_numpy(lut_celsius).double() + 273.15, atol=1e-4)
    assert parsed.get_unit("/MetaData/LookUpTable").name == "kelvin"


def test_lut_is_empty_without_calibration_data(intensity_frames: list[np.ndarray]):
    """Files without calibration data keep the empty lookup table."""
    parsed = EdevisParser.parse(_build_edevis_file(intensity_frames, ["0s", "0.02s"]))

    assert parsed.get_dataset("/MetaData/LookUpTable").numel() == 0


@pytest.mark.parametrize(
    ("frames", "offset"),
    [
        ([np.zeros((1, 2), dtype="<f4")] * 2, None),
        ([np.zeros((1, 2), dtype="<u2")] * 2, 0),
    ],
    ids=["float32-data", "zero-offset"],
)
def test_unsupported_lut_is_ignored_with_warning(
    caplog: pytest.LogCaptureFixture, frames: list[np.ndarray], offset: int | None
):
    """A LUT only maps 16-bit raw values; other cases keep the empty LUT and log why."""

    def set_offset(root: ET.Element) -> None:
        if offset is not None:
            root.find("Sequence/SequenceInfo").set("TarFileHeaderCalibrationOffset", str(offset))

    archive = _build_edevis_file(
        frames, ["0s", "0.02s"], lut_celsius=np.zeros(2**16, dtype=np.float32), edit_content=set_offset
    )

    with caplog.at_level(logging.WARNING, logger="pythermondt.io.parsers.edevis_parser"):
        parsed = EdevisParser.parse(archive)

    assert parsed.get_dataset("/MetaData/LookUpTable").numel() == 0
    assert "Ignoring LUT of sequence 0" in caplog.text


def test_other_sequences_are_logged(caplog: pytest.LogCaptureFixture, intensity_frames: list[np.ndarray]):
    """Only sequence 0 is parsed; ignored sequences are reported."""
    archive = _build_edevis_file(
        intensity_frames, ["0s", "0.02s"], edit_content=lambda root: ET.SubElement(root, "Sequence", id="1")
    )

    with caplog.at_level(logging.INFO, logger="pythermondt.io.parsers.edevis_parser"):
        EdevisParser.parse(archive)

    assert "ignoring sequences [1]" in caplog.text


@pytest.mark.parametrize(
    ("frame_indices", "frame_names"),
    [
        ([141603, 141604], ["sequence0/f141603.bin", "sequence0/f141604.bin"]),
        ([4296304903, 4296304904], ["sequence0/f0.bin", "sequence0/f1.bin"]),
    ],
    ids=["indexed", "zero-based"],
)
def test_frame_file_naming_schemes(
    intensity_frames: list[np.ndarray], frame_indices: list[int], frame_names: list[str]
):
    """Frame files can be named after the camera FrameIndex or numbered from zero; both schemes occur."""
    archive = _build_edevis_file(
        intensity_frames, ["0s", "0.02s"], frame_indices=frame_indices, frame_names=frame_names
    )

    parsed = EdevisParser.parse(archive)

    assert torch.equal(parsed.get_dataset("/Data/Tdata"), torch.from_numpy(np.stack(intensity_frames, axis=-1)))


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
def test_optional_acquisition_metadata(intensity_frames: list[np.ndarray], metadata: dict[str, str]):
    """Optional sequence fields are stored without surrounding whitespace and omitted when absent."""
    parsed = EdevisParser.parse(_build_edevis_file(intensity_frames, ["0s", "0.02s"], sequence_info=metadata))

    attributes = parsed.get_all_attributes("/MetaData")
    for name in ("CameraSerialNumber", "IntegrationMode", "Filter", "NucTableEntry"):
        assert attributes.get(name) == metadata.get(name)


@pytest.mark.parametrize(
    ("frames", "domain_values", "message"),
    [
        ([np.zeros((1, 2), dtype="<u2")] * 2, ["0s", " "], "Sequence 0, frame 1 seems corrupted! No FrameTime found"),
        (
            [np.zeros((1, 2), dtype=np.complex64)] * 2,
            ["0.1Hz", ""],
            "Sequence 0, frame 1 seems corrupted! No FourierFrequency found",
        ),
    ],
    ids=["frame-time", "fourier-frequency"],
)
def test_empty_domain_value_raises(frames: list[np.ndarray], domain_values: list[str], message: str):
    """A frame without its time or frequency would otherwise get the domain value 0."""
    with pytest.raises(ValueError, match=escape(message)):
        EdevisParser.parse(_build_edevis_file(frames, domain_values))


@pytest.mark.parametrize(
    ("frame_times", "invalid_index"),
    [
        (["0.02s", "0s"], 1),
        (["0s", "0s"], 1),
        (["nans", "0.02s"], 0),
        (["0s", "infs"], 1),
    ],
    ids=["decreasing", "duplicate", "nan", "infinite"],
)
def test_invalid_frame_times_raise(intensity_frames: list[np.ndarray], frame_times: list[str], invalid_index: int):
    """Invalid timestamps fail even when the recording does not need a flash marker."""
    archive = _build_edevis_file(intensity_frames, frame_times)

    with pytest.raises(ValueError, match=rf"Sequence 0, frame {invalid_index}: invalid FrameTime"):
        EdevisParser.parse(archive)


@pytest.mark.parametrize(
    ("target_fields", "expected"),
    [
        (["Window", "Missing", "Comment"], {"Window": "0,0,2,1"}),
        (None, {"Window": "0,0,2,1", "Lens": "Unknown"}),
        ([], {}),
    ],
    ids=["selected-fields", "all-fields", "no-fields"],
)
def test_extract_metadata_from_xml(target_fields: list[str] | None, expected: dict[str, str]):
    """Fields are stripped; missing and empty fields are skipped, also when they come first."""
    info = ET.fromstring("<SequenceInfo><Comment/><Window> 0,0,2,1 </Window><Lens>Unknown</Lens></SequenceInfo>")

    assert extract_metadata_from_xml(info, target_fields) == expected


def test_empty_file_raises():
    """An empty buffer is rejected before archive parsing."""
    with pytest.raises(ValueError, match=escape("The given BytesIO object is empty.")):
        EdevisParser.parse(IOPathWrapper(io.BytesIO()))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence_info": {"BitDepth": "8"}}, "Unsupported BitDepth: 8 for DataType: 2"),
        ({"sequence_info": {"DataType": "7"}}, "Unsupported DataType: 7"),
        ({"sequence_info": {"DataType": "0"}}, "Shearography Image data type is not implemented yet."),
        ({"sequence_info": {"DataType": "5"}}, "Temperature Image data type is not implemented yet."),
        ({"frame_names": ["sequence0/other0.bin", "sequence0/other1.bin"]}, "Frames in sequence 0 seem corrupted"),
    ],
    ids=["bit-depth", "data-type", "shearography", "temperature", "missing-frame-file"],
)
def test_invalid_archive_raises(intensity_frames: list[np.ndarray], kwargs: dict, message: str):
    """Invalid or unsupported sequence settings and missing frame files raise descriptive errors."""
    with pytest.raises(ValueError, match=escape(message)):
        EdevisParser.parse(_build_edevis_file(intensity_frames, ["0s", "0.02s"], **kwargs))


@pytest.mark.parametrize(
    ("remove", "message"),
    [
        ("FileInfo", "No FileInfo node found"),
        ("Sequence", "No Sequence node found"),
        ("Sequence/SequenceInfo", "Sequence 0 seems corrupted! No SequenceInfo node found"),
        ("Sequence/SequenceInfo/Window", "Sequence 0 seems corrupted! No Window node found"),
        ("Sequence/FrameInfo", "Sequence 0 seems corrupted! No FrameInfo node found"),
        ("Sequence/FrameInfo/Frame", "Sequence 0 seems corrupted! No Frame nodes found"),
        ("Sequence/FrameInfo/Frame/FrameTime", "Sequence 0, frame 0 seems corrupted! No FrameTime found"),
    ],
    ids=["file-info", "sequence", "sequence-info", "window", "frame-info", "frames", "frame-time"],
)
def test_missing_xml_part_raises(intensity_frames: list[np.ndarray], remove: str, message: str):
    """Each required part of content.xml is checked and named in the error."""
    with pytest.raises(ValueError, match=escape(message)):
        EdevisParser.parse(_build_edevis_file(intensity_frames, ["0s", "0.02s"], remove=[remove]))


@pytest.mark.parametrize("content_type", [None, tarfile.DIRTYPE], ids=["missing", "directory"])
def test_archive_without_content_raises(content_type: bytes | None):
    """Archives without a readable content.xml are reported as corrupted."""
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        member = tarfile.TarInfo("sequence0/f0.bin")
        member.size = 4
        tar.addfile(member, io.BytesIO(bytes(4)))
        if content_type is not None:
            content = tarfile.TarInfo("content.xml")
            content.type = content_type
            tar.addfile(content)

    with pytest.raises(ValueError, match=escape("content.xml not found")):
        EdevisParser.parse(IOPathWrapper(archive))
