import logging
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from enum import IntEnum
from xml.etree.ElementTree import Element

import torch

from ...data import DataContainer, ThermoContainer
from ...data.units import arbitrary, hertz, second
from ...io.utils import IOPathWrapper
from .base_parser import BaseParser

logger = logging.getLogger(__name__)

TAR_HEADER_SIZE = 512  # Size of the TAR header in bytes ==> needs to be skipped when reading the data


class DataType(IntEnum):
    SHEAROGRAPHY_IMAGE = 0
    INTENSITY_IMAGE = 2
    TEMPERATURE_IMAGE = 5
    COMPLEX_IMAGE = 13


class EdevisParser(BaseParser):
    supported_extensions = (".di", ".OTvis", ".ITvisPulse", ".ThermoVis", ".PTvis")

    @staticmethod
    def parse(data: IOPathWrapper) -> DataContainer:  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        """Parses the data from the given IOPathWrapper object into a DataContainer object.

        The IOPathWrapper object must contain an Edevis measurement archive in a supported format.

        Parameters:
            data (IOPathWrapper): IOPathWrapper object containing the data to be parsed.

        Returns:
            DataContainer: The parsed data as a DataContainer object.

        Raises:
            ValueError: If the file in the given IOPathWrapper object is corrupted
        """
        # Extract the file object from the IOPathWrapper
        data_bytes = data.file_obj

        # Check if the BytesIO object is empty
        if data_bytes.getbuffer().nbytes == 0:
            raise ValueError("The given BytesIO object is empty.")

        # Create an empty ThermoContainer - with predefined structure
        container = ThermoContainer()

        # Add source as an attribute
        container.add_attributes(path="/MetaData", Source="Measurement: Edevis")

        # Parse data
        try:
            # Open file in tar mode and extract content.xml
            with tarfile.open(fileobj=data_bytes, mode="r:", ignore_zeros=True) as tar_file:
                # Check if content.xml exists
                try:
                    content_extracted = tar_file.extractfile("content.xml")
                except KeyError as e:
                    raise ValueError("File seems corrupted! content.xml not found.") from e

                if content_extracted is None:
                    raise ValueError("File seems corrupted! content.xml not found.")

                # Parse the extracted content.xml
                content = ET.parse(content_extracted)

                # Extract metadata from FileInfo
                file_info = content.find("FileInfo")
                if file_info is None:
                    raise ValueError("File seems corrupted! No FileInfo node found.")

                excitation_fields = [
                    "ExcitationSignalType",
                    "ExcitationFrequency",
                    "ExcitationDelay",
                    "ExcitationPulseLength",
                    "ExcitationAmplitude",
                    "ExcitationPeriods",
                    "ExcitationPhaseShift",
                    "ExcitationOnTime",
                    "ExcitationAsymmetry",
                    "ExcitationArbitrarySource",
                ]

                # Extract target fields from FileInfo
                target_fields = [
                    "UniqueIdentifier",
                    "CreationDate",
                    "CreatingVersion",
                    "ModuleName",
                    "ModuleType",
                    "SensorType",
                    "MeasurementMode",
                    "IntegrationTime",
                    "CameraSynchronization",
                    "ExcitationDeviceSelection",
                    *excitation_fields,
                ]
                metadata = extract_metadata_from_xml(file_info, target_fields)

                # Process the Sequence data
                sequences = {int(seq.attrib.get("id", -1)): seq for seq in content.findall("Sequence")}

                # TODO: For now only the first sequence is processed ==> Extend to support multiple sequences in future
                if ignored_ids := sorted(seq_id for seq_id in sequences if seq_id != 0):
                    logger.info("Parsing sequence 0 only; ignoring sequences %s.", ignored_ids)
                sequences = {seq_id: seq for seq_id, seq in sequences.items() if seq_id == 0}

                if len(sequences) == 0:
                    raise ValueError("File seems corrupted! No Sequence node found.")

                for seq_id, sequence in sequences.items():
                    # Add the sequence ID to the metadata
                    metadata["SequenceID"] = seq_id

                    # Process sequence info
                    sequence_info = sequence.find("SequenceInfo")
                    if sequence_info is None:
                        raise ValueError(f"Sequence {seq_id} seems corrupted! No SequenceInfo node found.")

                    target_fields = [
                        "CameraManufacturer",
                        "CameraName",
                        "CameraSerialNumber",
                        "Lens",
                        "Filter",
                        "NucTableEntry",
                        "FrameCount",
                        "FrameRate",
                        "Window",
                        "MaxFrameRate",
                        "MaxWindow",
                        "DataType",
                        "BitDepth",
                        "IntegrationTime",
                        "IntegrationMode",
                        *excitation_fields,
                    ]
                    sequence_metadata = extract_metadata_from_xml(sequence_info, target_fields)
                    metadata.update(sequence_metadata)

                    # Determine width and height from the sequence info
                    if "Window" not in metadata:
                        raise ValueError(f"Sequence {seq_id} seems corrupted! No Window node found in SequenceInfo.")
                    # Window is in format "X_start,Y_Start, Width,Height"
                    width = int(metadata["Window"].split(",")[2])
                    height = int(metadata["Window"].split(",")[3])

                    # Get frame info
                    frame_info = sequence.find("FrameInfo")
                    if frame_info is None:
                        raise ValueError(f"Sequence {seq_id} seems corrupted! No FrameInfo node found.")

                    # Find all frames in FrameInfo node
                    frames = frame_info.findall("Frame")
                    if not frames:
                        raise ValueError(f"Sequence {seq_id} seems corrupted! No Frame nodes found in FrameInfo.")

                    # Get DataType and BitDepth to determine how to process the data
                    data_type = int(metadata.get("DataType", -1))
                    bit_depth = int(metadata.get("BitDepth", -1))

                    # Extract the LUT if available
                    tar_offset_lut = None
                    if "TarFileHeaderCalibrationOffset" in sequence_info.attrib:
                        tar_offset_lut = int(sequence_info.attrib["TarFileHeaderCalibrationOffset"])

                        # Extract and store the LUT if available
                        if tar_offset_lut and bit_depth == 16:
                            # Reset file position to start
                            data_bytes.seek(0)

                            # Skip to LUT position
                            data_bytes.seek(tar_offset_lut + TAR_HEADER_SIZE)

                            # Extract LUT data
                            lut_size = 2**16  # If bit_depth is 16 ==> UINT16 ==> 65536 entries
                            buffer = bytearray(data_bytes.read(lut_size * 4))  # 4 bytes per LUT entry assuming float32
                            lut_data = torch.frombuffer(buffer, dtype=torch.float32)

                            # Convert LUT data to Kelvin because ThermoContainer stores LUT in Kelvin
                            container.update_dataset("/MetaData/LookUpTable", lut_data + 273.15)
                        else:
                            logger.warning(
                                "Ignoring LUT of sequence %s: offset %s with BitDepth %s; "
                                "only 16-bit data with a nonzero offset is supported.",
                                seq_id,
                                tar_offset_lut,
                                bit_depth,
                            )

                    # Define supported bit depths for each data type
                    supported_bit_depths = {
                        13: [32, 64, 128],  # Complex images
                        "default": [16, 32, 64],  # Other data types
                    }

                    # Validate bit depth
                    valid_depths = supported_bit_depths.get(data_type, supported_bit_depths["default"])
                    if bit_depth not in valid_depths:
                        raise ValueError(
                            f"Unsupported BitDepth: {bit_depth} for DataType: {data_type}. "
                            f"Supported values are {valid_depths}."
                        )

                    # Map bit depth to corresponding torch data type
                    if data_type == DataType.COMPLEX_IMAGE:  # Complex images need complex dtypes
                        frame_dtype = {32: torch.complex32, 64: torch.complex64, 128: torch.complex128}
                    else:
                        frame_dtype = {16: torch.uint16, 32: torch.float32, 64: torch.float64}

                    # Each frame is stored in a separate file. Because of the block size of the tar file,
                    # we need to skip the header size of the tar file (512 bytes) and the size of the frame header
                    # The frame header size is not explicitly given, so we need to calculate it based on the frame size
                    # Get the size of the first frame to be able to dynamically calculate the frame header size
                    first_idx = frames[0].findtext("FrameIndex", default=None)
                    indexed_frame_name = f"sequence{seq_id}/f{first_idx}.bin"
                    zero_based_frame_name = f"sequence{seq_id}/f0.bin"

                    try:
                        file_size = tar_file.getmember(indexed_frame_name).size
                    except KeyError:
                        try:
                            # ThermoVis can use zero-based file names unrelated to FrameIndex.
                            file_size = tar_file.getmember(zero_based_frame_name).size
                        except KeyError as error:
                            msg = (
                                f"Frames in sequence {seq_id} seem corrupted. "
                                f"Neither {indexed_frame_name!r} nor {zero_based_frame_name!r} was found."
                            )
                            raise ValueError(msg) from error

                    # Calculate bytes per pixel and frame size
                    bytes_per_pixel = bit_depth // 8
                    frame_size_bytes = width * height * bytes_per_pixel

                    # Dynamically calculate the frame header size
                    header_size = file_size - frame_size_bytes

                    # Pre-allocate variables and arrays for better performance
                    domain_unit = None
                    frame_unit = None
                    num_frames = len(frames)
                    domain_values = torch.zeros(num_frames, dtype=torch.float32)
                    frame_data = torch.zeros((height, width, num_frames), dtype=frame_dtype[bit_depth])

                    for i, frame in enumerate(frames):
                        # Extract frame attributes
                        offset = int(frame.attrib.get("TarFileHeaderDataOffset", -1))

                        # Handle different data types
                        match data_type:
                            case DataType.SHEAROGRAPHY_IMAGE:
                                raise NotImplementedError("Shearography Image data type is not implemented yet.")
                            case DataType.INTENSITY_IMAGE:
                                domain_values[i] = _domain_value(frame, "FrameTime", "s", seq_id, i)
                                domain_unit = second

                                # TAR header offset = TarFileHeaderDataOffset
                                # frame record     = TarFileHeaderDataOffset + 512
                                # pixel data       = TarFileHeaderDataOffset + 512 + 28
                                # pixel length     = 640 * 512 * 2

                                # Read frame data
                                data_bytes.seek(offset + TAR_HEADER_SIZE + header_size)
                                buffer = bytearray(data_bytes.read(frame_size_bytes))
                                frame_data[:, :, i] = torch.frombuffer(buffer, dtype=frame_dtype[bit_depth]).reshape(
                                    height, width
                                )
                                frame_unit = arbitrary

                            case DataType.TEMPERATURE_IMAGE:
                                raise NotImplementedError("Temperature Image data type is not implemented yet.")
                            case DataType.COMPLEX_IMAGE:
                                # Read domain values from the data
                                domain_values[i] = _domain_value(frame, "FourierFrequency", "Hz", seq_id, i)
                                domain_unit = hertz

                                # Read frame data
                                data_bytes.seek(offset + TAR_HEADER_SIZE + header_size)
                                buffer = bytearray(data_bytes.read(frame_size_bytes))
                                frame_data[:, :, i] = torch.frombuffer(buffer, dtype=frame_dtype[bit_depth]).reshape(
                                    height, width
                                )
                                frame_unit = arbitrary  # Complex images do not have a specific unit

                            case _:
                                types = ", ".join(f"{t.value} ({t.name})" for t in DataType)
                                raise ValueError(f"Unsupported DataType: {data_type}. Supported values are {types}.")

                    # Check the complete time axis before storing it.
                    if domain_unit == second:
                        invalid_times = ~torch.isfinite(domain_values)
                        invalid_times[1:] |= domain_values[1:] <= domain_values[:-1]
                        if invalid_times.any():
                            frame_index = int(invalid_times.nonzero()[0, 0].item())
                            raise ValueError(
                                f"Sequence {seq_id}, frame {frame_index}: invalid FrameTime "
                                f"{domain_values[frame_index].item():g} s. Frame timestamps must be finite "
                                "and strictly increasing."
                            )

                    # Update container
                    if domain_unit:
                        container.update_dataset("/MetaData/DomainValues", domain_values)
                        container.update_unit("/MetaData/DomainValues", domain_unit)
                    if frame_unit:
                        container.update_dataset("/Data/Tdata", frame_data)
                        container.update_unit("/Data/Tdata", frame_unit)
                    logger.debug(
                        "Parsed sequence %s: DataType %s, BitDepth %s, %s frames of %s x %s pixels, LUT %s.",
                        seq_id,
                        data_type,
                        bit_depth,
                        num_frames,
                        height,
                        width,
                        "found" if container.get_dataset("/MetaData/LookUpTable").numel() else "absent",
                    )

                    if domain_unit == second:
                        excitation_signal = _build_excitation_signal(seq_id, domain_values, metadata, sequence_metadata)
                        if excitation_signal is not None:
                            container.update_dataset("/MetaData/ExcitationSignal", excitation_signal)

                # Add metadata to the container
                if metadata:
                    container.add_attributes("/MetaData", **metadata)
                return container

        except Exception as e:
            raise ValueError(f"Error parsing Edevis file: {e}") from e


def _build_excitation_signal(
    seq_id: int,
    frame_times: torch.Tensor,
    metadata: Mapping[str, str | int],
    sequence_metadata: Mapping[str, str],
) -> torch.Tensor | None:
    """Construct a frame-aligned signal from timing metadata, when supported."""
    if metadata.get("MeasurementMode") != "Pulse" or metadata.get("ExcitationDeviceSelection") != "Hensel Flash":
        return None

    # Read timing from SequenceInfo because the file-level settings can differ.
    try:
        delay = float(sequence_metadata.get("ExcitationDelay", "").removesuffix("s"))
        pulse_length = float(sequence_metadata.get("ExcitationPulseLength", "").removesuffix("s"))
    except ValueError:
        logger.warning(
            "Skipping Hensel Flash marker for sequence %s: missing or invalid "
            "ExcitationDelay or ExcitationPulseLength in SequenceInfo.",
            seq_id,
        )
        return None

    if not (delay >= 0 and pulse_length > 0):
        logger.warning(
            "Skipping Hensel Flash marker for sequence %s: expected a nonnegative delay and positive "
            "pulse length; got %s s and %s s.",
            seq_id,
            delay,
            pulse_length,
        )
        return None
    flash_end = delay + pulse_length
    # Allow one microsecond of timestamp rounding at the estimated pulse end.
    frames_to_discard = int((frame_times <= flash_end + 1e-6).sum().item())
    if not 0 < frames_to_discard < frame_times.numel():
        logger.warning(
            "Skipping Hensel Flash marker for sequence %s: no frames bracket the estimated end at %s s.",
            seq_id,
            flash_end,
        )
        return None

    # Estimate the flash signal as a Dirac pulse constructed from metadata.
    # Detecting the flash end is the responsibility of RemoveFlash.
    marker_index = frames_to_discard - 1
    excitation_signal = torch.zeros_like(frame_times)
    excitation_signal[marker_index] = 1
    logger.debug(
        "Hensel Flash marker for sequence %s: frame %s, estimated pulse end %s s.",
        seq_id,
        marker_index,
        flash_end,
    )
    return excitation_signal


def _domain_value(frame: Element, field: str, unit_suffix: str, seq_id: int, index: int) -> float:
    """Read a frame's domain value, e.g. "0.02s"; a missing value would silently become 0."""
    text = (frame.findtext(field) or "").strip()
    if not text:
        raise ValueError(f"Sequence {seq_id}, frame {index} seems corrupted! No {field} found.")
    return float(text.split(unit_suffix)[0])


def extract_metadata_from_xml(xml_root: Element, target_fields: Sequence[str] | None) -> dict:
    """Extract the stripped text of child elements of the XML root element.

    Args:
        xml_root (Element): The XML element whose direct children hold the metadata.
        target_fields (Sequence[str] | None): Child tags to extract; missing or empty fields are skipped.
            If None, all children with text are extracted.

    Returns:
        dict[str, str]: Extracted field names and their text values.
    """
    # Check for None explicitly: an empty list means "extract nothing", not "extract all"
    if target_fields is None:
        return {child.tag: child.text.strip() for child in xml_root if child.text}

    # find() per field is faster than looping over all children for few fields, and keeps their order
    metadata = {}
    for field_name in target_fields:
        element = xml_root.find(field_name)
        if element is not None and element.text:
            metadata[field_name] = element.text.strip()
    return metadata
