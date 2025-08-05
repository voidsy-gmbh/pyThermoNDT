import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from io import BytesIO
from xml.etree.ElementTree import Element

import torch

from ...data import DataContainer, ThermoContainer, Units
from ...io.utils import IOPathWrapper
from .base_parser import BaseParser

TAR_HEADER_SIZE = 512  # Size of the TAR header in bytes ==> needs to be skipped when reading the data


class EdevisParser(BaseParser):
    supported_extensions = (".OTvis",)

    @staticmethod
    def parse(data: IOPathWrapper) -> DataContainer:
        """Parses the data from the given IOPathWrapper object into a DataContainer object.

        The IOPathWrapper object must contain a .OTvis file with Measurement data from Edevis.

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

                # Extract and parse XML data
                xml_content = content_extracted.read()
                xml_data = ET.parse(BytesIO(xml_content))

                # Extract metadata from FileInfo
                file_info = xml_data.find("FileInfo")
                if file_info is None:
                    raise ValueError("File seems corrupted! No FileInfo node found.")

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
                ]
                metadata = extract_metadata_from_xml(file_info, target_fields)

                # Process the Sequence data
                sequences = {int(seq.attrib.get("id", -1)): seq for seq in xml_data.findall("Sequence")}

                # TODO: For now only the first sequence is processed ==> Extend to support multiple sequences in future
                sequences = {id: seq for id, seq in sequences.items() if id == 0}

                if len(sequences) == 0:
                    raise ValueError("File seems corrupted! No Sequence node found.")

                for id, sequence in sequences.items():
                    # Add the sequence ID to the metadata
                    metadata["SequenceID"] = id

                    # Process sequence info
                    sequence_info = sequence.find("SequenceInfo")
                    if sequence_info is None:
                        raise ValueError(f"Sequence {id} seems corrupted! No SequenceInfo node found.")

                    target_fields = [
                        "CameraManufacturer",
                        "CameraName",
                        "Lens",
                        "FrameCount",
                        "FrameRate",
                        "Window",
                        "MaxFrameRate",
                        "MaxWindow",
                        "DataType",
                        "BitDepth",
                        "IntegrationTime",
                    ]
                    metadata.update(extract_metadata_from_xml(sequence_info, target_fields))

                    # Determine width and height from the sequence info
                    if "Window" not in metadata:
                        raise ValueError(f"Sequence {id} seems corrupted! No Window node found in SequenceInfo.")
                    # Window is in format "X_start,Y_Start, Width,Height"
                    width = int(metadata["Window"].split(",")[2])
                    height = int(metadata["Window"].split(",")[3])

                    # Extract the LUT if available
                    tar_offset_lut = None
                    if "TarFileHeaderCalibrationOffset" in sequence_info.attrib:
                        tar_offset_lut = int(sequence_info.attrib["TarFileHeaderCalibrationOffset"])

                        # Extract and store the LUT if available
                        if tar_offset_lut is not None and metadata["BitDepth"] == 16:
                            # Reset file position to start
                            data_bytes.seek(0)

                            # Skip to LUT position
                            data_bytes.seek(tar_offset_lut + TAR_HEADER_SIZE)

                            # Extract LUT data
                            lut_size = 2**16
                            lut_data = torch.asarray(data_bytes.read(lut_size * 4), dtype=torch.float32, copy=True)
                            # lut_data = np.frombuffer(data_bytes.read(lut_size * 4), dtype=np.float32).copy()

                            # Convert LUT data to Kelvin because Thermocontainer stores LUT in Kelvin
                            container.update_dataset("/MetaData/LookUpTable", lut_data + 273.15)

                    # Get frame info
                    frame_info = sequence.find("FrameInfo")
                    if frame_info is None:
                        raise ValueError(f"Sequence {id} seems corrupted! No FrameInfo node found.")

                    # Find all frames in FrameInfo node
                    frames = frame_info.findall("Frame")
                    if not frames:
                        raise ValueError(f"Sequence {id} seems corrupted! No Frame nodes found in FrameInfo.")

                    # Get DataType and BitDepth to determine how to process the data
                    data_type = int(metadata.get("DataType", -1))
                    bit_depth = int(metadata.get("BitDepth", -1))
                    if bit_depth not in (16, 32, 64):
                        raise ValueError(f"Unsupported BitDepth: {bit_depth}. Supported values are 16, 32, or 64.")

                    # Map bit depth to corresponding torch data type
                    tdata_type = {16: torch.uint16, 32: torch.float32, 64: torch.float64}

                    # Each frame is stored in a separate file. Because of the block size of the tar file,
                    # we need to skip the header size of the tar file (512 bytes) and the size of the frame header
                    # The frame header size is not explicitly given, so we need to calculate it based on the frame size
                    # Get the size of the first frame to be able to dynamically calculate the frame header size
                    first_frame_idx = frames[0].findtext("FrameIndex", default=None)
                    try:
                        file_size = tar_file.getmember(f"sequence{id}/f{first_frame_idx}.bin").size
                    except KeyError as e:
                        msg = f"Frames in Sequence {id} seem corrupted! Frame file f{first_frame_idx}.bin not found."
                        raise ValueError(msg) from e

                    # Calculate bytes per pixel and frame size
                    bytes_per_pixel = bit_depth // 8
                    frame_size_bytes = width * height * bytes_per_pixel

                    # Dynamically calculate the frame header size
                    header_size = file_size - frame_size_bytes

                    # Pre-allocate arrays for better performance
                    num_frames = len(frames)
                    domain_values = torch.zeros(num_frames, dtype=torch.float32)
                    frame_data = torch.zeros((height, width, num_frames), dtype=tdata_type[bit_depth])

                    for i, frame in enumerate(frames):
                        # Extract frame attributes
                        offset = int(frame.attrib.get("TarFileHeaderDataOffset", -1))

                        # Handle different data types
                        match data_type:
                            case 0:  # Shearography Image
                                raise NotImplementedError("Shearography Image data type is not implemented yet.")
                            case 2:  # Intensisty Image
                                raise NotImplementedError("Intensity Image data type is not implemented yet.")
                            case 5:  # Temperature Image
                                raise NotImplementedError("Temperature Image data type is not implemented yet.")
                            case 13:  # Complex Image
                                # Read domain values from the data
                                domain_str = frame.findtext("FourierFrequency", default=None)
                                domain_unit = Units.hertz
                                if domain_str:
                                    domain_values[i] = float(domain_str.strip().split("Hz")[0])

                                # Read frame data
                                data_bytes.seek(offset + TAR_HEADER_SIZE + header_size)
                                buffer = bytearray(data_bytes.read(frame_size_bytes))
                                frame_data[:, :, i] = torch.frombuffer(buffer, dtype=tdata_type[bit_depth]).reshape(
                                    height, width
                                )

                            case _:
                                msg = (
                                    f"Unsupported DataType: {data_type}. "
                                    "Supported values are 0 (Shearography), 2 (Intensity), "
                                    "5 (Temperature), or 13 (Complex)."
                                )
                                raise ValueError(msg)

                    # Update container
                    container.update_dataset("/MetaData/DomainValues", domain_values)
                    container.update_unit("/MetaData/DomainValues", domain_unit)
                    container.update_dataset("/Data/Tdata", frame_data)

                # # Create a simple excitation signal based on pulse length and domain values
                # # TODO: This is a simplified approach - actual signal might need more processing
                # excitation_signal = np.zeros_like(domain_values)
                # if data_type == 2:  # Only for time domain
                #     # Set signal to 1 for the duration of the pulse
                #     excitation_signal[domain_values <= pulse_length] = 1

                # container.update_dataset("/MetaData/ExcitationSignal", excitation_signal)

                # Add metadata to the container
                if metadata:
                    container.add_attributes("/MetaData", **metadata)

                # Return the container
                return container

        except Exception as e:
            raise ValueError(f"Error parsing Edevis file: {str(e)}") from e


def extract_metadata_from_xml(xml_root: Element, target_fields: Sequence[str] | None) -> dict:
    """Extracts the target fields from the XML root element.

    Will iterate through all the children of the XML root element and extract the text
    for the specified target fields. Fields that are not found will be skipped without error.

    Args:
        xml_root (Element): The root element of the XML document.
        target_fields (Sequence[str] | None): A sequence of target field names to extract.
            If None, all fields will be extracted.

    Returns:
        dict[str, str]: A dictionary containing the extracted field names and their corresponding text values.
    """
    target = set(target_fields) if target_fields else set()  # Convert to set for faster lookup
    metadata = {}

    # If target is not that long, directly calling find is more efficient
    if 0 < len(target) < 10:
        for field in target:
            element = xml_root.find(field)
            if element is not None and element.text:
                metadata[field] = element.text.strip()

    # For longer target lists, or when target is empty, iterate through all children
    else:
        for field in xml_root:
            if (field.tag in target or len(target) == 0) and field.text:
                metadata[field.tag] = field.text.strip()

            # Stop iteration if all target fields are found
            if len(metadata) == len(target):
                break

    return metadata
