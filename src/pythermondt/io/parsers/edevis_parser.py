import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from io import BytesIO
from xml.etree.ElementTree import Element

from ...data import DataContainer, ThermoContainer
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

                # # Extract metadata
                # # Excitation amplitude
                # amplitude_str = get_element_text(node_seq_info, "ExcitationAmplitude")
                # amplitude = float(amplitude_str.split(" %")[0])

                # # Excitation pulse length
                # pulse_str = get_element_text(node_seq_info, "ExcitationPulseLength")
                # pulse_length = float(pulse_str.split("s")[0])

                # # Store metadata in container
                # container.add_attributes(
                #     "/MetaData",
                #     FrameCount=frame_count,
                #     Width=width,
                #     Height=height,
                #     FrameRate=frame_rate,
                #     DataType=data_type,
                #     BitDepth=bit_depth,
                #     ExcitationAmplitude=amplitude,
                #     ExcitationPulseLength=pulse_length,
                # )

                # # Get LUT offset if available
                # tar_offset_lut = None
                # if "TarFileHeaderCalibrationOffset" in node_seq_info.attributes:
                #     tar_offset_lut = int(node_seq_info.attributes["TarFileHeaderCalibrationOffset"].value)

                #     # Extract and store the LUT if available
                #     if tar_offset_lut is not None and bit_depth == 16:
                #         # Reset file position to start
                #         data_bytes.seek(0)

                #         # Skip to LUT position
                #         data_bytes.seek(tar_offset_lut + TAR_HEADER_SIZE)

                #         # Extract LUT data
                #         lut_size = 2**16
                #         lut_data = np.frombuffer(data_bytes.read(lut_size * 4), dtype=np.float32).copy()

                #         # Convert LUT data to Kelvin because Thermocontainer stores LUT in Kelvin
                #         container.update_dataset("/MetaData/LookUpTable", lut_data + 273.15)

                # # Get frame info
                # subnode_list = node_seq.getElementsByTagName("FrameInfo")
                # if not subnode_list:
                #     raise ValueError("No FrameInfo node found in the file.")

                # node_frame_info = subnode_list[0]
                # frame_nodes = node_frame_info.getElementsByTagName("Frame")

                # # Initialize arrays for frame offsets and domain values
                # frame_offsets = np.zeros(len(frame_nodes), dtype=np.int64)
                # domain_values = np.zeros(len(frame_nodes), dtype=float)

                # # Extract frame offsets and domain values
                # for j, frame_node in enumerate(frame_nodes):
                #     frame_offsets[j] = int(frame_node.attributes["TarFileHeaderDataOffset"].value)

                #     if data_type == 13:  # Fourier (frequency domain)
                #         freq_str = get_element_text(frame_node, "FourierFrequency")
                #         domain_values[j] = float(freq_str.split("Hz")[0])
                #         # Set unit to hertz for frequency domain
                #         container.update_unit("/MetaData/DomainValues", Units.hertz)
                #     elif data_type == 2:  # Raw (time domain)
                #         time_str = get_element_text(frame_node, "FrameTime")
                #         domain_values[j] = float(time_str.split("s")[0])
                #         # Set unit to second for time domain
                #         container.update_unit("/MetaData/DomainValues", Units.second)
                #     else:
                #         raise ValueError(f"Unsupported data type: {data_type}")

                # # Store domain values
                # container.update_dataset("/MetaData/DomainValues", domain_values)

                # # Extract and store frame data
                # # Map bit depth to corresponding numpy data type
                # tdata_type = {16: np.uint16, 32: np.float32, 64: np.float64}

                # # Check if the bit depth is supported
                # if bit_depth not in tdata_type:
                #     raise ValueError(f"Unsupported bit depth: {bit_depth}")

                # # Calculate bytes per pixel and frame size
                # bytes_per_pixel = bit_depth // 8
                # frame_size_bytes = width * height * bytes_per_pixel

                # # Each frame is stored in a separate file. Because of the block size of the tar file,
                # # we need to skip the header size of the tar file (512 bytes) and the size of the frame header
                # # Get first frame file
                # frame_files = [m for m in tar_file.getmembers() if re.match(r"^sequence0/f\d+\.bin$", m.name)]
                # if not frame_files:
                #     raise ValueError("No frame files found in the tar archive.")

                # # Determine the total size one frame file
                # total_size = frame_files[0].size

                # # Calculate the frame header size
                # header_size = total_size - frame_size_bytes

                # # Initialize temperature data array with the appropriate type
                # tdata = np.zeros((height, width, len(frame_nodes)), dtype=tdata_type[bit_depth])

                # # Reset file position
                # data_bytes.seek(0)

                # # Read each frame
                # for j, offset in enumerate(frame_offsets):
                #     # Seek to frame position
                #     data_bytes.seek(offset.item() + TAR_HEADER_SIZE + header_size)

                #     # test_frame_offset(data_bytes, offset, frame_size_bytes, height, width, bit_depth)

                #     # Read the entire frame at once and reshape
                #     tdata[:, :, j] = np.frombuffer(
                #         data_bytes.read(frame_size_bytes), dtype=tdata_type[bit_depth]
                #     ).reshape(height, width)

                # # Store temperature data
                # container.update_dataset("/Data/Tdata", tdata)

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


def extract_metadata_from_xml(xml_root: Element, target_fields: Sequence[str] | None) -> dict[str, str]:
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
