import tarfile
from xml.dom.minidom import parseString

import numpy as np

from ...data import DataContainer, ThermoContainer
from ...data.units import Units
from ...utils import IOPathWrapper
from .base_parser import BaseParser


class EdevisParser(BaseParser):
    supported_extensions = ("di", "ITvisPulse", "OTvis")

    @staticmethod
    def parse(data: IOPathWrapper) -> DataContainer:
        """Parses the data from the given IOPathWrapper object into a DataContainer object.

        The IOPathWrapper object must contain a .di, .ITvisPulse or OTvis file with Measurement data from Edevis.

        Parameters:
            data (IOPathWrapper): IOPathWrapper object containing the data to be parsed.

        Returns:
            DataContainer: The parsed data as a DataContainer object.

        Raises:
            ValueError: If the given IOPathWrapper object is corupted
        """
        # Check if the BytesIO object is empty
        if data.file_obj.getbuffer().nbytes == 0:
            raise ValueError("The given BytesIO object is empty.")

        # Create an empty ThermoContainer - with predefined structure
        container = ThermoContainer()

        # Add source as an attribute
        container.add_attributes(path="/MetaData", Source="Edevis")

        data_bytes = data.file_obj

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
                xml_data = parseString(xml_content)

                # Process XML data to extract metadata
                # Get the sequence node
                subnode_list = xml_data.getElementsByTagName("Sequence")
                if not subnode_list:
                    raise ValueError("No Sequence node found in the file.")

                node_seq = subnode_list[0]  # Use sequence 0

                # Get the sequence info node
                subnode_list = node_seq.getElementsByTagName("SequenceInfo")
                if not subnode_list:
                    raise ValueError("No SequenceInfo node found in the file.")

                node_seq_info = subnode_list[0]

                # Extract metadata
                # Frame count
                frame_count = int(node_seq_info.getElementsByTagName("FrameCount")[0].firstChild.data)

                # Window dimensions
                window_str = node_seq_info.getElementsByTagName("Window")[0].firstChild.data
                width = int(window_str.split(",")[2])
                height = int(window_str.split(",")[3])

                # Frame rate
                frame_rate_str = node_seq_info.getElementsByTagName("FrameRate")[0].firstChild.data
                frame_rate = int(frame_rate_str.split("H")[0])

                # Data type (e.g., dft: 13, raw: 2)
                data_type = int(node_seq_info.getElementsByTagName("DataType")[0].firstChild.data)

                # Bit depth
                bit_depth = int(node_seq_info.getElementsByTagName("BitDepth")[0].firstChild.data)

                # Excitation amplitude
                amplitude_str = node_seq_info.getElementsByTagName("ExcitationAmplitude")[0].firstChild.data
                amplitude = float(amplitude_str.split(" %")[0])

                # Excitation pulse length
                pulse_str = node_seq_info.getElementsByTagName("ExcitationPulseLength")[0].firstChild.data
                pulse_length = float(pulse_str.split("s")[0])

                # Store metadata in container
                container.add_attributes(
                    "/MetaData",
                    FrameCount=frame_count,
                    Width=width,
                    Height=height,
                    FrameRate=frame_rate,
                    DataType=data_type,
                    BitDepth=bit_depth,
                    ExcitationAmplitude=amplitude,
                    ExcitationPulseLength=pulse_length,
                )

                # Get LUT offset if available
                tar_offset_lut = None
                if "TarFileHeaderCalibrationOffset" in node_seq_info.attributes:
                    tar_offset_lut = int(node_seq_info.attributes["TarFileHeaderCalibrationOffset"].value)

                    # Extract and store the LUT if available
                    if tar_offset_lut is not None:
                        # Reset file position to start
                        data_bytes.seek(0)
                        # Skip to LUT position
                        data_bytes.seek(tar_offset_lut)
                        # TODO: Implement LUT extraction and storage
                        # This would depend on how the LUT is stored in the file
                        # For now, just store a placeholder
                        container.update_dataset("/MetaData/LookUpTable", np.ones(1))

                # Get frame info
                subnode_list = node_seq.getElementsByTagName("FrameInfo")
                if not subnode_list:
                    raise ValueError("No FrameInfo node found in the file.")

                node_frame_info = subnode_list[0]
                frame_nodes = node_frame_info.getElementsByTagName("Frame")

                # Initialize arrays for frame offsets and domain values
                frame_offsets = np.zeros(len(frame_nodes), dtype=np.int64)
                domain_values = np.zeros(len(frame_nodes), dtype=float)

                # Extract frame offsets and domain values
                for j, frame_node in enumerate(frame_nodes):
                    frame_offsets[j] = int(frame_node.attributes["TarFileHeaderDataOffset"].value)

                    if data_type == 13:  # Fourier (frequency domain)
                        freq_str = frame_node.getElementsByTagName("FourierFrequency")[0].firstChild.data
                        domain_values[j] = float(freq_str.split("Hz")[0])
                        # Set unit to hertz for frequency domain
                        container.update_unit("/MetaData/DomainValues", Units.hertz)
                    elif data_type == 2:  # Raw (time domain)
                        time_str = frame_node.getElementsByTagName("FrameTime")[0].firstChild.data
                        domain_values[j] = float(time_str.split("s")[0])
                        # Set unit to second for time domain
                        container.update_unit("/MetaData/DomainValues", Units.second)
                    else:
                        raise ValueError(f"Unsupported data type: {data_type}")

                # Store domain values
                container.update_dataset("/MetaData/DomainValues", domain_values)

                # Extract and store frame data
                # Initialize temperature data array
                tdata = np.zeros((height, width, len(frame_nodes)), dtype=np.float32)

                # Reset file position
                data_bytes.seek(0)

                # Read each frame
                for j, offset in enumerate(frame_offsets):
                    data_bytes.seek(offset)

                    # Read frame data based on bit depth
                    if bit_depth == 16:
                        # Read raw data for 16-bit frames
                        frame_data = np.frombuffer(
                            data_bytes.read(width * height * 2),  # 2 bytes per pixel
                            dtype=np.uint16,
                        ).reshape(height, width)
                    elif bit_depth == 32:
                        # Read raw data for 32-bit frames
                        frame_data = np.frombuffer(
                            data_bytes.read(width * height * 4),  # 4 bytes per pixel
                            dtype=np.float32,
                        ).reshape(height, width)
                    elif bit_depth == 64:
                        # Read raw data for 64-bit frames
                        frame_data = np.frombuffer(
                            data_bytes.read(width * height * 8),  # 8 bytes per pixel
                            dtype=np.float64,
                        ).reshape(height, width)
                    else:
                        raise ValueError(f"Unsupported bit depth: {bit_depth}")

                    # Store frame data
                    tdata[:, :, j] = frame_data

                # Store temperature data
                container.update_dataset("/Data/Tdata", tdata)

                # Create a simple excitation signal based on pulse length and domain values
                # This is a simplified approach - actual signal might need more processing
                excitation_signal = np.zeros_like(domain_values)
                if data_type == 2:  # Only for time domain
                    # Set signal to 1 for the duration of the pulse
                    excitation_signal[domain_values <= pulse_length] = 1

                container.update_dataset("/MetaData/ExcitationSignal", excitation_signal)

                # Return the container
                return container

        except Exception as e:
            raise ValueError(f"Error parsing Edevis file: {str(e)}") from e
