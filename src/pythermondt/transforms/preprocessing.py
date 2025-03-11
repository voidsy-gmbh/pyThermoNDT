from typing import Literal, get_args

import torch

from ..data import DataContainer
from ..data.units import Units
from .utils import ThermoTransform


class ApplyLUT(ThermoTransform):
    """Applies the LookUpTable of the container to the Temperature data in the container.

    This is done by indexing the LookUpTable (Float64) with the Temperature data (Uint16).
    As a result Tdata gets converted from uint16 to float64.
    """

    def __init__(self):
        """Applies the LookUpTable of the container to the Temperature data in the container.

        This is done by indexing the LookUpTable (Float64) with the Temperature data (Uint16).
        As a result Tdata gets converted from uint16 to float64.
        """
        super().__init__()

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        lut = container.get_dataset("/MetaData/LookUpTable")
        tdata = container.get_dataset("/Data/Tdata")

        # Check if LUT is available
        if lut is None or tdata is None:
            raise ValueError("LookUpTable or Tdata is not available in the container.")

        # Check if LUT has already been applied
        if container.get_unit("/Data/Tdata") != Units.arbitrary and torch.is_floating_point(tdata):
            raise ValueError("LookUpTable has already been applied to the Temperature data.")

        # Check if the data is of the correct type
        if not isinstance(lut, torch.Tensor):
            raise ValueError("LookUpTable is not a torch.Tensor")
        if not isinstance(tdata, torch.Tensor):
            raise ValueError("Tdata is not a torch.Tensor")

        # Convert Tdata from uin16 to int32, because indexing in pytorch does not work with unsigned integers
        tdata = tdata.to(torch.int32)

        # Check for index out of bounds
        if tdata.min() < 0 or tdata.max() >= lut.shape[0]:
            raise IndexError("Index out of bounds. Tdata contains indices that are not available in the LookUpTable.")

        # Apply the LUT to Tdata
        tdata = lut[tdata]

        # Update the container and return it
        container.update_dataset("/Data/Tdata", tdata)
        container.update_unit("/Data/Tdata", Units.kelvin)
        return container


class SubstractFrame(ThermoTransform):
    """Substracts 1 frame from all other frames in the Temperature data (Tdata) of the container."""

    def __init__(self, frame: int = 0):
        """Substracts 1 frame from all other frames in the Temperature data (Tdata) of the container.

        Parameters:
            frame (int): Frame number that should be substracted from the Temperature data.
                Default is the initial frame (frame 0).
        """
        super().__init__()

        # Check if frame is a positive integer
        if frame < 0 or not isinstance(frame, int):
            raise ValueError("Frame must be a positive integer.")

        self.frame = frame

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata = container.get_dataset("/Data/Tdata")

        # Check if data is available
        if tdata is None:
            raise ValueError("Tdata is not available in the container.")

        # Check if the data is of the correct type
        if not isinstance(tdata, torch.Tensor):
            raise ValueError("Tdata is not a torch.Tensor")

        # Check for index out of bounds
        if self.frame >= tdata.shape[2]:
            raise IndexError("Index out of bounds. Frame number is bigger than the number of frames in the data.")

        # Substract the frame from Tdata
        tdata = tdata - tdata[:, :, self.frame].unsqueeze(2)

        # Update the container and return it
        container.update_dataset("/Data/Tdata", tdata)
        return container


class RemoveFlash(ThermoTransform):
    """Automatically detect the flash and remove all the frames before it."""

    def __init__(self, method: Literal["excitation_signal", "max_temp"] = "excitation_signal", offset: int = 0):
        """Automatically detect the flash and remove all the frames before it.

        2 methods are available:
        - "excitation_signal": Detect the flash by finding the frame where the excitation signal goes from 1 back to 0.
        - "max_temp": Detect the flash by finding the frame with the maximum temperature value in it.
            May not work if the flash is not the hottest frame.

        Parameters:
            method (Literal["excitation_signal", "max_temp"]): Method to detect the flash.
                Default is "excitation_signal".
            offset (int): Offset in frames to add to the detected flash end. Default is 0.
        """
        super().__init__()
        self.offset = offset
        self.method = method

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract tdata and domain values
        tdata, excitation_signal, domain_values = container.get_datasets(
            "/Data/Tdata", "/MetaData/ExcitationSignal", "/MetaData/DomainValues"
        )

        # Detect the flash frame based on the method
        match self.method:
            case "excitation_signal":
                # Find frame where excitation signal goes from 1 back to 0 (flash end)
                flash_end_idx = None
                for i in range(len(excitation_signal)):
                    if excitation_signal[i - 1] == 1 and excitation_signal[i] == 0:
                        flash_end_idx = i + self.offset
                        break

                if flash_end_idx is None:
                    raise ValueError("Flash could not be detected in the excitation signal.")

            case "max_temp":
                # Find frame with maximum temperature value (flash end)
                flash_end_idx = tdata.argmax(dim=2).max().item() + self.offset

            case _:
                raise ValueError(f"Invalid method. Choose between {get_args(self.__init__.__annotations__['method'])}.")

        # Check if the flash end is valid
        if flash_end_idx < 0 or flash_end_idx >= len(domain_values):
            raise IndexError(
                f"Flash end index {flash_end_idx} is out of bounds. Valid range is {[0, len(domain_values) - 1]}."
            )

        # Keep only the frames after the flash
        tdata = tdata[..., flash_end_idx:]
        domain_values = domain_values[flash_end_idx:]
        excitation_signal = excitation_signal[flash_end_idx:]

        # Fix time shift in domain values by substracting the first time step
        domain_values = domain_values - domain_values[0]

        # Update the container and return it
        container.update_datasets(
            ("/Data/Tdata", tdata),
            ("/MetaData/DomainValues", domain_values),
            ("/MetaData/ExcitationSignal", excitation_signal),
        )
        return container


class CropFrames(ThermoTransform):
    """Crops the frames of the Temperature data (Tdata) of the container."""

    def __init__(self, height: int, width: int, method: Literal["C", "TL", "TR", "BL", "BR"]):
        """Crops the frames of the Temperature data (Tdata) of the container.

        Parameters:
            height (int): Height of the cropped frames.
            width (int): Width of the cropped frames.
            method (Literal["C", "TL", "TR", "BL", "BR"]): Strategy to crop the frames.
                - "C": Crop the frames around the center.
                - "TL": Crop the frames around the top left corner.
                - "TR": Crop the frames around the top right corner.
                - "BL": Crop the frames around the bottom left corner.
                - "BR": Crop the frames around the bottom right corner.
        """
        super().__init__()

        # Check if height and width are positive integers
        if height <= 0 or not isinstance(height, int):
            raise ValueError("Height must be a positive integer.")

        if width <= 0 or not isinstance(width, int):
            raise ValueError("Width must be a positive integer.")

        # Check if strategy is valid
        if method not in ["C", "TL", "TR", "BL", "BR"]:
            raise ValueError("Invalid method. Choose between 'C', 'TL', 'TR', 'BL', 'BR'.")

        self.height = height
        self.width = width
        self.strategy = method

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata, mask = container.get_datasets("/Data/Tdata", "/GroundTruth/DefectMask")

        if self.height > tdata.shape[0]:
            raise ValueError("Height is bigger than the data.")

        if self.width > tdata.shape[1]:
            raise ValueError("Width is bigger than the data.")

        match self.strategy:
            case "C":
                # Center cropping
                print((tdata.shape[0] - self.height) % 2 == 0)
                if (tdata.shape[0] - self.height) % 2 == 0:
                    top = (tdata.shape[0] - self.height) // 2
                    bottom = top + self.height
                else:
                    raise ValueError("Height is not even. Center cropping is not possible.")
                if (tdata.shape[1] - self.width) % 2 == 0:
                    left = (tdata.shape[1] - self.width) // 2
                    right = left + self.width
                else:
                    raise ValueError("Width is not even. Center cropping is not possible.")
            case "TL":
                # Top left cropping
                top = 0
                bottom = self.height
                left = 0
                right = self.width

            case "TR":
                # Top right cropping
                top = 0
                bottom = self.height
                left = tdata.shape[1] - self.width
                right = tdata.shape[1]

            case "BL":
                # Bottom left cropping
                top = tdata.shape[0] - self.height
                bottom = tdata.shape[0]
                left = 0
                right = self.width

            case "BR":
                # Bottom right cropping
                top = tdata.shape[0] - self.height
                bottom = tdata.shape[0]
                left = tdata.shape[1] - self.width
                right = tdata.shape[1]

            case _:
                raise ValueError("Invalid strategy.")

        # Crop the data
        tdata = tdata[top:bottom, left:right]
        mask = mask[top:bottom, left:right]

        # Update the container and return it
        container.update_datasets(("/Data/Tdata", tdata), ("/GroundTruth/DefectMask", mask))
        return container
