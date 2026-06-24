from collections.abc import Sequence
from typing import Literal, get_args

import torch

from ..data import DataContainer
from ..data.units import arbitrary, kelvin
from .base import ThermoTransform
from .utils import _get_optional_dataset


class ApplyLUT(ThermoTransform):
    """Applies the LookUpTable of the container to the Temperature data in the container.

    This is done by indexing the LookUpTable (Float64) with the Temperature data (Uint16).
    As a result Tdata gets converted from uint16 to float64.
    """

    def __init__(self, target_dtype: torch.dtype | None = None):
        """Applies the LookUpTable of the container to the Temperature data in the container.

        This is done by indexing the LookUpTable (Float64) with the Temperature data (Uint16).
        As a result Tdata gets converted from uint16 to float64.

        Args:
            target_dtype (torch.dtype, optional): The target data type to cast the LookUpTable (LUT) to before indexing.
                If None, the LUT's original dtype will be used. The dtype of the LUT determines the final output dtype
                after indexing the temperature data. Defaults to None.
        """
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        lut = container.get_dataset("/MetaData/LookUpTable")
        lut = lut.to(self.target_dtype) if self.target_dtype is not None else lut
        tdata = container.get_dataset("/Data/Tdata")

        # Check if LUT is available
        if lut is None or tdata is None:
            raise ValueError("LookUpTable or Tdata is not available in the container.")

        # Check if LUT has already been applied
        if container.get_unit("/Data/Tdata") != arbitrary and torch.is_floating_point(tdata):
            raise ValueError("LookUpTable has already been applied to the Temperature data.")

        # Sanity check if tdata is valid
        if tdata.dtype != torch.uint16 and tdata.negative().any():
            raise ValueError("Invalid values in Tdata. Applying LookUpTable is not supported.")

        # Catch invalid indices. Negative indices are not allowed because negative values in Tdata create
        # ambiguous results when applying the LUT. If Tdata is unsigned this will not be a problem.
        try:
            tdata = torch.take(lut, tdata.flatten().long()).reshape(tdata.shape)
        except IndexError as e:
            raise IndexError("Index out of bounds. Tdata contains invalid indices not available from LUT") from e

        # Update the container and return it
        container.update_dataset("/Data/Tdata", tdata)
        container.update_unit("/Data/Tdata", kelvin)
        return container


class CastTo(ThermoTransform):
    """Cast one or more datasets in the container to a specified torch dtype.

    ``datasets`` may be a single path or a sequence of paths. A single dtype is applied to all datasets;
    alternatively, a sequence of dtypes (one per dataset) allows casting different datasets to different dtypes.
    """

    def __init__(self, datasets: str | Sequence[str], dtype: torch.dtype | Sequence[torch.dtype]):
        """Cast one or more datasets in the container to a specified torch dtype.

        A single dtype is applied to all datasets; alternatively, a sequence of dtypes (one per dataset) allows
        casting different datasets to different dtypes.

        Args:
            datasets (str | Sequence[str]): Dataset path or sequence of paths to cast. Must be non-empty.
            dtype (torch.dtype | Sequence[torch.dtype]): Target torch dtype, either a single dtype applied to
                all datasets or a sequence with one dtype per dataset.

        Raises:
            TypeError: If ``dtype`` is not a ``torch.dtype`` or a sequence of ``torch.dtype``.
            ValueError: If ``datasets`` is empty or the number of dtypes does not match the number of datasets.
        """
        super().__init__()

        # Normalize datasets to a list (str is a Sequence, so check it first to avoid splitting the path)
        datasets = [datasets] if isinstance(datasets, str) else list(datasets)
        if not datasets:
            raise ValueError("datasets must be a non-empty sequence of dataset paths.")

        # Normalize dtype to a per-dataset list
        if isinstance(dtype, torch.dtype):
            dtypes = [dtype] * len(datasets)
        elif isinstance(dtype, Sequence):
            dtypes = list(dtype)
        else:
            raise TypeError(f"dtype must be a torch.dtype or a sequence of torch.dtype, got {type(dtype).__name__}.")

        if len(dtypes) != len(datasets):
            raise ValueError(f"Number of dtypes ({len(dtypes)}) must match number of datasets ({len(datasets)}).")
        if not all(isinstance(d, torch.dtype) for d in dtypes):
            raise TypeError("All dtypes must be torch.dtype instances.")

        self.datasets = datasets
        self.dtype = dtypes

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the datasets
        tensors = container.get_datasets(*self.datasets)

        # Cast each tensor to its target dtype
        updates = ((p, t.to(dt)) for p, t, dt in zip(self.datasets, tensors, self.dtype, strict=True))

        # Update the container and return it
        container.update_datasets(*updates)
        return container


class SubtractFrame(ThermoTransform):
    """Subtracts 1 frame from all other frames in the Temperature data (Tdata) of the container."""

    def __init__(self, frame: int = 0):
        """Subtracts 1 frame from all other frames in the Temperature data (Tdata) of the container.

        Args:
            frame (int): Frame number that should be subtracted from the Temperature data.
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

        # Subtract the frame from Tdata
        tdata = tdata - tdata[:, :, self.frame].unsqueeze(2)

        # Update the container and return it
        container.update_dataset("/Data/Tdata", tdata)
        return container


class RemoveFlash(ThermoTransform):
    """Automatically detect the flash and remove all the frames before it."""

    def __init__(
        self, method: Literal["excitation_signal", "max_temp", "mean_temp_drop"] = "excitation_signal", offset: int = 0
    ):
        """Automatically detect the flash and remove all the frames before it.

        2 methods are available:
        - "excitation_signal": Detect the flash by finding the frame where the excitation signal goes from 1 back to 0.
        - "max_temp": Detect the flash by finding the frame with the maximum temperature value in it.
            May not work if the flash is not the hottest frame.
        - "mean_temp_drop": Detect the flash by finding the largest temperature drop in the mean temperature over all
            frames. This is the most reliable method if excitation signal is not available.

        Args:
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
                for i in range(1, len(excitation_signal)):  # Start at 1
                    if excitation_signal[i - 1] == 1 and excitation_signal[i] == 0:
                        flash_end_idx = i + self.offset
                        break

                if flash_end_idx is None:
                    raise ValueError("Flash could not be detected in the excitation signal.")

            case "max_temp":
                # Find frame with maximum temperature value (flash end)
                flash_end_idx = int(tdata.argmax(dim=2).max().item()) + self.offset

            case "mean_temp_drop":
                # Find largest temperature drop (flash end) ==> minimum of the temperature difference
                mean_temps = tdata.mean(dim=(0, 1))
                diffs = torch.diff(mean_temps)
                flash_end_idx = int(diffs.argmin().item()) + self.offset  # Get the frame with biggest temperature drop

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

        # Fix time shift in domain values by subtracting the first time step
        domain_values = domain_values - domain_values[0]

        # Update the container and return it
        # pylint: disable=duplicate-code
        container.update_datasets(
            ("/Data/Tdata", tdata),
            ("/MetaData/DomainValues", domain_values),
            ("/MetaData/ExcitationSignal", excitation_signal),
        )
        # pylint: enable=duplicate-code
        return container


class CropFrames(ThermoTransform):
    """Crops the frames of the Temperature data (Tdata) of the container."""

    def __init__(self, height: int, width: int, method: Literal["C", "TL", "TR", "BL", "BR"] = "C"):
        """Crops the frames and the mask in the container to the specified height and width.

        Args:
            height (int): Height of the cropped frames.
            width (int): Width of the cropped frames.
            method (optional, Literal["C", "TL", "TR", "BL", "BR"]): Cropping strategy. Default is "C" (center).
                - "C": Center cropping
                - "TL": Top left cropping
                - "TR": Top right cropping
                - "BL": Bottom left cropping
                - "BR": Bottom right cropping
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
        # Extract the data (DefectMask is optional)
        tdata = container.get_dataset("/Data/Tdata")
        has_mask, mask = _get_optional_dataset(container, "/GroundTruth/DefectMask")

        if self.height > tdata.shape[0]:
            raise ValueError(
                f"Invalid cropping height: Requested height ({self.height}) is greater than "
                f"the data height ({tdata.shape[0]}). Ensure the height does not exceed the data dimensions."
            )

        if self.width > tdata.shape[1]:
            raise ValueError(
                f"Invalid cropping width: Requested width ({self.width}) is greater than "
                f"the data width ({tdata.shape[1]}). Ensure the width does not exceed the data dimensions."
            )

        match self.strategy:
            case "C":
                # Center cropping
                height_diff = tdata.shape[0] - self.height
                width_diff = tdata.shape[1] - self.width

                if height_diff % 2 == 0:
                    top = height_diff // 2
                    bottom = top + self.height
                else:
                    raise ValueError(
                        f"Invalid height for center cropping: "
                        f"Original height = {tdata.shape[0]}, Target height = {self.height}. "
                        f"Difference ({height_diff}) must be even for proper centering."
                    )

                if width_diff % 2 == 0:
                    left = width_diff // 2
                    right = left + self.width
                else:
                    raise ValueError(
                        f"Invalid width for center cropping: "
                        f"Original width = {tdata.shape[1]}, Target width = {self.width}. "
                        f"Difference ({width_diff}) must be even for proper centering."
                    )
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

        # Build update tuples (only include mask if present and non-empty)
        updates: list[tuple[str, torch.Tensor]] = [("/Data/Tdata", tdata)]
        if has_mask:
            mask = mask[top:bottom, left:right]  # type: ignore[index]
            updates.append(("/GroundTruth/DefectMask", mask))

        container.update_datasets(*updates)
        return container
