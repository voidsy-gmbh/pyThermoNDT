import torch

from ..data import DataContainer
from .base import RandomThermoTransform


class RandomFlip(RandomThermoTransform):
    """Randomly flips all frames in the Temperature data (Tdata) of the container.

    The flipping occurs along the height (vertical flip) and/or the width (horizontal flip) of Tdata.
    The probabilities of flipping along the height and width directions can be specified separately.
    """

    def __init__(self, p_height: float = 0.5, p_width: float = 0.5):
        """Randomly flips all frames in the Temperature data (Tdata) of the container.

        The flipping occurs along the height (vertical flip) and/or the width (horizontal flip) of Tdata.
        The probabilities of flipping along height and width can be specified separately.

        Args:
            p_height (float, optional): Probability of flipping along the height (vertical flip), in the range [0, 1].
                Default is 0.5.
            p_width (float, optional): Probability of flipping along the width (horizontal flip), in the range [0, 1].
                Default is 0.5.
        """
        super().__init__()

        # Check if p_height and p_width are probabilities
        if not 0 <= p_height <= 1:
            raise ValueError("p_height must be in the range [0, 1]")

        if not 0 <= p_width <= 1:
            raise ValueError("p_width must be in the range [0, 1]")

        # Store the probabilities
        self.p_height = p_height
        self.p_width = p_width

    def forward(self, container: DataContainer) -> DataContainer:
        # Extract the data
        tdata, mask = container.get_datasets("/Data/Tdata", "/GroundTruth/DefectMask")

        # Flip the data along the height if the random number is less than p_height
        if torch.rand(1).item() < self.p_height:
            tdata = torch.flip(tdata, [1])
            mask = torch.flip(mask, [1])

        # Flip the data along the width if the random number is less than p_width
        if torch.rand(1).item() < self.p_width:
            tdata = torch.flip(tdata, [0])
            mask = torch.flip(mask, [0])

        # Update the container and return it
        container.update_datasets(("/Data/Tdata", tdata), ("/GroundTruth/DefectMask", mask))
        return container


class GaussianNoise(RandomThermoTransform):
    """Add Gaussian noise to the Temperature data in the container based on specified mean and standard deviation."""

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        """Initializes the GaussianNoise transformation with specified mean and standard deviation.

        Args:
            mean (float): Mean of the Gaussian noise. Default is 0.0.
            std (float): Standard deviation of the Gaussian noise. Default is 0.1.

        Raises:
            ValueError: If std is negative.
        """
        super().__init__()

        # Ensure std is positive
        if std < 0:
            raise ValueError("std must be non-negative")

        self.mean = mean
        self.std = std

    def forward(self, container: DataContainer) -> DataContainer:
        tdata = container.get_dataset("/Data/Tdata")
        noise = torch.normal(mean=self.mean, std=self.std, size=tdata.size(), device=tdata.device, dtype=tdata.dtype)
        container.update_dataset("/Data/Tdata", tdata + noise)
        return container


class AdaptiveGaussianNoise(RandomThermoTransform):
    """Add Gaussian noise with std uniformly sampled from a given range."""

    def __init__(self, mean: float = 0.0, std_range: tuple[float, float] = (0.0, 0.1)):
        """Initializes the AdaptiveGaussianNoise transformation with specified mean, std range, and distribution.

        Args:
            mean (float): Mean of the Gaussian noise. Default is 0.0.
            std_range (tuple[float, float]): Range (min, max) for standard deviation of the Gaussian noise.
                Default is (0.0, 0.1).

        Raises:
            ValueError: If std_range is invalid.
        """
        super().__init__()

        # Validate std_range
        if len(std_range) != 2:
            raise ValueError("std_range must be a tuple of two numbers")

        self.mean = mean
        self.std_range = std_range

    def forward(self, container: DataContainer) -> DataContainer:
        tdata = container.get_dataset("/Data/Tdata")
        std = torch.ones(1, device=tdata.device).uniform_(*self.std_range).item()
        print(std)
        noise = torch.normal(mean=self.mean, std=std, size=tdata.size(), device=tdata.device, dtype=tdata.dtype)
        container.update_dataset("/Data/Tdata", tdata + noise)
        return container
