from tqdm.auto import tqdm


class TqdmCallback(tqdm):
    """Progress bar with callback support.

    This allows to display a progress bar for long running operations that provide a callback function.
    """

    def __init__(self, total: int, desc: str, delay: float = 2.0, **kwargs):
        """Initialize with automatic delay-based display.

        Args:
            total (int): Total bytes expected
            desc (str): Progress description
            delay (float, optional): Show progress only if operation takes > delay seconds
            **kwargs: Additional tqdm arguments to be passed to tqdm
        """
        default_kwargs = {
            "unit": "B",
            "disable": True if total < 20 * 1024 * 1024 else False,  # Disable progress for files smaller than 20MB
            "unit_scale": True,
            "unit_divisor": 1024,
            "delay": delay,
            "leave": False,
        }
        default_kwargs.update(kwargs)
        super().__init__(total=total, desc=desc, **default_kwargs)

    def callback(self, bytes_amount: int) -> None:
        """Callback function for boto3 operations."""
        self.update(bytes_amount)
