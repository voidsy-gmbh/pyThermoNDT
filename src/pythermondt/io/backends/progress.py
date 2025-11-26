from tqdm.auto import tqdm

TQDM_DEFAULT_KWARGS = {
    "unit": "B",
    "unit_scale": True,
    "unit_divisor": 1024,
    "leave": False,
}


def get_tqdm_delay(file_size: int, delay: float = 4.0) -> float:
    """Get delay value based on file size."""
    return delay if file_size < 40 * 1024 * 1024 else 0


def get_tqdm_default_kwargs(file_size: int = 0, delay: float = 4.0) -> dict:
    """Get default tqdm kwargs."""
    return {**TQDM_DEFAULT_KWARGS, "total": file_size, "delay": get_tqdm_delay(file_size, delay)}


class TqdmCallback(tqdm):
    """Progress bar with callback support.

    This allows to display a progress bar for long running operations that provide a callback function.
    """

    def __init__(self, total: int, desc: str, delay: float = 4.0, **kwargs):
        """Initialize with automatic delay-based display.

        Args:
            total (int): Total bytes expected
            desc (str): Progress description
            delay (float, optional): Show progress only if operation takes > delay seconds. Only for files < 40MB
            **kwargs: Additional tqdm arguments to be passed to tqdm
        """
        merged_kwargs = {
            **TQDM_DEFAULT_KWARGS,
            "total": total,
            "desc": desc,
            "delay": get_tqdm_delay(total, delay),
            **kwargs,  # User overrides
        }
        super().__init__(**merged_kwargs)

    def callback(self, bytes_amount: int) -> None:
        """Callback function for boto3 operations."""
        self.update(bytes_amount)
