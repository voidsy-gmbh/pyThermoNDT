from pydantic_settings import BaseSettings, SettingsConfigDict

from .__pkginfo__ import __version__
from .data import ThermoContainer, ThermoDataset
from .io import HDF5Parser, SimulationParser
from .readers import LocalReader, S3Reader
from .transforms import augmentation, normalization, preprocessing, sampling, utils
from .writers import LocalWriter, S3Writer


class Settings(BaseSettings):
    """Global settings for pyThermoNDT."""

    download_dir: str = "."

    model_config = SettingsConfigDict(
        env_prefix="PYTHERMONDT_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


# Global settings instance
settings = Settings()

__all__ = [
    "__version__",
    "ThermoContainer",
    "ThermoDataset",
    "HDF5Parser",
    "SimulationParser",
    "LocalReader",
    "S3Reader",
    "augmentation",
    "normalization",
    "preprocessing",
    "sampling",
    "utils",
    "LocalWriter",
    "S3Writer",
    "settings",
]

# TODO: Implement logging instead of print statements according to this guide: https://docs.python.org/3/howto/logging.html
# TODO: Implement async data loading
# TODO: Implement multi threading for data loading / writing
# TODO: Add more tests
# TODO: Add glob patterns to S3Reader
