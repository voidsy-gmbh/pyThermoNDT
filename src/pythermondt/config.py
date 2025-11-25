import logging
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def configure_logging(level: str | None = None):
    """Configure logging for pythermondt.

    Simple convenience function for quick debugging. Advanced users should
    configure Python's logging directly.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses settings.log_level.

    Example:
        >>> import pythermondt
        >>> pythermondt.configure_logging("DEBUG")  # See all logs
        >>> pythermondt.configure_logging()  # Use settings.log_level
    """
    # Add a basic configuration but keep loglevel at default (WARNING)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing config
    )

    # Capture warnings from the warnings module aswell
    logging.captureWarnings(True)

    # Configure log level specifically for the pythermondt logger ==> avoids spamming from other libraries
    level = level or settings.log_level
    logger = logging.getLogger("pythermondt")
    logger.setLevel(getattr(logging, level.upper()))


class Settings(BaseSettings):
    """Global settings for PyThermoNDT."""

    # Configuration parameters
    download_dir: str = Field(default="./", description="Base directory where PyThermoNDT will download files to.")
    num_workers: int = Field(
        default=os.cpu_count() or 1, description="Default number of workers used for parallel operations."
    )
    log_level: str = Field(
        default="WARNING", description="Default log level for pythermondt (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    model_config = SettingsConfigDict(
        env_prefix="PYTHERMONDT_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @field_validator("download_dir", mode="after")
    @classmethod
    def ensure_exists(cls, v: str) -> str:
        # Use os.path for validation/normalization
        expanded = os.path.expanduser(v)
        absolute = os.path.abspath(expanded)

        # Create if missing
        os.makedirs(absolute, exist_ok=True)

        return absolute

    @field_validator("num_workers", mode="before")
    @classmethod
    def validate_num_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_workers must be at least 1")
        return v

    @field_validator("log_level", mode="after")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {v}")
        return v_upper


# Global settings instance
settings = Settings()
