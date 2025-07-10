import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings for pyThermoNDT."""

    # Configuration parameters
    download_dir: str = Field(default="./", description="Base directory where pythermondt will download files to.")
    num_workers: int = Field(
        default=os.cpu_count() or 1, description="Default number of workers used for parallel operations."
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


# Global settings instance
settings = Settings()
