from dataclasses import dataclass, fields


@dataclass(slots=True, frozen=True)
class BackendClientOptions:
    """Shared helpers for backend client option dataclasses."""

    def as_kwargs(self) -> dict[str, object]:
        """Return only configured client kwargs."""
        return {field.name: value for field in fields(self) if (value := getattr(self, field.name)) is not None}


@dataclass(slots=True, frozen=True)
class AzureBlobClientOptions(BackendClientOptions):
    """Azure Blob Storage client tuning options."""

    max_single_put_size: int | None = None
    max_block_size: int | None = None
    connection_timeout: int | None = None
    read_timeout: int | None = None


@dataclass(slots=True, frozen=True)
class S3ClientOptions(BackendClientOptions):
    """S3 client tuning options."""

    connect_timeout: int | float | None = None
    read_timeout: int | float | None = None
    max_pool_connections: int | None = None
