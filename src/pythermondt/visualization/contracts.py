from __future__ import annotations

from typing import Any, TypeAlias, TypedDict

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, Any]

API_PREFIX = "/api/v1"
DEFAULT_VECTOR_COUNT = 20000
MAX_VECTOR_COUNT = 500000
MAX_BINARY_RESPONSE_BYTES = 128 * 1024 * 1024
DEFAULT_MATRIX_ROWS = 20
DEFAULT_MATRIX_COLS = 20
MAX_MATRIX_ROWS = 200
MAX_MATRIX_COLS = 200

BINARY_CONTENT_TYPE = "application/octet-stream"
BINARY_DTYPE = "float32"

HEADER_DTYPE = "X-PTNDT-Dtype"
HEADER_SHAPE = "X-PTNDT-Shape"
HEADER_NDIM = "X-PTNDT-Ndim"
HEADER_COUNT = "X-PTNDT-Count"
HEADER_ORDER = "X-PTNDT-Order"
HEADER_FRAME_AXIS = "X-PTNDT-Frame-Axis"
HEADER_FRAME_INDEX = "X-PTNDT-Frame-Index"
ARRAY_ORDER = "C"


class ApiError(TypedDict):
    error: str
