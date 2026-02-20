from __future__ import annotations

from typing import Any, TypeAlias, TypedDict

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, Any]

API_PREFIX = "/api/v1"
MAX_PREVIEW_LIMIT = 5000
DEFAULT_PREVIEW_LIMIT = 256


class ApiError(TypedDict):
    error: str
