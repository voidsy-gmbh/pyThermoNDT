from typing import Any, TypedDict, TypeGuard


class Unit(TypedDict):
    """TypedDict for unit information."""

    name: str  # The name of the unit (e.g. kelvin, celsius, etc.)
    quantity: str  # The quantity the unit represents (e.g. temperature, time, etc.)
    symbol: str  # The symbol of the unit (e.g. K, °C, etc.)


def is_unit_info(obj: Any) -> TypeGuard[Unit]:
    """TypeGuard to check if an object is a valid UnitInfo.

    Args:
        obj (Any): The object to check.

    Returns:
        TypeGuard[UnitInfo]: True if the object is a valid UnitInfo, False otherwise.
    """
    return (
        isinstance(obj, dict)
        and "name" in obj
        and isinstance(obj["name"], str)
        and "quantity" in obj
        and isinstance(obj["quantity"], str)
        and "symbol" in obj
        and isinstance(obj["symbol"], str)
    )
