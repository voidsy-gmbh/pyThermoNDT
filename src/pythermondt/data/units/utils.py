from ._unit import Unit
from .units import Units


def generate_label(unit: Unit) -> str:
    """Generates a label from a UnitInfo object.

    Args:
        unit (UnitInfo): The UnitInfo object to generate the label from.

    Returns:
        str: The generated label.
    """
    # Check for special units
    if unit == Units.dimensionless:
        return "dimensionless"
    if unit == Units.arbitrary:
        return "arbitrary"
    if unit == Units.undefined:
        return ""

    return f"{unit['quantity']} in {unit['symbol']}"
