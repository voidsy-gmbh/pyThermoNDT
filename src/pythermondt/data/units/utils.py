from .units import Unit, arbitrary, dimensionless, undefined


def generate_label(unit: Unit) -> str:
    """Generates a label from a Unit object.

    Args:
        unit (Unit): The Unit object to generate the label from.

    Returns:
        str: The generated label.
    """
    # Check for special units
    if unit == dimensionless:
        return "dimensionless"
    if unit == arbitrary:
        return "arbitrary"
    if unit == undefined:
        return ""

    return f"{unit.quantity} in {unit.symbol}"
