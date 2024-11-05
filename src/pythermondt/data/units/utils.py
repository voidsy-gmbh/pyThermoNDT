from typing import TypeGuard, Any
from .units import UnitInfo, Units

def is_unit(obj: Any) -> TypeGuard[UnitInfo]:
    '''TypeGuard to check if an object is a valid UnitInfo.

    Parameters:
        obj (Any): The object to check.

    Returns:
        TypeGuard[UnitInfo]: True if the object is a valid UnitInfo, False otherwise.
    '''
    return (isinstance(obj, dict) and 
            "name" in obj and isinstance(obj["name"], str) and
            "quantity" in obj and isinstance(obj["quantity"], str) and
            "symbol" in obj and isinstance(obj["symbol"], str))

def generate_label(unit: UnitInfo) -> str:
    '''Generates a label from a UnitInfo object.

    Parameters:
        unit (UnitInfo): The UnitInfo object to generate the label from.

    Returns:
        str: The generated label.
    '''
    # Check for special units
    if unit == Units.dimensionless:
        return "dimensionless"
    if unit == Units.arbitrary:
        return "arbitrary"
    if unit == Units.undefined:
        return ""
    
    return f"{unit['quantity']} in {unit['symbol']}"