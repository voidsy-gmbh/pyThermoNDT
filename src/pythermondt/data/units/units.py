from typing import TypedDict

class UnitInfo(TypedDict):
    '''TypedDict for unit information.'''
    name: str
    quantity: str
    symbol: str

UNITS = {
    # Temperature
    "kelvin": UnitInfo(name="Kelvin", quantity="Temperature", symbol="K"),
    "celsius": UnitInfo(name="Celsius", quantity="Temperature", symbol="°C"),

    # Time units
    "second": UnitInfo(name="second", quantity="Time", symbol="s"),
    "millisecond": UnitInfo(name="millisecond", quantity="Time", symbol="ms"),

    # Generic
    "dimensionless": UnitInfo(name="dimensionless", quantity="dimensionless", symbol="1"),

    # Raw data
    "arbitrary": UnitInfo(name="arbitrary", quantity="arbitrary", symbol="a. u."),
}