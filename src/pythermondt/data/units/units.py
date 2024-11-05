from typing import TypedDict
from enum import Enum

class UnitInfo(TypedDict):
    '''TypedDict for unit information.'''
    name: str
    quantity: str
    symbol: str

class Units(Enum):
    '''Enum for all units inside pythermondt.'''
    # Temperature
    kelvin = UnitInfo(name="kelvin", quantity="temperature", symbol="K"),
    celsius = UnitInfo(name="celsius", quantity="temperature", symbol="°C"),
    
    # Time 
    second = UnitInfo(name="second", quantity="time", symbol="s"),
    millisecond = UnitInfo(name="millisecond", quantity="time", symbol="ms"),

    # Generic
    dimensionless = UnitInfo(name="dimensionless", quantity="dimensionless", symbol="1"),
    arbitrary = UnitInfo(name="arbitrary", quantity="arbitrary", symbol="a. u.")

    # Add more units here if needed