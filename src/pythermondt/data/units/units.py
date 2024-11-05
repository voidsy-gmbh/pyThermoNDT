from typing import TypedDict
from enum import Enum

class UnitInfo(TypedDict):
    '''TypedDict for unit information.'''
    name: str
    quantity: str
    symbol: str

class Units:
    '''Container for all units inside pythermondt.'''
    # Temperature
    kelvin = UnitInfo(name="kelvin", quantity="temperature", symbol="K")
    celsius = UnitInfo(name="celsius", quantity="temperature", symbol="°C")
    
    # Time 
    second = UnitInfo(name="second", quantity="time", symbol="s")
    millisecond = UnitInfo(name="millisecond", quantity="time", symbol="ms")

    # Generic
    dimensionless = UnitInfo(name="dimensionless", quantity="dimensionless", symbol="1")
    arbitrary = UnitInfo(name="arbitrary", quantity="arbitrary", symbol="a. u.")

    # Add more units here as needed

    def __init__(self):
        raise TypeError("This class is static and should not be instantiated.")

# Test
if __name__ == "__main__":
    print(type(Units.kelvin))
    print(Units.kelvin)
    print(Units.celsius)
    print(Units.second)
    print(Units.millisecond)
    print(Units.dimensionless)
    print(Units.arbitrary)