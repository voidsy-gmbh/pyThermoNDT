from ._unit import UnitInfo, is_unit_info

class Units:
    '''Container for all units inside pythermondt.'''
    # Special units
    dimensionless = UnitInfo(name="dimensionless", quantity="dimensionless", symbol="1")
    arbitrary = UnitInfo(name="arbitrary", quantity="arbitrary", symbol="a. u.") # Special unit for data that is not really dimensionaless but has not been processed yet (e.g. temp without LUT applied)
    undefined = UnitInfo(name="undefined", quantity="undefined", symbol="N/A") # Return value for datasets without a unit defined

    # Temperature
    kelvin = UnitInfo(name="kelvin", quantity="temperature", symbol="K")
    celsius = UnitInfo(name="celsius", quantity="temperature", symbol="°C")
    
    # Time 
    second = UnitInfo(name="second", quantity="time", symbol="s")
    millisecond = UnitInfo(name="millisecond", quantity="time", symbol="ms")
    
    # Add more units here as needed
    
    @classmethod
    def print_available_units(cls):
        '''Print all units in the Units class.'''
        print("Available units:")
        print("name: quantity (symbol)\n")    
        for unit in cls.__dict__.values():
            if is_unit_info(unit):
                print(f"{unit['name']}: {unit['quantity']} ({unit['symbol']})")

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