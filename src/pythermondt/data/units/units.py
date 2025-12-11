import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Unit:
    """Dataclass that stores unit information."""

    name: str  # The name of the unit (e.g. kelvin, celsius, etc.)
    quantity: str  # The quantity the unit represents (e.g. temperature, time, etc.)
    symbol: str  # The symbol of the unit (e.g. K, °C, etc.)

    def to_dict(self) -> dict:
        """Convert the Unit dataclass to a dictionary.

        Returns:
            dict: The dictionary representation of the Unit.
        """
        return asdict(self)


def print_available_units() -> None:
    """Print all available units."""
    module = sys.modules[__name__]
    for name in dir(module):
        if isinstance(obj := getattr(module, name), Unit):
            print(obj)


# Units definitions
# Special units
dimensionless = Unit(name="dimensionless", quantity="dimensionless", symbol="1")
# arbitrary: Special unit for data that is not really dimensionless but has not been processed yet
# (e.g. temp without LUT applied)
arbitrary = Unit(name="arbitrary", quantity="arbitrary", symbol="a. u.")
# undefined: Return value for datasets without a unit defined
undefined = Unit(name="undefined", quantity="undefined", symbol="N/A")

# Temperature
kelvin = Unit(name="kelvin", quantity="temperature", symbol="K")
celsius = Unit(name="celsius", quantity="temperature", symbol="°C")

# Time
second = Unit(name="second", quantity="time", symbol="s")
millisecond = Unit(name="millisecond", quantity="time", symbol="ms")

# Frequency
hertz = Unit(name="hertz", quantity="frequency", symbol="Hz")

# Add more units here as needed
