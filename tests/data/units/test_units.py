import pytest

from pythermondt.data.units.units import (
    Unit,
    arbitrary,
    celsius,
    dimensionless,
    hertz,
    kelvin,
    millisecond,
    print_available_units,
    second,
    undefined,
)
from pythermondt.data.units.utils import generate_label


def test_unit_creation():
    """Test creating a Unit instance."""
    unit = Unit(name="test", quantity="test_quantity", symbol="T")
    assert unit.name == "test"
    assert unit.quantity == "test_quantity"
    assert unit.symbol == "T"


def test_unit_immutability():
    """Test that Unit is immutable (frozen=True)."""
    unit = Unit(name="test", quantity="test_quantity", symbol="T")
    with pytest.raises(AttributeError):
        unit.name = "new_name"  # type: ignore


def test_unit_equality():
    """Test Unit equality."""
    unit1 = Unit(name="test", quantity="test_quantity", symbol="T")
    unit2 = Unit(name="test", quantity="test_quantity", symbol="T")
    unit3 = Unit(name="different", quantity="test_quantity", symbol="T")
    assert unit1 == unit2
    assert unit1 != unit3


def test_to_dict():
    """Test the to_dict method."""
    unit = Unit(name="test", quantity="test_quantity", symbol="T")
    expected = {"name": "test", "quantity": "test_quantity", "symbol": "T"}
    assert unit.to_dict() == expected


def test_dimensionless_unit():
    """Test dimensionless unit."""
    assert dimensionless.name == "dimensionless"
    assert dimensionless.quantity == "dimensionless"
    assert dimensionless.symbol == "1"


def test_arbitrary_unit():
    """Test arbitrary unit."""
    assert arbitrary.name == "arbitrary"
    assert arbitrary.quantity == "arbitrary"
    assert arbitrary.symbol == "a. u."


def test_undefined_unit():
    """Test undefined unit."""
    assert undefined.name == "undefined"
    assert undefined.quantity == "undefined"
    assert undefined.symbol == "N/A"


def test_kelvin_unit():
    """Test kelvin unit."""
    assert kelvin.name == "kelvin"
    assert kelvin.quantity == "temperature"
    assert kelvin.symbol == "K"


def test_celsius_unit():
    """Test celsius unit."""
    assert celsius.name == "celsius"
    assert celsius.quantity == "temperature"
    assert celsius.symbol == "°C"


def test_second_unit():
    """Test second unit."""
    assert second.name == "second"
    assert second.quantity == "time"
    assert second.symbol == "s"


def test_millisecond_unit():
    """Test millisecond unit."""
    assert millisecond.name == "millisecond"
    assert millisecond.quantity == "time"
    assert millisecond.symbol == "ms"


def test_hertz_unit():
    """Test hertz unit."""
    assert hertz.name == "hertz"
    assert hertz.quantity == "frequency"
    assert hertz.symbol == "Hz"


def test_print_available_units_no_error(capsys):
    """Test that print_available_units runs without error and prints all units."""
    # This should not raise any exceptions
    print_available_units()

    # Check that something was printed
    captured = capsys.readouterr()
    assert captured.out  # Should have some output

    # Check that all predefined units are printed (by their string representation)
    output = captured.out
    assert str(dimensionless) in output
    assert str(arbitrary) in output
    assert str(undefined) in output
    assert str(kelvin) in output
    assert str(celsius) in output
    assert str(second) in output
    assert str(millisecond) in output
    assert str(hertz) in output


def test_generate_label_dimensionless():
    """Test generate_label for dimensionless unit."""
    assert generate_label(dimensionless) == "dimensionless"


def test_generate_label_arbitrary():
    """Test generate_label for arbitrary unit."""
    assert generate_label(arbitrary) == "arbitrary"


def test_generate_label_undefined():
    """Test generate_label for undefined unit."""
    assert generate_label(undefined) == ""


def test_generate_label_temperature_units():
    """Test generate_label for temperature units."""
    assert generate_label(kelvin) == "temperature in K"
    assert generate_label(celsius) == "temperature in °C"


def test_generate_label_time_units():
    """Test generate_label for time units."""
    assert generate_label(second) == "time in s"
    assert generate_label(millisecond) == "time in ms"


def test_generate_label_frequency_units():
    """Test generate_label for frequency units."""
    assert generate_label(hertz) == "frequency in Hz"


def test_generate_label_custom_unit():
    """Test generate_label for a custom unit."""
    custom_unit = Unit(name="custom", quantity="custom_quantity", symbol="C")
    assert generate_label(custom_unit) == "custom_quantity in C"
