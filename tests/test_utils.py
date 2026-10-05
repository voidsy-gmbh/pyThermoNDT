import pytest

from tests.utils import format_container_diff


@pytest.mark.parametrize(
    "differences, expected",
    [
        ([], ""),
        (
            ["Data content mismatch at path /Values", "Attribute value mismatch at /MetaData['value']: 1 vs 2"],
            "  - Data content mismatch at path /Values\n  - Attribute value mismatch at /MetaData['value']: 1 vs 2",
        ),
    ],
)
def test_format_container_diff(differences: list[str], expected: str, capsys: pytest.CaptureFixture[str]):
    """Format lists and iterators without writing to stdout."""
    assert format_container_diff(differences) == expected
    assert format_container_diff(iter(differences)) == expected
    assert capsys.readouterr().out == ""
