from pathlib import Path

from pythermondt.readers import LocalReader
from tests.support.storage import PlainTextParser


def test_local_reader_recursive_includes_nested_test_files(tmp_path: Path):
    """Test recursive LocalReader discovery includes nested supported files."""
    asset_dir = Path(__file__).parents[1] / "assets" / "reader"

    # Keep the test independent from the committed asset directory layout.
    for relative_path in ("sample1.test", "nested/sample3.test"):
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((asset_dir / relative_path).read_text())

    reader = LocalReader(pattern=str(tmp_path), recursive=True, parser=PlainTextParser)

    assert sorted(reader.file_names) == ["sample1.test", "sample3.test"]
