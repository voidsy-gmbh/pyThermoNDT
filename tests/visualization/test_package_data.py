from importlib.resources import files


def test_static_assets_are_available():
    """Visualization static files are bundled with the package."""
    static_dir = files("pythermondt.visualization").joinpath("static")
    assert static_dir.joinpath("index.html").is_file()
    assert static_dir.joinpath("app.js").is_file()
    assert static_dir.joinpath("styles.css").is_file()
