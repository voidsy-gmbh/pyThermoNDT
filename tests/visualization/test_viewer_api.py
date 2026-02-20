from pythermondt.visualization.viewer import BaseViewer, view


def test_view_blocks_by_default(monkeypatch, viewer_container):
    """`view()` blocks by default and stops cleanly on interrupt."""
    events = {"start": False, "wait": False, "stop": False}

    def fake_start(self, open_browser: bool = True):
        events["start"] = True
        return "http://127.0.0.1:0"

    def fake_wait(self):
        events["wait"] = True
        raise KeyboardInterrupt

    def fake_stop(self):
        events["stop"] = True

    monkeypatch.setattr(BaseViewer, "start", fake_start)
    monkeypatch.setattr(BaseViewer, "wait", fake_wait)
    monkeypatch.setattr(BaseViewer, "stop", fake_stop)

    viewer = view(viewer_container, open_browser=False)

    assert isinstance(viewer, BaseViewer)
    assert events == {"start": True, "wait": True, "stop": True}


def test_view_non_blocking_returns_running_viewer(viewer_container):
    """`block=False` returns immediately with a running viewer."""
    viewer = view(viewer_container, open_browser=False, block=False)

    try:
        assert viewer.url.startswith("http://127.0.0.1:")
    finally:
        viewer.stop()
