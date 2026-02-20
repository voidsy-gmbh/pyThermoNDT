from __future__ import annotations

from typing import TYPE_CHECKING

from .adapter import DataContainerAdapter
from .server import ViewerServer

if TYPE_CHECKING:
    from ..data import DataContainer


class BaseViewer:
    """Lightweight browser viewer for exploring a DataContainer structure."""

    def __init__(self, container: DataContainer, host: str = "127.0.0.1", port: int = 0):
        self._adapter = DataContainerAdapter(container)
        self._server = ViewerServer(self._adapter, host=host, port=port)

    @property
    def url(self) -> str:
        """Get viewer URL.

        Returns:
            str: URL of running viewer server.
        """
        return self._server.url

    def start(self, open_browser: bool = True) -> str:
        """Start viewer server.

        Args:
            open_browser (bool): If True, open viewer in default browser.

        Returns:
            str: URL of running viewer server.
        """
        return self._server.start(open_browser=open_browser)

    def stop(self) -> None:
        """Stop viewer server."""
        self._server.stop()

    def wait(self, poll_interval: float = 0.2) -> None:
        """Block until the viewer server stops.

        Args:
            poll_interval (float): Polling interval in seconds for interruptible waiting.
        """
        self._server.wait(poll_interval=poll_interval)

    def __enter__(self) -> BaseViewer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def view(
    container: DataContainer,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    block: bool = True,
) -> BaseViewer:
    """Create and start a BaseViewer.

    Args:
        container (DataContainer): Container to visualize.
        host (str): Host binding for local server.
        port (int): Port binding. Use 0 for automatic free port.
        open_browser (bool): If True, open viewer URL in default browser.
        block (bool): If True, keep process alive until interrupted (Ctrl+C).

    Returns:
        BaseViewer: Running viewer instance.
    """
    viewer = BaseViewer(container=container, host=host, port=port)
    viewer.start(open_browser=open_browser)

    if block:
        try:
            viewer.wait()
        except KeyboardInterrupt:
            pass
        finally:
            viewer.stop()

    return viewer
