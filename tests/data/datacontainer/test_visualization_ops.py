import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.contour import QuadContourSet

from pythermondt.data import ThermoContainer
from pythermondt.readers import LocalReader


@pytest.fixture(scope="module")
def thermo_container() -> ThermoContainer:
    """ThermoContainer with simulation data including defect mask for visualization tests."""
    reader = LocalReader(pattern="./tests/assets/integration/simulation/source2.mat")
    container = next(iter(reader))
    assert isinstance(container, ThermoContainer)
    return container


@pytest.fixture(autouse=True)
def _setup_matplotlib(monkeypatch):
    """Use Agg backend and disable plt.show for all visualization tests."""
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)


def test_frame_number_negative(thermo_container: ThermoContainer):
    """Negative frame number raises ValueError."""
    with pytest.raises(ValueError, match=r"Frame -1 out of range \[0, \d+\)"):
        thermo_container.show_frame(-1)


def test_frame_number_out_of_range(thermo_container: ThermoContainer):
    """Frame number >= total frames raises ValueError."""
    total = thermo_container.get_dataset("/Data/Tdata").shape[2]
    with pytest.raises(ValueError, match=rf"Frame {total} out of range \[0, {total}\)"):
        thermo_container.show_frame(total)


@pytest.mark.parametrize("frame", [0, "last"])
def test_frame_number_valid(thermo_container: ThermoContainer, frame):
    """Valid frame numbers (first, last) do not raise."""
    if isinstance(frame, int):
        fn = frame
    else:
        fn = thermo_container.get_dataset("/Data/Tdata").shape[2] - 1
    thermo_container.show_frame(fn)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, -0.5, 1.5])
def test_alpha_out_of_range(thermo_container: ThermoContainer, alpha):
    """Overlay alpha outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match=r"Overlay alpha must be in the range \[0, 1\], got"):
        thermo_container.show_frame(0, option="OverlayGroundTruth", overlay_alpha=alpha)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 0.6])
def test_alpha_valid(thermo_container: ThermoContainer, alpha):
    """Overlay alpha within [0, 1] does not raise."""
    thermo_container.show_frame(0, option="OverlayGroundTruth", overlay_alpha=alpha)


def test_default_mode_figure(thermo_container: ThermoContainer):
    """Default mode creates figure with correct size, title, and colorbar."""
    thermo_container.show_frame(0)

    fig = plt.gcf()
    assert fig.get_size_inches() == pytest.approx((7, 6))
    ax = fig.axes[0]
    assert ax.get_title() == "Frame 0"
    assert len(fig.axes) >= 1  # frame axis exists


def test_show_ground_truth_figure(thermo_container: ThermoContainer):
    """ShowGroundTruth mode creates two subplots with correct titles."""
    thermo_container.show_frame(0, option="ShowGroundTruth")

    fig = plt.gcf()
    assert fig.get_size_inches() == pytest.approx((11, 5.5))
    assert len(fig.axes) >= 2  # Both subplots present
    assert fig.axes[0].get_title() == "Frame 0"
    assert fig.axes[1].get_title() == "Ground Truth"


def test_overlay_mode_figure(thermo_container: ThermoContainer):
    """OverlayGroundTruth mode creates figure with colorbar."""
    thermo_container.show_frame(0, option="OverlayGroundTruth")

    fig = plt.gcf()
    assert fig.get_size_inches() == pytest.approx((7, 6))
    ax = fig.axes[0]
    assert ax.get_title() == "Frame 0"


@pytest.mark.parametrize("color", ["red", "green", "blue"])
def test_overlay_rgba_channel(thermo_container: ThermoContainer, color):
    """Each overlay color activates the correct RGBA channel."""
    thermo_container.show_frame(0, option="OverlayGroundTruth", overlay_color=color)

    fig = plt.gcf()
    ax = fig.axes[0]
    overlay_images = [im for im in ax.get_images() if (arr := im.get_array()) is not None and arr.shape[-1] == 4]
    assert len(overlay_images) == 1, f"Expected 1 RGBA overlay image, got {len(overlay_images)}"

    overlay_array = overlay_images[0].get_array()
    assert overlay_array is not None
    assert overlay_array.ndim == 3 and overlay_array.shape[2] == 4  # HxWx4

    channel_idx = {"red": 0, "green": 1, "blue": 2}[color]

    # Defect pixels (gt > 0) should have the correct channel set to 1.0
    gt = thermo_container.get_dataset("/GroundTruth/DefectMask").numpy(force=True)
    defect_mask = gt > 0

    if defect_mask.any():
        channel_values = overlay_array[defect_mask, channel_idx]
        assert np.allclose(channel_values, 1.0), f"{color} channel not set on defect pixels"

        # Non-defect pixels should have all channels at 0 (fully transparent)
        non_defect_mask = ~defect_mask
        assert np.allclose(overlay_array[non_defect_mask, :], 0.0), "Non-defect pixels should be transparent"


@pytest.mark.parametrize("color", ["red", "green", "blue"])
def test_overlay_contour_color(thermo_container: ThermoContainer, color):
    """Contour color should be 0.6x darker than the overlay color."""
    # Calculate expected contour color as 0.6 times the RGBA values of the overlay color
    expected_r, expected_g, expected_b, _ = to_rgba(color)
    expected_contour = (expected_r * 0.6, expected_g * 0.6, expected_b * 0.6)

    thermo_container.show_frame(0, option="OverlayGroundTruth", overlay_color=color)

    fig = plt.gcf()
    ax = fig.axes[0]
    # Contour adds QuadContourSet collections to the axis
    contours = list(ax.collections)
    assert contours, "Expected contour collections not found in the plot"
    assert all(isinstance(c, QuadContourSet) for c in contours), "Expected contour collections to be QuadContourSet"

    # Extract the first contour set and check its color
    first = contours[0]
    assert isinstance(first, QuadContourSet), f"Expected a QuadContourSet, got {type(first)}"
    actual_colors = first.colors
    assert isinstance(actual_colors, list), "Expected contour colors to be defined"
    actual = actual_colors[0][:3]
    assert np.allclose(actual, expected_contour, atol=0.01), (
        f"Contour color {actual} does not match expected {expected_contour}"
    )


def test_coordinate_negative_x(thermo_container: ThermoContainer):
    """Negative X coordinate raises ValueError."""
    with pytest.raises(ValueError, match=r"Pixel \(-1, 0\) out of bounds"):
        thermo_container.show_pixel_profile(-1, 0)


def test_coordinate_negative_y(thermo_container: ThermoContainer):
    """Negative Y coordinate raises ValueError."""
    with pytest.raises(ValueError, match=r"Pixel \(0, -1\) out of bounds"):
        thermo_container.show_pixel_profile(0, -1)


def test_coordinate_out_of_range(thermo_container: ThermoContainer):
    """Coordinate beyond data dimensions raises ValueError."""
    data = thermo_container.get_dataset("/Data/Tdata")
    width, height = data.shape[1], data.shape[0]
    with pytest.raises(ValueError, match=rf"Pixel \({width}, {height}\) out of bounds"):
        thermo_container.show_pixel_profile(width, height)


@pytest.mark.parametrize("x, y", [(0, 0), (50, 50)])
def test_coordinate_valid(thermo_container: ThermoContainer, x, y):
    """Valid coordinates do not raise."""
    thermo_container.show_pixel_profile(x, y)
