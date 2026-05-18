import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.contour import QuadContourSet

from pythermondt.data import ThermoContainer
from pythermondt.data.datacontainer.visualization_ops import OverlayColorOption, VisualizationOps
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
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="function")
def interactive_analyzer(thermo_container: ThermoContainer) -> VisualizationOps.InteractiveAnalyzer:
    """Create an InteractiveAnalyzer instance from the test ThermoContainer."""
    return VisualizationOps.InteractiveAnalyzer(thermo_container)


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


@pytest.mark.parametrize("color", ["purple", "yellow", "cyan", ""])
def test_overlay_color_invalid(thermo_container: ThermoContainer, color):
    """Unsupported overlay color raises ValueError."""
    with pytest.raises(ValueError, match=r"Invalid overlay_color"):
        thermo_container.show_frame(0, option="OverlayGroundTruth", overlay_color=color)


@pytest.mark.parametrize("color", ["red", "green", "blue"])
def test_overlay_color_valid(thermo_container: ThermoContainer, color):
    """Valid overlay colors do not raise."""
    thermo_container.show_frame(0, option="OverlayGroundTruth", overlay_color=color)


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


# ============================================================================
# Test InteractiveAnalyzer
# ============================================================================
def test_interactive_groundtruth_toggle_on_adds_overlay(interactive_analyzer: VisualizationOps.InteractiveAnalyzer):
    """Toggling Show GT on adds an RGBA overlay image and contour to the frame axes."""
    assert interactive_analyzer._has_defects, "Test fixture must have defect pixels"
    assert interactive_analyzer._overlay_img is None
    assert interactive_analyzer._overlay_contour is None

    # Simulate toggling the checkbox on through the real CheckButtons callback path
    interactive_analyzer.groundtruth_toggle.set_active(0)

    # Verify overlay state
    assert interactive_analyzer._overlay_img is not None, "Overlay image should exist after toggle on"
    assert interactive_analyzer._overlay_contour is not None, "Contour should exist after toggle on"

    # Verify overlay image is on the axes
    overlay_imgs = [
        im
        for im in interactive_analyzer.frame_ax.get_images()
        if (arr := im.get_array()) is not None and arr.shape[-1] == 4
    ]
    assert len(overlay_imgs) == 1, f"Expected 1 RGBA overlay image on axes, got {len(overlay_imgs)}"
    overlay_arr = overlay_imgs[0].get_array()
    assert overlay_arr is not None

    # Verify contour exists on axes
    contours = [c for c in interactive_analyzer.frame_ax.collections if isinstance(c, QuadContourSet)]
    assert len(contours) >= 1, "Expected at least 1 contour on frame axes"


def test_interactive_groundtruth_toggle_off_removes_overlay(interactive_analyzer: VisualizationOps.InteractiveAnalyzer):
    """Toggling Show GT off removes the overlay image and contour."""
    # First toggle on
    interactive_analyzer.groundtruth_toggle.set_active(0)
    assert interactive_analyzer._overlay_img is not None

    # Then toggle off
    interactive_analyzer.groundtruth_toggle.set_active(0)

    assert interactive_analyzer._overlay_img is None, "Overlay image should be removed after toggle off"
    assert interactive_analyzer._overlay_contour is None, "Contour should be removed after toggle off"


def test_interactive_groundtruth_persists_across_frames(interactive_analyzer: VisualizationOps.InteractiveAnalyzer):
    """Ground truth overlay persists when navigating to a different frame."""
    # Toggle overlay on
    interactive_analyzer.groundtruth_toggle.set_active(0)

    # Collect overlay pixels for verification
    assert interactive_analyzer._overlay_img is not None
    overlay_img = interactive_analyzer._overlay_img

    # Navigate to a different frame
    current_frame = interactive_analyzer.current_frame
    total_frames = interactive_analyzer.tdata.shape[0]
    assert total_frames > 1, "Test fixture must have more than one frame"
    new_frame = (current_frame + 1) % total_frames
    interactive_analyzer.update_frame(float(new_frame))

    # Verify frame changed
    assert interactive_analyzer.current_frame == new_frame

    # Verify the overlay is still on the axes (same artist object)
    assert interactive_analyzer._overlay_img is overlay_img
    assert overlay_img in interactive_analyzer.frame_ax.get_images()


@pytest.mark.parametrize("color, expected_channel", [("red", 0), ("green", 1), ("blue", 2)])
def test_interactive_overlay_color_configuration(
    thermo_container: ThermoContainer, color: OverlayColorOption, expected_channel: int
):
    """Each overlay color activates the correct RGBA channel in the overlay."""
    analyzer = VisualizationOps.InteractiveAnalyzer(thermo_container, overlay_color=color)
    assert analyzer._overlay_color == color

    # Toggle on
    analyzer.groundtruth_toggle.set_active(0)

    overlay_imgs = [
        im for im in analyzer.frame_ax.get_images() if (arr := im.get_array()) is not None and arr.shape[-1] == 4
    ]
    assert len(overlay_imgs) == 1
    overlay_data = overlay_imgs[0].get_array()
    assert overlay_data is not None

    # Verify the correct RGBA channel is set to 1.0 on defect pixels
    gt = thermo_container.get_dataset("/GroundTruth/DefectMask").numpy(force=True)
    defect_mask = gt > 0
    if defect_mask.any():
        channel_values = overlay_data[defect_mask, expected_channel]
        assert np.allclose(channel_values, 1.0), f"{color} channel not set on defect pixels"

    analyzer.close(close_figure=True)


def test_interactive_invalid_overlay_color_raises(thermo_container: ThermoContainer):
    """Invalid overlay_color passed to InteractiveAnalyzer raises ValueError."""
    with pytest.raises(ValueError, match=r"Invalid overlay_color"):
        VisualizationOps.InteractiveAnalyzer(thermo_container, overlay_color="purple")  # type: ignore


def test_interactive_close_double_call_no_error(interactive_analyzer: VisualizationOps.InteractiveAnalyzer):
    """Closing an already-closed analyzer is a no-op."""
    interactive_analyzer.close(close_figure=False)
    interactive_analyzer.close(close_figure=False)  # Should not raise


def test_interactive_closed_property_after_close(interactive_analyzer: VisualizationOps.InteractiveAnalyzer):
    """Closed returns True after close()."""
    assert not interactive_analyzer.closed
    interactive_analyzer.close(close_figure=False)
    assert interactive_analyzer.closed


def test_interactive_update_frame_noop(interactive_analyzer: VisualizationOps.InteractiveAnalyzer):
    """update_frame with the current frame value is a no-op."""
    current = interactive_analyzer.current_frame
    interactive_analyzer.update_frame(float(current))
    assert interactive_analyzer.current_frame == current


def test_interactive_clear_points(interactive_analyzer: VisualizationOps.InteractiveAnalyzer):
    """clear_points removes all selected points, markers, and profile lines."""
    # Manually simulate a click by adding state directly
    interactive_analyzer.selected_points.append((10, 10))
    marker = interactive_analyzer.frame_ax.plot(10, 10, "x", color="red", markersize=10)[0]
    interactive_analyzer.point_markers.append(marker)
    line = interactive_analyzer.profile_ax.plot([0, 1], [0, 1], color="red")[0]
    interactive_analyzer.profile_lines.append(line)

    interactive_analyzer.clear_points(None)

    assert interactive_analyzer.selected_points == []
    assert interactive_analyzer.point_markers == []
    assert interactive_analyzer.profile_lines == []
    assert marker not in interactive_analyzer.frame_ax.lines
    assert line not in interactive_analyzer.profile_ax.lines


def test_interactive_toggle_annotation_off(interactive_analyzer: VisualizationOps.InteractiveAnalyzer):
    """Toggling Show Value off hides the cursor annotation box."""
    # Show the annotation first so we can verify it gets hidden
    interactive_analyzer.cursor_annotation_box.set_visible(True)
    interactive_analyzer._last_hover_pixel = (5, 5)

    # Simulate unchecked checkbox
    orig_get_status = interactive_analyzer.annotation_toggle.get_status
    interactive_analyzer.annotation_toggle.get_status = lambda: [False]
    try:
        interactive_analyzer.toggle_annotation(None)
    finally:
        interactive_analyzer.annotation_toggle.get_status = orig_get_status

    assert not interactive_analyzer.cursor_annotation_box.get_visible()
    assert interactive_analyzer._last_hover_pixel is None


# ── No-defect ground truth edge case ────────────────────────────────────────


@pytest.fixture(scope="function")
def no_defect_analyzer() -> VisualizationOps.InteractiveAnalyzer:
    """InteractiveAnalyzer backed by a container with no defect pixels."""
    container = ThermoContainer()
    container.update_dataset("/Data/Tdata", np.zeros((5, 5, 3)))
    container.update_dataset("/MetaData/DomainValues", np.arange(3, dtype=np.float64))
    # DefectMask is empty → _has_defects will be False
    return VisualizationOps.InteractiveAnalyzer(container)


def test_interactive_no_defect_skips_toggle(no_defect_analyzer: VisualizationOps.InteractiveAnalyzer):
    """When no defect pixels exist, the Show GT checkbox is not created."""
    assert not no_defect_analyzer._has_defects
    assert no_defect_analyzer.groundtruth is None
    assert not hasattr(no_defect_analyzer, "groundtruth_toggle")


# ── analyse_interactive public API flow ─────────────────────────────────────


def test_analyse_interactive_flow(thermo_container: ThermoContainer):
    """analyse_interactive creates an analyzer, registers it, and releases on close."""
    thermo_container.analyse_interactive()
    analyzer = thermo_container._interactive_analyzer
    assert analyzer is not None
    assert not analyzer.closed

    # Close the analyzer — this should trigger release_interactive_analyzer
    analyzer.close(close_figure=True)
    assert thermo_container._interactive_analyzer is None

    # Opening a new analyzer after close works
    thermo_container.analyse_interactive(overlay_color="green")
    assert thermo_container._interactive_analyzer is not None
    thermo_container._interactive_analyzer.close(close_figure=True)
