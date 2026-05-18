from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import to_rgba
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.widgets import Button, CheckButtons, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..units import generate_label
from .attribute_ops import AttributeOps
from .dataset_ops import DatasetOps
from .group_ops import GroupOps

# Type aliases for visualization options
FrameOption: TypeAlias = Literal["ShowGroundTruth", "OverlayGroundTruth", ""]
OverlayColorOption: TypeAlias = Literal["red", "green", "blue"]


def _create_overlay_rgba(
    binary_mask: np.ndarray,
    color: OverlayColorOption,
    alpha: float = 0.6,
) -> np.ndarray:
    """Build an RGBA overlay array for a ground truth binary mask.

    Args:
        binary_mask: Boolean array (H x W) where True marks defect pixels.
        color: One of "red", "green", "blue".
        alpha: Overlay opacity in the range [0, 1]. Defaults to 0.6.

    Returns:
        RGBA array (H x W x 4) with the configured color channel set to 1.0
        on defect pixels and the alpha channel set to *alpha*.
    """
    height, width = binary_mask.shape
    channel_idx = {"red": 0, "green": 1, "blue": 2}[color]

    overlay = np.zeros((height, width, 4))
    overlay[binary_mask, channel_idx] = 1.0
    overlay[binary_mask, 3] = alpha

    return overlay


class VisualizationOps(GroupOps, DatasetOps, AttributeOps):
    _interactive_analyzer: "VisualizationOps.InteractiveAnalyzer | None" = None

    # TODO: Refactor visualization logic to reduce the tight coupling between data handling and visualization.
    class InteractiveAnalyzer:  # pylint: disable=too-many-instance-attributes
        def __init__(self, parent: "VisualizationOps", overlay_color: OverlayColorOption = "red"):
            """Initialize the interactive analyzer for thermographic data visualization.

            Args:
                parent (VisualizationOps): The parent container for the interactive analysis.
                overlay_color: Color for the ground truth overlay. One of "red", "green", "blue".
                    Defaults to "red".
            """
            # 1.) Validate and store overlay configuration
            if overlay_color not in {"red", "green", "blue"}:
                raise ValueError(f"Invalid overlay_color '{overlay_color}'. Must be one of: red, green, blue")
            self._overlay_color = overlay_color

            # 2.) Retrieve data from the container
            self.container = parent
            # Transpose to (frame, y, x) for faster access - this avoids the need to squeeze the data
            self.tdata = parent.get_dataset("/Data/Tdata").numpy(force=True).transpose(2, 0, 1)
            self.domain_values = parent.get_dataset("/MetaData/DomainValues").numpy(force=True)
            self.data_unit = parent.get_unit("/Data/Tdata")
            self.domain_unit = parent.get_unit("/MetaData/DomainValues")

            # Load ground truth and check for any defect pixels
            gt_tensor = parent.get_dataset("/GroundTruth/DefectMask")
            self.groundtruth: np.ndarray | None = None
            if gt_tensor is not None:
                gt = gt_tensor.numpy(force=True)
                if (gt > 0).any():
                    self.groundtruth = gt

            # 3.) Setup the figure, axes and colorbar
            # Create a dedicated bottom row for controls so widgets never overlap plot labels.
            self.fig = plt.figure(figsize=(16, 7))
            gs = self.fig.add_gridspec(2, 16, height_ratios=[24, 1], hspace=0.35, wspace=0.7)
            self.frame_ax = self.fig.add_subplot(gs[0, :8])
            self.profile_ax = self.fig.add_subplot(gs[0, 8:])

            # Initialize the frame display
            self.current_frame = 0  # type: int
            self.current_frame_data = self.tdata[self.current_frame]  # type: np.ndarray
            initialize_vmin = round(self.current_frame_data.min(), 8)
            initialize_vmax = round(self.current_frame_data.max(), 8)
            self.frame_img = self.frame_ax.imshow(
                self.current_frame_data, aspect="auto", cmap="plasma", vmin=initialize_vmin, vmax=initialize_vmax
            )
            self.frame_ax.set_title(f"Frame {self.current_frame}")

            # Setup the profile plot
            self.profile_ax.set_xlabel(generate_label(self.domain_unit))
            self.profile_ax.set_ylabel(generate_label(self.data_unit))
            self.profile_ax.grid(True)

            # Add colorbar with formatter to avoid offset
            formatter = ticker.ScalarFormatter(useMathText=False, useOffset=False)
            self.colorbar = plt.colorbar(self.frame_img, ax=self.frame_ax, format=formatter)

            # 4.) Setup the interactive elements in the dedicated bottom row
            # Setup the slider
            slider_ax = self.fig.add_subplot(gs[1, :9])
            self.frame_slider = Slider(
                ax=slider_ax, label="Frame", valmin=0, valmax=self.tdata.shape[0] - 1, valinit=0, valstep=1
            )

            # Setup the clear button
            clear_ax = self.fig.add_subplot(gs[1, 10:12])
            self.clear_button = Button(clear_ax, "Clear Points")

            # Create checkbox for annotation toggle
            check_ax = self.fig.add_subplot(gs[1, 12:14])
            self.annotation_toggle = CheckButtons(
                check_ax,
                ["Show Value"],
                [True],  # Initially checked
            )

            # Create checkbox for ground truth overlay toggle (only if defects exist)
            if self.groundtruth is not None:
                check_gt_ax = self.fig.add_subplot(gs[1, 14:])
                self.groundtruth_toggle = CheckButtons(
                    check_gt_ax,
                    ["Show GT"],
                    [False],  # Initially unchecked
                )

            # 5.) Initialize state variables
            # Store selected points and their profiles
            self.selected_points: list[tuple[int, int]] = []
            self.point_markers: list[Line2D] = []
            self.profile_lines: list[Line2D] = []
            self.colors = ["red", "blue", "green", "purple"]  # Colors for up to 4 points
            self._last_hover_pixel: tuple[int, int] | None = None
            self._closed = False

            # Ground truth overlay state
            self._overlay_img: AxesImage | None = None
            self._overlay_contour: QuadContourSet | None = None

            # Initialize annotation box once
            self.cursor_annotation_text = TextArea("", textprops={"color": "white", "backgroundcolor": "black"})
            self.cursor_annotation_box = AnnotationBbox(
                self.cursor_annotation_text,
                (0, 0),  # Initial position
                xybox=(10, 10),
                boxcoords="offset points",
                frameon=False,
            )
            self.cursor_annotation_box.set_visible(False)  # Hide initially
            self.frame_ax.add_artist(self.cursor_annotation_box)

            # 6.) Connect events
            self._slider_cid = self.frame_slider.on_changed(self.update_frame)
            self._clear_btn_cid = self.clear_button.on_clicked(self.clear_points)
            self._annotation_cid = self.annotation_toggle.on_clicked(self.toggle_annotation)
            self._gt_cid: int | None = None
            if self.groundtruth is not None:
                self._gt_cid = self.groundtruth_toggle.on_clicked(self.toggle_groundtruth)
            self._canvas_connection_ids = [
                self.fig.canvas.mpl_connect("button_press_event", self.on_click),
                self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move),
                self.fig.canvas.mpl_connect("close_event", self.on_close),
            ]

            # 7.) Initialize blitting for faster rendering (if possible)
            self.fig.canvas.draw_idle()

        @property
        def closed(self) -> bool:
            """Whether the interactive analyzer has been closed and cleaned up."""
            return self._closed

        def close(self, close_figure: bool = False):
            """Disconnect callbacks and release analyzer resources."""
            if self._closed:
                return

            self._closed = True

            for connection_id in self._canvas_connection_ids:
                self.fig.canvas.mpl_disconnect(connection_id)

            self.frame_slider.disconnect(self._slider_cid)
            self.clear_button.disconnect(self._clear_btn_cid)
            self.annotation_toggle.disconnect(self._annotation_cid)
            if self._gt_cid is not None:
                self.groundtruth_toggle.disconnect(self._gt_cid)

            self.container.release_interactive_analyzer(self)

            if close_figure and plt.fignum_exists(self.fig.number):
                plt.close(self.fig)

        def on_close(self, event):  # pylint: disable=unused-argument
            """Handle figure close event by disconnecting callbacks."""
            self.close(close_figure=False)

        def toggle_annotation(self, event):  # pylint: disable=unused-argument
            """Toggle cursor annotation on/off."""
            # Hide annotation if disabled
            if not self.annotation_toggle.get_status()[0]:
                self.cursor_annotation_box.set_visible(False)
                self._last_hover_pixel = None
                self.fig.canvas.draw_idle()

        def toggle_groundtruth(self, event):  # pylint: disable=unused-argument
            """Toggle ground truth overlay on/off."""
            if self.groundtruth is None:
                return
            if self.groundtruth_toggle.get_status()[0]:
                # Show overlay: create RGBA mask and contour from ground truth based on requested color
                binary_gt = self.groundtruth > 0
                overlay = _create_overlay_rgba(binary_gt, self._overlay_color)
                self._overlay_img = self.frame_ax.imshow(overlay, aspect="auto", interpolation="none")

                # Add contour outline in a darker shade of the overlay color
                r, g, b, _ = to_rgba(self._overlay_color)
                contour_color = (r * 0.6, g * 0.6, b * 0.6)
                self._overlay_contour = self.frame_ax.contour(
                    binary_gt.astype(float), levels=[0.5], colors=[contour_color], linewidths=1.5
                )
            else:
                # Remove overlay and contour
                if self._overlay_img is not None:
                    self._overlay_img.remove()
                    self._overlay_img = None
                if self._overlay_contour is not None:
                    self._overlay_contour.remove()
                    self._overlay_contour = None

            self.fig.canvas.draw_idle()

        def on_mouse_move(self, event):
            """Update annotation when mouse moves over the image."""
            # Check if annotation is enabled
            if not self.annotation_toggle.get_status()[0]:
                return

            if event.inaxes != self.frame_ax:
                if self.cursor_annotation_box.get_visible():
                    self.cursor_annotation_box.set_visible(False)
                    self._last_hover_pixel = None
                    self.fig.canvas.draw_idle()
                return

            if event.xdata is None or event.ydata is None:
                return

            # Get mouse coordinates
            x, y = round(event.xdata), round(event.ydata)

            if 0 <= y < self.current_frame_data.shape[0] and 0 <= x < self.current_frame_data.shape[1]:
                if self._last_hover_pixel == (x, y) and self.cursor_annotation_box.get_visible():
                    return

                # Get current value
                val = self.current_frame_data[y, x]

                # Update annotation
                self.cursor_annotation_box.xy = (x, y)
                self.cursor_annotation_text.set_text(f"({x}, {y})\n{val:.5f}")
                self.cursor_annotation_box.set_visible(True)
                self._last_hover_pixel = (x, y)

                self.fig.canvas.draw_idle()
            elif self.cursor_annotation_box.get_visible():
                self.cursor_annotation_box.set_visible(False)
                self._last_hover_pixel = None
                self.fig.canvas.draw_idle()

        def update_frame(self, frame_idx: float):
            """Update the displayed frame."""
            new_frame = int(frame_idx)
            if new_frame == self.current_frame:
                return

            # Extract frame data
            self.current_frame = new_frame
            self.current_frame_data = self.tdata[new_frame]

            # Invalidate hover cache because pixel values changed with the frame
            self._last_hover_pixel = None
            self.cursor_annotation_box.set_visible(False)

            # Update image data and title
            self.frame_img.set_data(self.current_frame_data)
            self.frame_ax.set_title(f"Frame {self.current_frame}")

            # Get the colorbar limits according to min/max of the current frame
            vmin = round(self.current_frame_data.min(), 8)
            vmax = round(self.current_frame_data.max(), 8)

            old_min, old_max = self.frame_img.get_clim()
            if not np.isclose(old_min, vmin, atol=1e-8, rtol=0.0) or not np.isclose(old_max, vmax, atol=1e-8, rtol=0.0):
                self.frame_img.set_clim(vmin=vmin, vmax=vmax)

            # Redraw
            self.fig.canvas.draw_idle()

        def on_click(self, event):
            """Handle click events on the frame plot."""
            if event.inaxes != self.frame_ax:
                return

            if len(self.selected_points) >= 4:
                print("Maximum number of points (4) reached. Clear points to add more.")
                return

            # Check if click is within frame boundaries
            x, y = int(event.xdata), int(event.ydata)
            if not 0 <= y < self.current_frame_data.shape[0] or not 0 <= x < self.current_frame_data.shape[1]:
                return

            # Add point and plot profile
            color = self.colors[len(self.selected_points)]
            self.selected_points.append((x, y))

            # Plot point on frame
            marker = self.frame_ax.plot(x, y, "x", color=color, markersize=10)[0]
            self.point_markers.append(marker)

            # Plot temperature profile
            profile = self.tdata[:, y, x]
            line = self.profile_ax.plot(self.domain_values, profile, color=color, label=f"Point ({x}, {y})")[0]
            self.profile_lines.append(line)
            self.profile_ax.legend()

            self.fig.canvas.draw_idle()

        def clear_points(self, event):  # pylint: disable=unused-argument
            """Clear all selected points and profiles."""
            self.selected_points.clear()
            self.profile_ax.clear()

            for marker in self.point_markers:
                marker.remove()
            self.point_markers.clear()
            self.profile_lines.clear()

            # Reset profile plot
            self.profile_ax.set_xlabel(generate_label(self.domain_unit))
            self.profile_ax.set_ylabel(generate_label(self.data_unit))
            self.profile_ax.grid(True)

            # Redraw frame without points
            self.frame_img.set_data(self.current_frame_data)

            self.fig.canvas.draw_idle()

    def show_frame(
        self,
        frame_number: int,
        option: FrameOption = "",
        cmap: str = "plasma",
        overlay_color: OverlayColorOption = "red",
        overlay_alpha: float = 0.6,
    ):  # pylint: disable=too-many-locals
        """Visualize a specific frame from the dataset with optional ground truth visualization and color mapping.

        Args:
            frame_number (int): The frame number to visualize.
            option (FrameOption): The visualization option to apply.
                Options are "ShowGroundTruth", "OverlayGroundTruth", or an empty string.
            cmap (str): The color map to use for the visualization. Defaults to "plasma".
            overlay_color (OverlayColorOption): Color for the ground truth overlay. Defaults to "red".
            overlay_alpha (float): Opacity for the ground truth overlay, in the range [0, 1]. Defaults to 0.6.

        Raises:
            ValueError: If frame_number is out of valid range.
            ValueError: If overlay_color is not one of "red", "green", "blue".
            ValueError: If overlay_alpha is not in the range [0, 1].
        """
        data = self.get_dataset("/Data/Tdata")
        groundtruth = self.get_dataset("/GroundTruth/DefectMask")

        # Validate frame number
        total_frames = data.shape[2]
        if not 0 <= frame_number < total_frames:
            raise ValueError(f"Frame {frame_number} out of range [0, {total_frames})")

        # Get the frame to show
        data_to_show = data[:, :, frame_number].numpy(force=True)

        # Show the frame with the selected option
        match option:
            case "ShowGroundTruth":
                # Create a figure with two subplots: one for the frame and one for the ground truth
                fig = plt.figure(figsize=(11, 5.5), layout="constrained")
                gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
                frame_ax = fig.add_subplot(gs[0])
                gt_ax = fig.add_subplot(gs[1])

                # Display the frame and ground truth
                im = frame_ax.imshow(data_to_show, aspect="auto", cmap=cmap)
                frame_ax.set_title(f"Frame {frame_number}")
                gt_ax.imshow(groundtruth.numpy(force=True), aspect="auto")
                gt_ax.set_title("Ground Truth")

                # Attach colorbar tightly to the thermal image
                divider = make_axes_locatable(frame_ax)
                cbar_ax = divider.append_axes("right", size="8%", pad=0.16)
                cbar = fig.colorbar(im, cax=cbar_ax, format=ticker.ScalarFormatter(useMathText=False, useOffset=False))
                cbar.ax.tick_params(pad=4)

            case "OverlayGroundTruth":
                _, frame_ax = plt.subplots(figsize=(7, 6))

                im = frame_ax.imshow(data_to_show, aspect="auto", cmap=cmap)
                frame_ax.set_title(f"Frame {frame_number}")

                # Validate overlay parameters
                if overlay_color not in {"red", "green", "blue"}:
                    raise ValueError(f"Invalid overlay_color '{overlay_color}'. Must be one of: red, green, blue")
                if not 0 <= overlay_alpha <= 1:
                    raise ValueError(f"Overlay alpha must be in the range [0, 1], got {overlay_alpha}")

                if groundtruth is not None:
                    # Convert to numpy for matplotlib compatibility
                    gt: np.ndarray = groundtruth.numpy(force=True)
                    binary_gt: np.ndarray = gt > 0

                    # Create RGBA overlay with specified color and alpha
                    overlay = _create_overlay_rgba(binary_gt, overlay_color, overlay_alpha)

                    frame_ax.imshow(overlay, aspect="auto", interpolation="none")

                    # Compute darker version of overlay color for contour
                    r, g, b, _ = to_rgba(overlay_color)
                    contour_color = (r * 0.6, g * 0.6, b * 0.6)

                    # Add contour outline for defect boundaries (darker than overlay)
                    frame_ax.contour(binary_gt.astype(float), levels=[0.5], colors=[contour_color], linewidths=1.5)

                plt.colorbar(im, ax=frame_ax, format=ticker.ScalarFormatter(useMathText=False, useOffset=False))

            # Default case, just show the frame data
            case _:
                _, frame_ax = plt.subplots(figsize=(7, 6))

                im = frame_ax.imshow(data_to_show, aspect="auto", cmap=cmap)
                frame_ax.set_title(f"Frame {frame_number}")

                plt.colorbar(im, ax=frame_ax, format=ticker.ScalarFormatter(useMathText=False, useOffset=False))

        # Show the plot
        plt.show()

    def show_pixel_profile(self, pixel_pos_x: int, pixel_pos_y: int):
        """Plot the profile of a specific pixel across the dataset's domain values with an option for data adjustment.

        The X-axis of the plot is labeled according to the domaintype attribute, reflecting the dataset's domain
        (e.g., time, frequency). The Y-axis is generically labeled as 'Temperature in K'.

        Args:
            pixel_pos_x (int): The X-coordinate (column index) of the pixel.
                Must be within the dataset's second dimension range.
            pixel_pos_y (int): The Y-coordinate (row index) of the pixel.
                Must be within the dataset's first dimension range.

        Raises:
            ValueError: If pixel position is outside valid data bounds.
        """
        # Extract the data from the container
        data = self.get_dataset("/Data/Tdata")
        domainvalues = self.get_dataset("/MetaData/DomainValues")
        data_unit = self.get_unit("/Data/Tdata")
        domain_unit = self.get_unit("/MetaData/DomainValues")

        # Validate pixel positions to be within the data dimensions
        height, width = data.shape[:2]
        if not (0 <= pixel_pos_x < width and 0 <= pixel_pos_y < height):
            raise ValueError(f"Pixel ({pixel_pos_x}, {pixel_pos_y}) out of bounds [{width}x{height}]")

        # Extract temperature profile of the pixel
        temperature_profile = data[pixel_pos_y, pixel_pos_x, :]

        # Plot the temperature profile
        plt.plot(domainvalues.numpy(force=True), temperature_profile.numpy(force=True))
        plt.title(f"Profile of Pixel: {pixel_pos_x},{pixel_pos_y}")
        plt.xlabel(generate_label(domain_unit))
        plt.ylabel(generate_label(data_unit))
        plt.show()

    def release_interactive_analyzer(self, analyzer: "VisualizationOps.InteractiveAnalyzer") -> None:
        """Release the stored interactive analyzer if it matches the provided instance."""
        if self._interactive_analyzer is analyzer:
            self._interactive_analyzer = None

    def analyse_interactive(self, overlay_color: OverlayColorOption = "red"):
        """Launch interactive analysis session for thermographic data visualization.

        Args:
            overlay_color: Color for the ground truth overlay when the "Show GT" toggle is active.
                Must be one of "red", "green", "blue". Defaults to "red".
        """
        if self._interactive_analyzer is not None and not self._interactive_analyzer.closed:
            self._interactive_analyzer.close(close_figure=True)
        self._interactive_analyzer = self.InteractiveAnalyzer(self, overlay_color=overlay_color)
        plt.show()
