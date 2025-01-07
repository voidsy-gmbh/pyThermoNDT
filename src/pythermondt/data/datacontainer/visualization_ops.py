import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import numpy as np
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Tuple
from .group_ops import GroupOps
from .dataset_ops import DatasetOps
from .attribute_ops import AttributeOps
from ..units import generate_label

class VisualizationOps(GroupOps, DatasetOps, AttributeOps):
    class InteractiveAnalyzer:
        def __init__(self, parent: 'VisualizationOps'):
            """Initialize the interactive analyzer for thermographic data visualization.
            
            Args:
                container: DataContainer with thermographic data
            """
            # 1.) Retrieve data from the container
            self.container = parent
            self.tdata = parent.get_dataset('/Data/Tdata').numpy(force=True)
            self.domain_values = parent.get_dataset('/MetaData/DomainValues').numpy(force=True)
            self.data_unit = parent.get_unit('/Data/Tdata')
            self.domain_unit = parent.get_unit('/MetaData/DomainValues')
            
            #2.) Setup the figure, axes and colorbar
            # Create the main figure with two subplots
            self.fig = plt.figure(figsize=(15, 6))
            self.frame_ax = plt.subplot2grid((1, 2), (0, 0))
            self.profile_ax = plt.subplot2grid((1, 2), (0, 1))

            # Initialize the frame display
            self.current_frame = 0 #type: int
            self.current_frame_data = self.tdata[..., self.current_frame].squeeze() #type: np.ndarray
            self.frame_img = self.frame_ax.imshow(
                self.current_frame_data,
                aspect='auto',
                cmap='plasma',
                vmin=self.tdata.min(),
                vmax=self.tdata.max()
            )
            self.frame_ax.set_title(f'Frame {self.current_frame}')

            # Setup the profile plot
            self.profile_ax.set_xlabel(generate_label(self.domain_unit))
            self.profile_ax.set_ylabel(generate_label(self.data_unit))
            self.profile_ax.grid(True)

            # Add colorbar with formatter to avoid offset
            formatter = ticker.ScalarFormatter(useMathText=False, useOffset=False)
            self.colorbar = plt.colorbar(self.frame_img, ax=self.frame_ax, format=formatter)

            # 3.) Setup the interactive elements
            # Setup the slider
            slider_ax = plt.axes((0.2, 0.02, 0.6, 0.03))
            self.frame_slider = Slider(
                ax=slider_ax,
                label='Frame',
                valmin=0,
                valmax=self.tdata.shape[-1]-1,
                valinit=0,
                valstep=1
            )
            
            # Setup the clear button
            clear_ax = plt.axes((0.85, 0.02, 0.1, 0.03))
            self.clear_button = Button(clear_ax, 'Clear Points')

            # Create checkbox for annotation toggle
            check_ax = plt.axes((0.85, 0.07, 0.1, 0.03))  # Position below clear button
            self.annotation_toggle = CheckButtons(
                check_ax, 
                ['Show Value'], 
                [True]  # Initially checked
            )

            # 4.) Initialize state variables
            # Store selected points and their profiles
            self.selected_points: List[Tuple[int, int]] = []
            self.colors = ['red', 'blue', 'green', 'purple']  # Colors for up to 4 points

            # Initialize annotation box once
            self.cursor_annotation_text = TextArea('', textprops={'color': 'white', 'backgroundcolor': 'black'})
            self.cursor_annotation_box = AnnotationBbox(
                self.cursor_annotation_text,
                (0, 0),  # Initial position
                xybox=(10, 10),
                boxcoords="offset points",
                frameon=False
            )
            self.cursor_annotation_box.set_visible(False)  # Hide initially
            self.frame_ax.add_artist(self.cursor_annotation_box)
            
            # 5.) Connect events
            self.frame_slider.on_changed(self.update_frame)
            self.clear_button.on_clicked(self.clear_points)
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.annotation_toggle.on_clicked(self.toggle_annotation)

            # 6.) Initialze blitting for faster rendering (if possible)
            self.fig.canvas.draw_idle()

        def toggle_annotation(self, event):
            """Toggle cursor annotation on/off."""           
            # Hide annotation if disabled
            if not self.annotation_toggle.get_status()[0]:
                self.cursor_annotation_box.set_visible(False)
                self.fig.canvas.draw_idle()

        def on_mouse_move(self, event):
            """Update annotation when mouse moves over the image."""
            # Check if annotation is enabled
            if not self.annotation_toggle.get_status()[0]:
                return

            if event.inaxes != self.frame_ax:
                self.cursor_annotation_box.set_visible(False)
                self.fig.canvas.draw_idle()
                return

            # Get mouse coordinates
            x, y = int(round(event.xdata)), int(round(event.ydata))
            
            if 0 <= y < self.current_frame_data.shape[0] and 0 <= x < self.current_frame_data.shape[1]:
                # Get current value
                val = self.current_frame_data[y, x]

                # Update annotation
                self.cursor_annotation_box.xy = (x, y)
                self.cursor_annotation_text.set_text(f'({x}, {y})\n{val:.5f}')
                self.cursor_annotation_box.set_visible(True)
            
                self.fig.canvas.draw_idle()
            
        def update_frame(self, frame_idx: float):
            """Update the displayed frame."""
            # Extract frame data
            self.current_frame = int(frame_idx)
            self.current_frame_data = self.tdata[..., self.current_frame].squeeze()

            # Update image data
            self.frame_img.set_data(self.current_frame_data)

            # Get the colorbar limits according to min/max of the current frame
            vmin = round(self.current_frame_data.min(), 8)
            vmax = round(self.current_frame_data.max(), 8)

            # Directly set the image norm, because set_clim does call color sanitazion inside, which can lead to wrong updates
            self.frame_img.norm = Normalize(vmin, vmax)
                    
            # Redraw points on the new frame
            for idx, (x, y) in enumerate(self.selected_points):
                self.frame_ax.plot(x, y, 'x', color=self.colors[idx], markersize=10)
                
            # Redraw
            self.fig.canvas.draw_idle()
            
        def on_click(self, event):
            """Handle click events on the frame plot."""
            if event.inaxes != self.frame_ax:
                return
                
            if len(self.selected_points) >= 4:
                print("Maximum number of points (4) reached. Clear points to add more.")
                return
                
            x, y = int(event.xdata), int(event.ydata)
            if x < 0 or y < 0 or x >= self.tdata.shape[1] or y >= self.tdata.shape[0]:
                return
                
            # Add point and plot profile
            color = self.colors[len(self.selected_points)]
            self.selected_points.append((x, y))
            
            # Plot point on frame
            self.frame_ax.plot(x, y, 'x', color=color, markersize=10)
            
            # Plot temperature profile
            profile = self.tdata[y, x, :]
            self.profile_ax.plot(self.domain_values, profile, color=color, 
                            label=f'Point ({x}, {y})')
            self.profile_ax.legend()
            
            self.fig.canvas.draw_idle()
            
        def clear_points(self, event):
            """Clear all selected points and profiles."""
            self.selected_points.clear()
            self.profile_ax.clear()
            
            # Reset profile plot
            self.profile_ax.set_xlabel(generate_label(self.domain_unit))
            self.profile_ax.set_ylabel(generate_label(self.data_unit))
            self.profile_ax.grid(True)

            # Remove points from frame plot
            for artist in self.frame_ax.lines:
                artist.remove()
            
            # Redraw frame without points
            self.frame_img.set_data(self.current_frame_data)
            
            self.fig.canvas.draw_idle()

    def show_frame(self, frame_number: int, option: str="", cmap: str = 'plasma'):
        """ Visualize a specific frame from the dataset with optional ground truth visualization and color mapping.

        Parameters:
            frame_number (int): The frame number to visualize.
            option (str): The visualization option to apply. Options are "ShowGroundTruth", "OverlayGroundTruth", or an empty string. 
            cmap (str): The color map to use for the visualization. Defaults to 'plasma'.
        """
        # Clear current figure
        plt.clf()

        # Extract the data from the container
        data = self.get_dataset('/Data/Tdata')
        groundtruth = self.get_dataset('/GroundTruth/DefectMask')
        
        # Get the frame to show
        data_to_show = data[:, :, frame_number]

        # Show the frame with the selected option
        match option:
            case "ShowGroundTruth":
                plt.subplot(1, 2, 1)
                image = plt.imshow(data_to_show, aspect='auto', cmap=cmap)
                plt.title(f'Frame Number: {frame_number}')
                
                plt.subplot(1, 2, 2)
                plt.imshow(groundtruth, aspect='auto')
                plt.title('Ground Truth')
            
            case "OverlayGroundTruth":
                image = plt.imshow(data_to_show, aspect='auto', cmap=cmap)  # Display the original data
                plt.title(f'Frame Number: {frame_number}')
                
                if groundtruth is not None:
                    # Prepare the overlay
                    binary_gt = groundtruth > 0  # Create a binary mask of the ground truth
                    rows, cols = groundtruth.shape
                    gt_overlay = torch.zeros((rows, cols, 3))  # Initialize an all-zero RGB image for the overlay
                    gt_overlay[:, :, 1] = binary_gt  # Apply green in the binary mask areas
                    
                    plt.imshow(gt_overlay, alpha=0.5)  # Display overlay with transparency

            # Default case, just show the frame data
            case _:  
                image = plt.imshow(data_to_show, aspect='auto', cmap=cmap)
                plt.title(f'Frame Number: {frame_number}')
        
        # Custom formatter for the colorbar to ensure that the colorbar ticks are displayed without offset
        formatter = ticker.ScalarFormatter(useMathText=False, useOffset=False)

        # Show the plot
        plt.colorbar(image, format=formatter)
        plt.show()

    def show_pixel_profile(self, pixel_pos_x: int, pixel_pos_y: int):
        """ Plot the profile of a specific pixel across the dataset's domain values with an option for data adjustment. 
        
        The X-axis of the plot is labeled according to the domaintype attribute, reflecting the dataset's domain (e.g., time, frequency). The Y-axis is generically labeled as 'Temperature in K'.

        Parameters:
            pixel_pos_x (int): The X-coordinate (column index) of the pixel. Must be within the dataset's second dimension range.
            pixel_pos_y (int): The Y-coordinate (row index) of the pixel. Must be within the dataset's first dimension range.
        """
        #Clear the current figure
        plt.clf()

        # Extract the data from the container
        data = self.get_dataset('/Data/Tdata')
        domainvalues = self.get_dataset('/MetaData/DomainValues')
        data_unit = self.get_unit('/Data/Tdata')
        domain_unit = self.get_unit('/MetaData/DomainValues')

        # Validate pixel positions to be within the data dimensions
        if pixel_pos_x < 0 or pixel_pos_y < 0 or pixel_pos_x >= data.shape[0] or pixel_pos_y >= data.shape[1]:
            raise ValueError("Pixel positions must be within the range of data dimensions.")
        
        # Extract temperature profile of the pixel
        temperature_profile = data[pixel_pos_y, pixel_pos_x, :]
        
        # Plot the temperature profile
        plt.plot(domainvalues, temperature_profile)
        plt.title(f'Profile of Pixel: {pixel_pos_x},{pixel_pos_y}')
        plt.xlabel(generate_label(domain_unit))
        plt.ylabel(generate_label(data_unit))   
        plt.show()

    def analyse_interactive(self):
        """Launch interactive analysis session for thermographic data visualization."""
        self.InteractiveAnalyzer(self)
        plt.show()