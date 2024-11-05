import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from .group_ops import GroupOps
from .dataset_ops import DatasetOps
from .attribute_ops import AttributeOps
from ..units import generate_label

class VisualizationOps(GroupOps, DatasetOps, AttributeOps):
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