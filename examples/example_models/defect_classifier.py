from torch import nn
from torch.nn import functional as F
import logging

class DefectClassifier3DCNN(nn.Module):
    def __init__(self, time_dim=64):
        super(DefectClassifier3DCNN, self).__init__()
        # Input: (1, 100, 100, time_dim)
        self.time_dim = time_dim

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # First 3D Convolutional Layer
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce size by half
        # Output: (16, 50, 50, time_dim/2)
        
        # Second 3D Convolutional Layer
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce size by half again
        # Output: (32, 25, 25, time_dim/4)
        
        # Third 3D Convolutional Layer
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce size by half again
        # Output: (64, 12, 12, time_dim/8)
        
        # Fourth 3D Convolutional Layer
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce size by half
        # Output: (128, 6, 6, time_dim/16)

        # Calculate the flattened size dynamically
        time_after_conv = time_dim // 16  # After 4 pooling layers with stride 2
        flattened_size = 128 * 6 * 6 * time_after_conv
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # Output layer (2 classes)

    def forward(self, x):
        # Print shapes for debugging
        self.logger.info("Input shape: %s", x.shape)
        
        # Apply the first conv layer + ReLU + pooling
        x = self.pool1(F.relu(self.conv1(x)))
        self.logger.info("After first conv layer: %s", x.shape)
        
        # Apply the second conv layer + ReLU + pooling
        x = self.pool2(F.relu(self.conv2(x)))
        self.logger.info("After second conv layer: %s", x.shape)
        
        # Apply the third conv layer + ReLU + pooling
        x = self.pool3(F.relu(self.conv3(x)))
        self.logger.info("After third conv layer: %s", x.shape)
        
        # Apply the fourth conv layer + ReLU + pooling
        x = self.pool4(F.relu(self.conv4(x)))
        self.logger.info("After fourth conv layer: %s", x.shape)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # Flattening automatically
        self.logger.info("After flattening: %s", x.shape)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x