from torch import nn
from torch.nn import functional as F
import logging

class DefectClassifier3DCNN(nn.Module):
    def __init__(self):
        super(DefectClassifier3DCNN, self).__init__()
        # Input: (1, 100, 100, 500)

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # First 3D Convolutional Layer
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce size by half
        # Output: (16, 50, 50, 250)
        
        # Second 3D Convolutional Layer
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce size by half again
        # Output: (32, 25, 25, 125)
        
        # Third 3D Convolutional Layer
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce size by half again
        # Output: (64, 12, 12, 62)
        
        # Fourth 3D Convolutional Layer
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce size by half
        # Output: (128, 6, 6, 62)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6 * 62, 512)  # Flatten and connect
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # Output layer (2 classes)

    def forward(self, x):
        # Apply the first conv layer + ReLU + pooling
        x = self.pool1(F.relu(self.conv1(x)))
        self.logger.info("After first conv layer: %s", x.shape)
        
        # Apply the second conv layer + ReLU + pooling
        x = self.pool2(F.relu(self.conv2(x)))
        # print("After second conv layer: ", x.shape)
        # Apply the third conv layer + ReLU + pooling
        x = self.pool3(F.relu(self.conv3(x)))
        # print("After third conv layer: ", x.shape)
        # Apply the fourth conv layer + ReLU + pooling
        x = self.pool4(F.relu(self.conv4(x)))
        # print("After fourth conv layer: ", x.shape)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 128 * 6 * 6 * 62)  # Flattening the 3D output to 1D
        # print("After flattening: ", x.shape)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        # print("After first FC layer: ", x.shape)
        x = F.relu(self.fc2(x))
        # print("After second FC layer: ", x.shape)   
        x = self.fc3(x)  # No activation for the final layer (logits will be handled by loss)
        # print("After third FC layer: ", x.shape)
        
        return x