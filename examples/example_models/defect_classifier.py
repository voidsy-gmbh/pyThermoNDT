import torch
from torch import nn
from torch.nn import functional as F
import logging

class DefectClassifier3DCNN(nn.Module):
    def __init__(self, time_dim=32):
        super().__init__()
        
        # Add stronger regularization
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout3d(0.1),  
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(0.1),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        # Add dropout to classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits
