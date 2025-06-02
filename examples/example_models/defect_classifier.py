from torch import nn
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)
    
class UNetClassifier(nn.Module):
    """Defect classifier with UNet-like encoder and classification head."""
    def __init__(self, time_channels: int):
        super().__init__()
        
        # Use your successful UNet encoder
        self.enc1 = DoubleConv(time_channels, 64)
        self.enc2 = DoubleConv(64, 128) 
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classification head (instead of decoder)
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),  # Global pool only at the end
            nn.Flatten(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Classify
        return self.classifier(enc4)
