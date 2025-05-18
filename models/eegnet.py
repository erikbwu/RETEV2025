import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    EEGNet: compact CNN with depthwise & separable convolutions:contentReference[oaicite:2]{index=2}.
    Expects input shape (batch, channels, time).
    """
    def __init__(self, n_channels=8, n_time=125, n_classes=2):
        super(EEGNet, self).__init__()
        # First temporal convolution (1 x 51 filter)
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        # Depthwise spatial convolution (n_channels x 1 filter, groups=16)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )
        # Separable convolution (1 x 15 filter)
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25)
        )
        # Compute output feature size after convolutions
        def conv_output_size(length, kernel_size, padding=0, stride=1):
            return (length + 2*padding - (kernel_size - 1) - 1)//stride + 1
        out_time = n_time
        out_time = conv_output_size(out_time, 51, padding=25)
        out_time = out_time // 4  # after first AvgPool
        out_time = conv_output_size(out_time, 15, padding=7)
        out_time = out_time // 8  # after second AvgPool
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * out_time, n_classes)
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.unsqueeze(1)          # reshape to (batch, 1, channels, time)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classifier(x)
        return x
