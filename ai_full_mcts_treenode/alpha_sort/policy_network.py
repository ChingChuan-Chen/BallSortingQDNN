import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual block with skip connections."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity mapping if input and output channels differ
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.projection is not None:
            identity = self.projection(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity  # Residual connection
        return F.relu(out)


class PolicyNetwork(nn.Module):
    def __init__(self, max_colors, num_colors, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.action_dim = (num_colors + 2) * (num_colors + 1)
        self.hidden_dim = hidden_dim
        in_channels = max_colors + 1  # One-hot encoding channels

        # First Conv Layer
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Single Residual Block
        self.res1 = ResBlock(128, 256)

        # Adaptive Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)
