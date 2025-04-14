import torch
import torch.nn as nn
import torch.nn.functional as F
from alpha_sort.resnet_block import ResBlock


class ValueNetwork(nn.Module):
    def __init__(self, max_colors, hidden_dim=1024):
        super(ValueNetwork, self).__init__()
        in_channels = max_colors + 1  # One-hot encoding channels

        # First Conv Layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        # Single Residual Block
        self.res1 = ResBlock(64, 128)

        # Adaptive Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Single output for state value

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)
