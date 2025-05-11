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


class Network(nn.Module):
    def __init__(self, max_num_colors, max_tube_capacity, num_colors, num_res_blocks=9, value_output_channels=2, policy_output_channels=4):
        super(Network, self).__init__()
        base_channels = 256
        hidden_channels = 512
        self.action_dim = (num_colors + 2) * (num_colors + 1)
        in_channels = max_num_colors + 1  # One-hot encoding channels
        max_num_tubes = max_num_colors + 2  # Including empty tubes
        self.conv_output_channels = max_num_tubes * max_tube_capacity

        # First Conv Layer
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # residual blocks
        blocks = [ResBlock(base_channels, base_channels) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Policy network head
        self.policy_conv = nn.Conv2d(base_channels, policy_output_channels, kernel_size=1, stride=1, padding=0)
        self.policy_bn = nn.BatchNorm2d(policy_output_channels)
        self.policy_fc1 = nn.Linear(self.conv_output_channels * policy_output_channels, hidden_channels)
        self.policy_fc2 = nn.Linear(hidden_channels, self.action_dim)

        # Value network head
        self.value_conv = nn.Conv2d(base_channels, value_output_channels, kernel_size=1, stride=1, padding=0)
        self.value_bn = nn.BatchNorm2d(value_output_channels)
        self.value_fc1 = nn.Linear(self.conv_output_channels * value_output_channels, hidden_channels)
        self.value_fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        # value network head
        v = self.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value = torch.tanh(v).squeeze(1)

        # policy network head
        p = self.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.relu(self.policy_fc1(p))
        p = self.policy_fc2(p)
        return value, p
