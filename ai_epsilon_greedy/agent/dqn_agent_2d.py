import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class ResBlock(nn.Module):
    """Residual block with skip connections."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity mapping if input and output channels match, else projection
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


class ResNetDQN(nn.Module):
    """ResNet-based DQN using encoded 2D states."""
    def __init__(self, max_colors, num_colors, max_tube_capacity, hidden_dim=1024):
        super(ResNetDQN, self).__init__()
        self.action_dim = (num_colors + 2) * (num_colors + 1)
        self.hidden_dim = hidden_dim
        in_channels = max_colors + 1  # One-hot encoding channels

        # First Conv Layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual Blocks to Preserve Information
        self.res1 = ResBlock(64, 128)
        self.res2 = ResBlock(128, 256)

        self.dropout = nn.Dropout(0.2)

        # Dilated Convolutions for Global Context
        self.dilated_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)

        # Self-Attention Layer to Improve Spatial Encoding
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        # Adaptive Average Pooling to Fix Feature Map Size
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # Ensures fixed size input to FC layers

        # Fully Connected Layers
        self.fc1 = nn.Linear(4 * 4 * 256, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.dropout(x)

        # Apply Dilated Convolution for Global Context
        x = F.relu(self.dilated_conv(x))

        # Apply Self-Attention Mechanism
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)  # Reshape for attention
        x, _ = self.attention(x, x, x)  # Apply attention
        x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)  # Reshape back to CNN format

        # Adaptive Pooling to Fix Feature Map Size
        x = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN2DAgent:
    def __init__(self, num_colors, max_colors, max_tube_capacity, device, lr=1e-3, gamma=0.99, batch_size=64, target_update=100):
        self.max_colors = max_colors
        self.max_tube_capacity = max_tube_capacity
        self.num_colors = num_colors
        self.action_dim = (num_colors + 2) * (num_colors + 1)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.steps_done = 0

        self.device = device  # store device
        self.policy_net = ResNetDQN(max_colors, num_colors, max_tube_capacity).to(self.device)
        self.target_net = ResNetDQN(max_colors, num_colors, max_tube_capacity).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory()

    def select_action(self, state, valid_actions, epsilon):
        self.steps_done += 1

        if not valid_actions:
            return None

        if random.random() < epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.policy_net(state_tensor).squeeze(0).detach().cpu()

        # Mask out invalid actions
        mask = torch.full((self.action_dim,), float('-inf'))
        for a in valid_actions:
            mask[a] = 0
        masked_q_values = q_values + mask

        return int(torch.argmax(masked_q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) * 10 < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
