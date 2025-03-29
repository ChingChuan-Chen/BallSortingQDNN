import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from alpha_sort.policy_network import PolicyNetwork
from alpha_sort.replay_memory import ReplayMemory


class AlphaSortAgent:
    def __init__(self, num_colors, max_num_colors, max_tube_capacity, device, lr=1e-3, gamma=0.99, batch_size=256, target_update=100, relay_memory_size=10000):
        self.max_num_colors = max_num_colors
        self.max_tube_capacity = max_tube_capacity
        self.num_colors = num_colors
        self.action_dim = (num_colors + 2) * (num_colors + 1)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.relay_memory_size = relay_memory_size

        self.device = device

        # Policy network for Q-value estimation
        self.policy_net = PolicyNetwork(max_num_colors, num_colors, max_tube_capacity).to(self.device)

        # Target network for Q-value updates
        self.target_net = PolicyNetwork(max_num_colors, num_colors, max_tube_capacity).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)

        # Replay memory for experience replay
        self.memory = ReplayMemory(relay_memory_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Train policy network
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        policy_loss = F.mse_loss(q_values, target_q_values.detach())
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_pretrained_weights(self, pretrained_model_path, map_location=None):
        # Load the pretrained state dictionary
        pretrained_state = torch.load(pretrained_model_path, map_location=map_location)

        # Transfer weights for the policy network
        policy_state = self.policy_net.state_dict()
        old_action_dim = pretrained_state["fc2.weight"].shape[0]  # Output size of old model
        new_action_dim = policy_state["fc2.weight"].shape[0]  # Output size of new model

        # Transfer matching layers
        for name, param in pretrained_state.items():
            if name in policy_state and param.shape == policy_state[name].shape:
                policy_state[name] = param  # Transfer matching weights

        # Reinitialize output layer if action_dim changed
        if old_action_dim != new_action_dim:
            torch.nn.init.xavier_uniform_(policy_state["fc2.weight"])
            policy_state["fc2.bias"].zero_()

        self.policy_net.load_state_dict(policy_state, strict=False)
