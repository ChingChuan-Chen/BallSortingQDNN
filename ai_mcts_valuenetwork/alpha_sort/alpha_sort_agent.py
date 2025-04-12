import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from alpha_sort.policy_network import PolicyNetwork
from alpha_sort.value_network import ValueNetwork
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

        # Policy network
        self.policy_net = PolicyNetwork(max_num_colors, num_colors).to(self.device)

        # Value network
        self.value_net = ValueNetwork(max_num_colors, num_colors).to(self.device)

        # Target networks
        self.target_policy_net = PolicyNetwork(max_num_colors, num_colors).to(self.device)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_policy_net.eval()

        self.target_value_net = ValueNetwork(max_num_colors, num_colors).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_value_net.eval()

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=1e-5)

        # Replay memory
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

        # Policy network loss
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_policy_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        policy_loss = F.mse_loss(q_values, target_q_values.detach())

        # Value network loss
        state_values = self.value_net(states).squeeze()
        next_state_values = self.target_value_net(next_states).squeeze()
        target_values = rewards + (1 - dones) * self.gamma * next_state_values
        value_loss = F.mse_loss(state_values, target_values.detach())

        # Backpropagation
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def update_target_networks(self):
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_value_net.load_state_dict(self.value_net.state_dict())

    def load_pretrained_weights(self, pretrained_model_path, map_location=None):
        # Load the pretrained state dictionary
        pretrained_state = torch.load(pretrained_model_path, map_location=map_location)
        policy_pretrained_state = pretrained_state["policy_net"]
        value_pretrained_state = pretrained_state["value_net"]

        # Load weights for the policy network
        policy_state = self.policy_net.state_dict()
        old_action_dim = policy_pretrained_state["fc2.weight"].shape[0]  # Output size of old policy network
        new_action_dim = policy_state["fc2.weight"].shape[0]  # Output size of new policy network

        # Transfer matching layers for the policy network
        for name, param in policy_pretrained_state.items():
            if param.shape == policy_state[name].shape:
                policy_state[name] = param  # Transfer matching weights

        # Reinitialize output layer if action_dim changed
        if old_action_dim != new_action_dim:
            torch.nn.init.xavier_uniform_(policy_state["fc2.weight"])
            policy_state["fc2.bias"].zero_()

        self.policy_net.load_state_dict(policy_state, strict=False)

        # Load weights for the value network
        value_state = self.value_net.state_dict()
        for name, param in value_pretrained_state.items():
            if param.shape == value_state[name].shape:
                value_state[name] = param  # Transfer matching weights

        self.value_net.load_state_dict(value_state, strict=False)
