import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from alpha_sort.network import Network
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

        # network
        self.network = Network(max_num_colors, max_tube_capacity, num_colors).to(self.device)

        # Target networks
        self.target_network = Network(max_num_colors, max_tube_capacity, num_colors).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        # Optimizers
        self.network_optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-5)

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

        # Forward pass through the policy network
        current_values, current_logits = self.network(states)
        current_log_probs = F.log_softmax(current_logits, dim=1)
        action_log_probs = current_log_probs.gather(1, actions)

        # Compute value loss (MSE between predicted value and target value)
        with torch.no_grad():
            next_values, _ = self.target_network(next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)
        value_loss = F.mse_loss(current_values, target_values)

        # Compute policy loss (negative log likelihood of actions)
        policy_loss = -action_log_probs.mean()

        # Combine losses
        loss = value_loss + policy_loss

        # Back-Propagation
        self.network_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

        self.network_optimizer.step()

    def update_target_networks(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def load_pretrained_weights(self, pretrained_model_path, map_location=None):
        # Load the pretrained state dictionary
        pretrained_state = torch.load(pretrained_model_path, map_location=map_location)
        policy_pretrained_state = pretrained_state["model"]

        # Load weights for the policy network
        policy_state = self.network.state_dict()
        old_action_dim = policy_pretrained_state["policy_fc2.weight"].shape[0]  # Output size of old policy network
        new_action_dim = policy_state["policy_fc2.weight"].shape[0]  # Output size of new policy network

        # Transfer matching layers for the policy network
        for name, param in policy_pretrained_state.items():
            if param.shape == policy_state[name].shape:
                policy_state[name] = param  # Transfer matching weights

        # Reinitialize output layer if action_dim changed
        if old_action_dim != new_action_dim:
            torch.nn.init.xavier_uniform_(policy_state["policy_fc2.weight"])
            policy_state["policy_fc2.bias"].zero_()

        self.network.load_state_dict(policy_state, strict=False)
