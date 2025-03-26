import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from networks.resnet_dqn import ResNetDQN, ValueNetwork
from memory.replay_memory import ReplayMemory


class AlphaSortAgent:
    def __init__(self, num_colors, max_colors, max_tube_capacity, device, lr=1e-3, gamma=0.99, batch_size=256, target_update=100):
        self.max_colors = max_colors
        self.max_tube_capacity = max_tube_capacity
        self.num_colors = num_colors
        self.action_dim = (num_colors + 2) * (num_colors + 1)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.steps_done = 0

        self.device = device

        # Policy network for Q-value estimation
        self.policy_net = ResNetDQN(max_colors, num_colors, max_tube_capacity).to(self.device)

        # Value network for state evaluation
        self.value_net = ValueNetwork(max_colors, max_tube_capacity).to(self.device)

        # Target network for Q-value updates
        self.target_net = ResNetDQN(max_colors, num_colors, max_tube_capacity).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=1e-5)

        # Replay memory for experience replay
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

    def evaluate_state(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        value = self.value_net(state_tensor).squeeze(0).item()
        return value

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

        # Train value network
        predicted_values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(predicted_values, target_q_values.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_value_network(self):
        self.target_value_net.load_state_dict(self.value_net.state_dict())

    def load_pretrained_weights(self, pretrained_path, map_location=None):
        # Load the pretrained state dictionary
        pretrained_state = torch.load(pretrained_path, map_location=map_location)

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
        print(f"✅ Loaded pretrained weights for policy network from {pretrained_path}")

        # Transfer weights for the value network (no size change expected)
        value_state = self.value_net.state_dict()
        for name, param in pretrained_state.items():
            if name in value_state and param.shape == value_state[name].shape:
                value_state[name] = param  # Transfer matching weights

        self.value_net.load_state_dict(value_state, strict=False)
        print(f"✅ Loaded pretrained weights for value network from {pretrained_path}")
