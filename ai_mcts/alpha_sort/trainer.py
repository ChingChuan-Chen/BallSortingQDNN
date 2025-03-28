import time
import math
import random
import datetime
from collections import deque, defaultdict

import numpy as np
import torch
from joblib import Parallel, delayed
from alpha_sort.ball_sort_env import BallSortEnv
from alpha_sort.utils import save_model, hash_state
from alpha_sort.lib._state_utils import state_encode, state_decode

class AlphaSortTrainer:
    def __init__(self, envs, agent, max_num_colors, max_tube_capacity):
        self.envs = envs
        self.agent = agent
        self.max_num_colors = max_num_colors
        self.max_tube_capacity = max_tube_capacity

        # Initialize a random number generator
        self.rng = np.random.default_rng(random.randint(0, 2**31-1))

        # Environment properties
        self.num_colors = envs[0].num_colors
        self.tube_capacity = envs[0].tube_capacity
        self.num_empty_tubes = envs[0].num_empty_tubes
        self.num_tubes = self.num_colors + self.num_empty_tubes
        self.num_envs = len(envs)
        self.max_tubes = max_num_colors + self.num_empty_tubes
        self.max_step_count = 125 * envs[0].num_colors
        self.hard_factor = self.tube_capacity * self.num_colors ** 2

        # Precompute all possible actions
        self.all_possible_actions = [
            (i, j) for i in range(self.num_tubes) for j in range(self.num_tubes) if i != j
        ]
        self.action_to_index = {a: idx for idx, a in enumerate(self.all_possible_actions)}
        self.index_to_action = {idx: a for a, idx in self.action_to_index.items()}

    def select_actions(self, all_valid_actions, epsilon, dones, mcts_simulations=10, mcts_depth=3, top_k=5):
        action_indices = []
        for env_idx in range(self.num_envs):
            if dones[env_idx] or len(all_valid_actions[env_idx]) == 0:
                action_indices.append(None)
                continue

            # Get valid actions for the current environment
            valid_actions = all_valid_actions[env_idx]

            # Perform epsilon-greedy exploration
            if self.rng.random() < epsilon:
                # Perform DFS to explore mcts_depth branches
                best_action = self.mtcs_search(
                    state=self.envs[env_idx].state.copy(),
                    valid_actions=valid_actions,
                    depth=mcts_depth,
                    top_k=top_k
                )
            else:
                # Evaluate the state using the policy network
                encoded_state = np.array([state_encode(self.envs[env_idx].state, self.max_num_colors, self.num_empty_tubes, self.max_tube_capacity)])
                action_logits = self.agent.policy_net(torch.tensor(encoded_state, dtype=torch.float32).to(self.agent.device))
                action_logits = action_logits.detach().cpu().numpy().flatten()
                exp_logits = np.exp(action_logits - np.max(action_logits))

                valid_action_indices = [self.action_to_index[action] for action in valid_actions]
                if np.sum(exp_logits[valid_action_indices]) > 0.0:
                    masked_probs = exp_logits[valid_action_indices] / np.sum(exp_logits[valid_action_indices])
                    masked_probs /= masked_probs.sum()
                else:
                    masked_probs = np.ones(len(valid_action_indices)) / len(valid_action_indices)

                # Select the best action based on the policy network
                chosen_action_idx = self.rng.choice(valid_action_indices, p=masked_probs)
                best_action = self.index_to_action[chosen_action_idx]

            # Convert the selected action to its index
            action_indices.append(self.action_to_index[best_action])

        return action_indices

    def mtcs_search(self, state, valid_actions, depth, top_k, discount_factor=0.95):
        if len(valid_actions) == 0:
            # If no valid actions, return None
            return None

        # Initialize cumulative rewards for each action
        action_rewards = {action: 0.0 for action in valid_actions}

        # Simulate the environment for each action
        for action in valid_actions:
            simulated_env = BallSortEnv(
                num_colors=self.num_colors,
                tube_capacity=self.tube_capacity,
                num_empty_tubes=self.num_empty_tubes,
                state=state.copy()
            )
            src, dst = action
            if simulated_env.is_valid_move(src, dst):
                simulated_env.move(src, dst)
                immediate_reward = self.compute_action_reward(simulated_env, src, dst)
            else:
                immediate_reward = -150.0 / self.hard_factor  # Penalty for invalid moves

            # Initialize cumulative reward for this action
            cumulative_reward = immediate_reward

            # Iteratively explore deeper levels
            for d in range(1, depth + 1):
                # Get valid actions for the current state
                next_valid_actions = simulated_env.get_valid_actions()
                if len(next_valid_actions) == 0:
                    break

                if len(next_valid_actions) > top_k:
                    # Randomly select top-K actions
                    self.rng.shuffle(next_valid_actions)
                    next_valid_actions = next_valid_actions[:top_k]

                # Evaluate rewards for all valid actions
                next_action_rewards = []
                for next_action in next_valid_actions:
                    next_src, next_dst = next_action
                    if simulated_env.is_valid_move(next_src, next_dst):
                        simulated_env.move(next_src, next_dst)
                        reward = self.compute_action_reward(simulated_env, next_src, next_dst)
                        next_action_rewards.append((reward, next_action))
                        simulated_env.undo_move(next_src, next_dst)  # Undo the move to restore the state
                    else:
                        next_action_rewards.append((-150.0 / self.hard_factor, next_action))  # Penalty for invalid moves

                # Select the best action based on reward
                best_next_action = max(next_action_rewards, key=lambda x: x[0])[1]

                # Simulate the best action
                best_src, best_dst = best_next_action
                if simulated_env.is_valid_move(best_src, best_dst):
                    simulated_env.move(best_src, best_dst)
                    reward = self.compute_action_reward(simulated_env, best_src, best_dst)
                else:
                    reward = -150.0 / self.hard_factor  # Penalty for invalid moves

                # Add the reward to the cumulative reward
                cumulative_reward += reward * discount_factor ** d

            # Store the cumulative reward for the action
            action_rewards[action] = cumulative_reward

        # Select the action with the highest cumulative reward
        best_action = max(action_rewards.items(), key=lambda x: x[1])[0]

        return best_action

    def compute_action_reward(self, env, src: int, dst: int) -> float:
        reward = -25.0 / self.hard_factor

        if env.is_solved():
            return self.hard_factor * 10.0
        else:
            dst_top_color_streak = env.get_top_color_streak(dst)
            complete_tube_reward = self.tube_capacity * self.num_colors / self.num_tubes * 3.0
            streak_base_reward = complete_tube_reward / self.tube_capacity / 2.0
            if env.is_completed_tube(dst):
                reward += complete_tube_reward
            else:
                reward += streak_base_reward / (self.tube_capacity - dst_top_color_streak)

            src_top_color_streak = env.get_top_color_streak(src)
            if src_top_color_streak < self.tube_capacity:
                reward += streak_base_reward / (self.tube_capacity - src_top_color_streak)
            return reward

    def compute_rewards_and_dones(self, actions, dones):
        states = np.array([env.state.copy() for env in self.envs])
        next_states = np.zeros((self.num_envs, self.num_tubes, self.tube_capacity), dtype=np.int8)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        step_dones = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if dones[i]:
                next_states[i] = states[i]
                step_dones[i] = True
                continue

            if not step_dones[i] and env.get_move_count() >= self.max_step_count:
                step_dones[i] = True
                rewards[i] = -350.0 / self.hard_factor
                continue

            src, dst = actions[i]
            if not env.is_valid_move(src, dst):
                rewards[i] = -150.0 / self.hard_factor
                next_states[i] = states[i]
                step_dones[i] = False
            else:
                env.move(src, dst)
                next_states[i] = env.state.copy()
                rewards[i] = -25.0 / self.hard_factor
                rewards[i] += self.compute_action_reward(env, src, dst)
                step_dones[i] = env.is_solved()

        return states, next_states, rewards, step_dones

    def store_transitions(self, states, action_indices, rewards, next_states, step_dones, dones):
        for i in range(self.num_envs):
            if dones[i]:
                continue
            encoded_state = state_encode(states[i], self.max_num_colors, self.num_empty_tubes, self.max_tube_capacity)
            if action_indices[i] is None:
                self.agent.store_transition(encoded_state, 0, -10.0, encoded_state, True)
            else:
                encoded_next_state = state_encode(next_states[i], self.max_num_colors, self.num_empty_tubes, self.max_tube_capacity)
                self.agent.store_transition(
                    encoded_state,
                    action_indices[i],
                    rewards[i],
                    encoded_next_state,
                    step_dones[i]
                )

    def train(self, num_episodes, mcts_simulations=10, mcts_depth=3, top_k=5, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.975, train_steps_per_move=2):
        epsilon = epsilon_start
        recent_results = deque(maxlen=10)

        for episode in range(num_episodes):
            # Reset all environments
            for env in self.envs:
                env.reset()
            total_rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = np.zeros(self.num_envs, dtype=bool)
            step_count = 0

            start_time = datetime.datetime.now()
            print(f"🕒 Episode {episode: 03d} | Training for Episode {episode: 03d} started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            total_select_actions_time = 0.0
            total_compute_rewards_time = 0.0
            total_train_step_time = 0.0
            while not np.all(dones) and step_count < self.max_step_count:
                if step_count % 50 == 0:
                    avg_unsolved_reward = np.mean([total_rewards[i] for i in range(self.num_envs) if not self.envs[i].is_solved()])
                    print(
                        f"🕒 Episode {episode: 03d} | Step {step_count: 03d} | Solve Rate: {np.mean([env.is_solved() for env in self.envs]) * 100: 6.2f}% "
                        f"| Done Rate: {np.average(dones) * 100: 6.2f}% | Avg. Rewards: {np.mean(total_rewards): 6.2f} | Avg. Unsolved Rewards: {avg_unsolved_reward: 6.2f}"
                    )

                # Get valid actions and select actions for each environment
                select_actions_start = time.time_ns()
                all_valid_actions = [env.get_valid_actions() for env in self.envs]
                action_indices = self.select_actions(all_valid_actions, epsilon, dones, mcts_simulations, mcts_depth, top_k)
                total_select_actions_time += time.time_ns() - select_actions_start

                # Map action indices to actions
                actions = [self.index_to_action[idx] if idx is not None else (-1, -1) for idx in action_indices]

                # Compute rewards and update environments
                compute_rewards_start = time.time_ns()
                states, next_states, rewards, step_dones = self.compute_rewards_and_dones(actions, dones)
                total_compute_rewards_time += time.time_ns() - compute_rewards_start

                # Store transitions in replay memory
                self.store_transitions(states, action_indices, rewards, next_states, step_dones, dones)

                # Train the agent
                train_step_start = time.time_ns()
                for _ in range(train_steps_per_move):
                    self.agent.train_step()
                total_train_step_time += time.time_ns() - train_step_start

                # Update states and done flags
                dones |= step_dones
                total_rewards += rewards
                step_count += 1

            # Count puzzles that are not solved but have no valid actions
            unsolved_no_valid_actions_count = 0
            for i, env in enumerate(self.envs):
                if not env.is_solved() and len(env.get_valid_actions()) == 0:
                    unsolved_no_valid_actions_count += 1

            # Update the target network periodically
            update_target_network_start = time.time_ns()
            if episode % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            total_update_target_network_time = time.time_ns() - update_target_network_start

            # Decay epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_end)

            # Log training progress
            out_of_move_rate = unsolved_no_valid_actions_count / self.num_envs
            reach_max_steps_rate = np.mean([env.get_move_count() >= self.max_step_count for env in self.envs])
            solve_step_counts = [env.get_move_count() for env in self.envs if env.is_solved()]
            if len(solve_step_counts) > 0:
                avg_solve_steps = np.mean(solve_step_counts)
            else:
                avg_solve_steps = 0.0
            avg_rewards = np.mean(total_rewards)
            solve_rewards = [total_rewards[i] for i in range(self.num_envs) if self.envs[i].is_solved()]
            if len(solve_rewards) > 0:
                avg_solve_rewards = np.mean(solve_rewards)
            else:
                avg_solve_rewards = 0.0
            unsolved_rewards = [total_rewards[i] for i in range(self.num_envs) if not self.envs[i].is_solved()]
            if len(unsolved_rewards) > 0:
                avg_unsolved_rewards = np.mean(unsolved_rewards)
            else:
                avg_unsolved_rewards = 0.0
            solve_rate = np.mean([env.is_solved() for env in self.envs])
            recent_results.append(solve_rate)
            recent_solve_rate = np.mean(recent_results)

            end_time = datetime.datetime.now()
            elapsed_seconds = (end_time - start_time).total_seconds()
            print(
                f"🎯 Episode {episode: 03d} | Epsilon: {epsilon: .3f} | Solve Rate: {solve_rate * 100: 6.1f}%"
                f"| Out of Move Rate: {out_of_move_rate * 100: 6.1f}% | Reach Max Steps Rate: {reach_max_steps_rate * 100: 6.1f}% "
                f"| Last 10 Solve Rate: {recent_solve_rate * 100: 6.1f}%"
            )
            print(
                f"🎯 Episode {episode: 03d} | Avg. Rewards : {avg_rewards: 6.2f} "
                f"| Avg. Solve Rewards: {avg_solve_rewards: 6.2f} | Avg. Unsolved Rewards: {avg_unsolved_rewards: 6.2f} "
                f"| Avg. Solve Steps: {avg_solve_steps: 6.1f} "
            )
            print(
                f"🕒 Episode {episode: 03d} | Training for Episode {episode: 03d} ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"| Total elapsed time: {elapsed_seconds} seconds "
            )
            print(
                f"🕒 Episode {episode: 03d} Elapsed Time "
                f"| Select actions: {total_select_actions_time / 10**9:.2f} seconds"
                f"| Compute rewards: {total_compute_rewards_time / 10**9:.2f} seconds"
                f"| Train step: {total_train_step_time / 10**9:.2f} seconds"
                f"| Update target network: {total_update_target_network_time / 10**9:.2f} seconds"
            )

            # Save the model periodically
            if episode > 0 and episode % 5 == 0:
                save_model(self.agent, self.num_colors, self.tube_capacity, episode)
