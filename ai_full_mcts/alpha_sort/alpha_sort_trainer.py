import time
import math
import random
import datetime
import logging
import itertools
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
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
        self.max_step_count = 75 * envs[0].num_colors
        self.min_dcount_state = self.num_tubes + 2
        self.trying_step_count = self.num_tubes * self.num_colors * 2
        self.hard_factor = self.tube_capacity * self.num_colors ** 2

        # Precompute all possible actions
        self.all_possible_actions = [
            (i, j) for i in range(self.num_tubes) for j in range(self.num_tubes) if i != j
        ]
        self.action_to_index = {a: idx for idx, a in enumerate(self.all_possible_actions)}
        self.index_to_action = {idx: a for a, idx in self.action_to_index.items()}

        # initialize logger
        self.logger = logging.getLogger()

    def state_encode_helper(self, state):
        # Encode the state using the provided function
        return state_encode(
            state,
            self.max_num_colors,
            self.num_empty_tubes,
            self.max_tube_capacity
        )

    def select_actions(self, all_valid_actions, mcts_depth):
        action_indices = []
        for env_idx, env in enumerate(self.envs):
            if env.done or len(all_valid_actions[env_idx]) == 0:
                action_indices.append(None)
                continue

            # Get valid actions for the current environment
            valid_actions = all_valid_actions[env_idx]

            # Use MCTS to select the best action
            best_action = self.mtcs_search(
                state=env.state.copy(),
                state_history=env.state_history.copy(),
                state_dcount_history=env.state_dcount_history.copy(),
                valid_actions=valid_actions,
                mcts_depth=mcts_depth
            )

            # Convert the selected action to its index
            action_indices.append(self.action_to_index[best_action])

        return action_indices

    def mtcs_search(self, state, state_history, state_dcount_history, valid_actions, mcts_depth, discount_factor=0.9):
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
                immediate_reward = self.compute_action_reward(simulated_env, src, dst, state_history, state_dcount_history)
            else:
                immediate_reward = -150.0 / self.hard_factor  # Penalty for invalid moves

            # Initialize cumulative reward for this action
            cumulative_reward = immediate_reward

            # Iteratively explore deeper levels
            for depth in range(1, mcts_depth + 1):
                # Get valid actions for the current state
                next_valid_actions = simulated_env.get_valid_actions()
                if len(next_valid_actions) == 0:
                    break

                # Use the policy network to estimate Q-values or probabilities
                with torch.no_grad():
                    action_logits_nn_output = self.agent.policy_net(torch.tensor(np.array([self.state_encode_helper(simulated_env.state)]), dtype=torch.float32).to(self.agent.device))
                    probs = F.softmax(action_logits_nn_output, dim=0).detach().cpu().numpy().flatten()

                # # Mask invalid actions
                masked_probs = np.zeros_like(probs)
                valid_action_indices = [self.action_to_index[action] for action in next_valid_actions]
                masked_probs[valid_action_indices] = probs[valid_action_indices]
                if masked_probs.sum() == 0 or np.sum(masked_probs >= 1e-2) < 3:
                    masked_probs[valid_action_indices] = 1.0 / len(valid_action_indices)
                else:
                    masked_probs /= masked_probs.sum()

                # Evaluate rewards for all valid actions
                next_action_rewards = []
                for next_action in next_valid_actions:
                    next_src, next_dst = next_action
                    if simulated_env.is_valid_move(next_src, next_dst):
                        simulated_env.move(next_src, next_dst)
                        reward = self.compute_action_reward(simulated_env, next_src, next_dst, state_history, state_dcount_history) * discount_factor ** depth
                        next_action_rewards.append((reward, next_action))
                        simulated_env.undo_move(next_src, next_dst)  # Undo the move to restore the state
                    else:
                        next_action_rewards.append((-150.0 / self.hard_factor, next_action))  # Penalty for invalid moves

                # Simulate the best action
                best_next_action = max(next_action_rewards, key=lambda x: x[0] * masked_probs[self.action_to_index[x[1]]])[1]

                # Simulate the selected action
                best_src, best_dst = best_next_action
                if simulated_env.is_valid_move(best_src, best_dst):
                    simulated_env.move(best_src, best_dst)
                    reward = self.compute_action_reward(simulated_env, best_src, best_dst, state_history, state_dcount_history) * discount_factor ** depth
                else:
                    reward = -150.0 / self.hard_factor  # Penalty for invalid moves

                # Add the reward to the cumulative reward
                cumulative_reward += reward

            # Store the cumulative reward for the action
            action_rewards[action] = cumulative_reward

        # Select the action with the highest cumulative reward
        best_action = max(action_rewards.items(), key=lambda x: x[1])[0]

        return best_action

    def compute_action_reward(self, env, src: int, dst: int, state_history=None, state_dcount_history=None) -> float:
        reward = -25.0 / self.hard_factor

        if env.is_solved():
            return self.hard_factor * 10.0

        reward += self._compute_tube_rewards(env, src, dst)
        # reward += self._compute_state_history_penalty(env, state_history, state_dcount_history)

        return reward

    def _compute_tube_rewards(self, env, src: int, dst: int) -> float:
        dst_top_color_streak = env.get_top_color_streak(dst)
        complete_tube_reward = self.tube_capacity * self.num_colors / self.num_tubes * 2.0
        streak_base_reward = complete_tube_reward / self.tube_capacity / 2.0
        reward = 0.0

        if env.is_completed_tube(dst):
            reward += complete_tube_reward
        else:
            reward += streak_base_reward / (self.tube_capacity - dst_top_color_streak)

        src_top_color_streak = env.get_top_color_streak(src)
        if src_top_color_streak < self.tube_capacity:
            reward += streak_base_reward / (self.tube_capacity - src_top_color_streak)

        return reward

    def _compute_state_history_penalty(self, env, state_history, state_dcount_history) -> float:
        penalty = 0.0

        if state_history is None:
            if env.state_history[env.state_key] >= self.num_tubes:
                penalty -= 0.3 * (env.state_history[env.state_key] - self.num_tubes + 1)
            if self._check_recursive_moves(env.state_key, env.state_history, env.state_dcount_history):
                penalty -= 150.0 / self.hard_factor
        else:
            if state_history[env.state_key] >= self.num_tubes:
                penalty -= 0.3 * (state_history[env.state_key] - self.num_tubes + 1)
            if self._check_recursive_moves(env.state_key, env.state_history, state_dcount_history):
                penalty -= 150.0 / self.hard_factor

        return penalty

    def _check_recursive_moves(self, state_key, state_history, state_dcount_history) -> bool:
        if len(state_dcount_history) < self.trying_step_count:
            return False
        state_count = len(state_dcount_history)
        last_k_counts = itertools.islice(state_dcount_history, state_count-self.trying_step_count, state_count)
        if state_key in state_history and len(set(last_k_counts)) == 1 and state_dcount_history[-1] >= self.min_dcount_state:
            return True
        return False

    def compute_rewards_and_dones(self, actions):
        states = np.array([env.state.copy() for env in self.envs])
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        step_dones = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if env.done:
                step_dones[i] = True
                continue

            if not step_dones[i] and env.get_move_count() >= self.max_step_count:
                step_dones[i] = True
                rewards[i] = -350.0 / self.hard_factor
                continue

            src, dst = actions[i]
            if not env.is_valid_move(src, dst):
                rewards[i] = -150.0 / self.hard_factor
                step_dones[i] = True
            else:
                env.move(src, dst)

                # if self._check_recursive_moves(env.state_key, env.state_history, env.state_dcount_history):
                #     env.is_in_recursive_moves = True
                #     rewards[i] = -200.0 / self.hard_factor
                #     step_dones[i] = True
                # else:
                rewards[i] += self.compute_action_reward(env, src, dst)
                step_dones[i] = env.is_solved()
        next_states = np.array([env.state.copy() for env in self.envs])
        return states, next_states, rewards, step_dones

    def store_transitions(self, states, action_indices, rewards, next_states, step_dones):
        for i, env in enumerate(self.envs):
            if env.done:
                continue
            encoded_state = self.state_encode_helper(states[i])
            if action_indices[i] is None:
                self.agent.store_transition(encoded_state, 0, -10.0, encoded_state, True)
            else:
                self.agent.store_transition(
                    encoded_state,
                    action_indices[i],
                    rewards[i],
                    self.state_encode_helper(next_states[i]),
                    step_dones[i]
                )

    def train(self, num_episodes, mcts_depth=4, train_steps_per_move=2):
        recent_results = deque(maxlen=10)

        for episode in range(num_episodes):
            # Reset all environments
            for env in self.envs:
                env.reset()
            total_rewards = np.zeros(self.num_envs, dtype=np.float32)
            step_count = 0

            start_time = datetime.datetime.now()
            self.logger.info(f"Episode {episode: 03d} | Start training for Episode {episode: 03d}")

            total_select_actions_time = 0.0
            total_compute_rewards_time = 0.0
            total_train_step_time = 0.0
            while step_count < self.max_step_count:
                if step_count % 30 == 0:
                    solved_envs_count = np.sum([env.is_solved() for env in self.envs])
                    unsolved_envs_count = np.sum([env.is_solved() == 0 for env in self.envs])
                    solve_rate = 1.0 - unsolved_envs_count / self.num_envs
                    avg_solve_reward = np.sum([total_rewards[i] for i in range(self.num_envs) if self.envs[i].is_solved() == 1]) / solved_envs_count if solved_envs_count > 0 else 0.0
                    avg_unsolved_reward = np.sum([total_rewards[i] for i in range(self.num_envs) if self.envs[i].is_solved() == 0]) / unsolved_envs_count if unsolved_envs_count > 0 else 0.0
                    out_of_move_rate = np.sum([1 for env in self.envs if env.is_solved() == 0 and env.out_of_moves]) / self.num_envs
                    reach_recursive_moves_rate = np.sum([1 for env in self.envs if env.is_solved() == 0 and env.is_in_recursive_moves]) / self.num_envs
                    done_rate = np.sum([1 for env in self.envs if env.done]) / self.num_envs
                    self.logger.info(
                        f"Episode {episode: 03d} | Step {step_count: 03d} | Solve Rate: {solve_rate * 100.0: 6.2f}% "
                        f"| Avg. Solve Rewards: {avg_solve_reward: 6.2f} | Avg. Unsolved Rewards: {avg_unsolved_reward: 6.2f} "
                        f"| Reach Recursive Moves Rate: {reach_recursive_moves_rate * 100.0: 6.2f}% "
                        f"| Out of Move Rate: {out_of_move_rate * 100.0: 6.2f}% | Done Rate: {done_rate * 100.0: 6.2f}%"
                    )

                # Get valid actions and select actions for each environment
                select_actions_start = time.time_ns()
                all_valid_actions = [env.get_valid_actions() for env in self.envs]
                action_indices = self.select_actions(all_valid_actions, mcts_depth)
                total_select_actions_time += time.time_ns() - select_actions_start

                # Map action indices to actions
                actions = [self.index_to_action[idx] if idx is not None else (-1, -1) for idx in action_indices]

                # Compute rewards and update environments
                compute_rewards_start = time.time_ns()
                states, next_states, rewards, step_dones = self.compute_rewards_and_dones(actions)
                total_compute_rewards_time += time.time_ns() - compute_rewards_start

                # Store transitions in replay memory
                self.store_transitions(states, action_indices, rewards, next_states, step_dones)
                for i, env in enumerate(self.envs):
                    if step_dones[i]:
                        env.done = True

                # Train the agent
                train_step_start = time.time_ns()
                for _ in range(train_steps_per_move):
                    self.agent.train_step()
                total_train_step_time += time.time_ns() - train_step_start

                # Update states and done flags
                total_rewards += rewards
                step_count += 1

                if np.sum([1 for env in self.envs if env.done]) == self.num_envs:
                    break

            # Update the target network periodically
            update_target_network_start = time.time_ns()
            if episode % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            total_update_target_network_time = time.time_ns() - update_target_network_start

            # Log training progress
            out_of_move_rate = np.sum([1 for env in self.envs if env.is_solved() == 0 and env.out_of_moves]) / self.num_envs
            reach_max_steps_rate = np.mean([env.get_move_count() >= self.max_step_count for env in self.envs])
            reach_recursive_moves_rate = np.sum([1 for env in self.envs if env.is_solved() == 0 and env.is_in_recursive_moves]) / self.num_envs
            solved_envs_count = np.sum([env.is_solved() for env in self.envs])
            unsolved_envs_count = np.sum([env.is_solved() == 0 for env in self.envs])
            avg_solve_step = np.sum([env.get_move_count() for env in self.envs if env.is_solved()]) / solved_envs_count if solved_envs_count > 0 else 0.0
            avg_solve_reward = np.sum([total_rewards[i] for i in range(self.num_envs) if self.envs[i].is_solved() == 1]) / solved_envs_count if solved_envs_count > 0 else 0.0
            avg_unsolve_reward = np.sum([total_rewards[i] for i in range(self.num_envs) if self.envs[i].is_solved() == 0]) / unsolved_envs_count if unsolved_envs_count > 0 else 0.0
            solve_rate = np.mean([env.is_solved() for env in self.envs])
            recent_results.append(solve_rate)
            recent_solve_rate = np.mean(recent_results)

            elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Episode {episode: 03d} | Solve Rate: {solve_rate * 100: 6.1f}%"
                f"| Out of Move Rate: {out_of_move_rate * 100: 6.1f}% | Reach Max Steps Rate: {reach_max_steps_rate * 100: 6.1f}% "
                f"| Reach Recursive Moves Rate: {reach_recursive_moves_rate * 100: 6.1f}% "
                f"| Last 10 Solve Rate: {recent_solve_rate * 100: 6.1f}%"
            )
            self.logger.info(
                f"Episode {episode: 03d} | Avg. Solve Rewards: {avg_solve_reward: 6.2f} | Avg. Unsolved Rewards: {avg_unsolve_reward: 6.2f} "
                f"| Avg. Solve Steps: {avg_solve_step: 6.1f} "
            )
            self.logger.info(
                f"Episode {episode: 03d} | Total elapsed time for Episode {episode: 03d}: {elapsed_seconds} seconds "
            )
            self.logger.info(
                f"Episode {episode: 03d} | Elapsed Time "
                f"| Select actions: {total_select_actions_time / 10**9:.2f} seconds"
                f"| Compute rewards: {total_compute_rewards_time / 10**9:.2f} seconds"
                f"| Train step: {total_train_step_time / 10**9:.2f} seconds"
                f"| Update target network: {total_update_target_network_time / 10**9:.2f} seconds"
            )

            # Save the model periodically
            if episode > 0 and episode % 5 == 0:
                model_save_path = save_model(self.agent, self.num_colors, self.tube_capacity, episode)
                logging.info(f"Model for the checkpoint at episode {episode: 03d} is saved to {model_save_path}.")
