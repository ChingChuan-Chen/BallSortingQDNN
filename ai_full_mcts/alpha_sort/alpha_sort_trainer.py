import time
import math
import random
import datetime
import logging
import itertools
from collections import deque, defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from alpha_sort.ball_sort_env import BallSortEnv
from alpha_sort.utils import save_model, hash_state
from alpha_sort.lib._state_utils import state_encode, state_decode


CurrentEnv = namedtuple('CurrentEnv', ['env', 'depth', 'reward', 'original_env_idx', 'action_idx'])


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

    def select_actions(self, mcts_depth, top_k, discount_factor=0.9):
        # Initialize the list of current environments and their corresponding rewards
        current_env_list = []
        for i, env in enumerate(self.envs):
            if env.is_done or len(env.valid_actions) == 0:
                continue
            current_env_list.append(CurrentEnv(env.clone(), 0, 0.0, i, 0))
        action_rewards = [defaultdict(float) for _ in range(self.num_envs)]

        probs_time = 0.0
        mcts_time = 0.0
        num_envs_in_depth = defaultdict(int)
        for dd in range(mcts_depth+1):
            num_envs_in_depth[dd] += len(current_env_list)

            state_inputs = []
            for idx, current_env in enumerate(current_env_list):
                state_inputs.append(self.state_encode_helper(current_env.env.state))

            assert len(state_inputs) == len(current_env_list), "State inputs and current environment list length mismatch."

            if not state_inputs:
                break

            # Batch policy network inference
            probs_start_time = time.time_ns()
            with torch.no_grad():
                logits = self.agent.policy_net(torch.tensor(np.array(state_inputs), dtype=torch.float32).to(self.agent.device))
                probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_time += time.time_ns() - probs_start_time

            # Define per-environment processing
            def process_env(current_env, valid_actions, valid_action_indices, prob):
                masked_probs = np.zeros_like(prob)
                masked_probs[valid_action_indices] = prob[valid_action_indices]
                if masked_probs.sum() == 0:
                    masked_probs[valid_action_indices] = 1.0 / len(valid_action_indices)
                else:
                    masked_probs /= masked_probs.sum()

                valid_action_indices_loop = []
                valid_actions_loop = []
                if top_k > 0 and len(valid_action_indices) > top_k:
                    top_k_indices = self.rng.choice(valid_action_indices, top_k, replace=False, p=masked_probs[valid_action_indices])
                    for action, action_idx in zip(valid_actions, valid_action_indices):
                        if action_idx in top_k_indices:
                            valid_action_indices_loop.append(action_idx)
                            valid_actions_loop.append(action)
                        else:
                            masked_probs[action_idx] = 0.0
                    masked_probs /= masked_probs.sum()
                else:
                    valid_action_indices_loop = valid_action_indices
                    valid_actions_loop = valid_actions

                next_env_list = []
                for action, action_idx in zip(valid_actions_loop, valid_action_indices_loop):
                    sim_env = current_env.env.clone()
                    src, dst = action
                    if current_env.env.is_valid_move(src, dst):
                        sim_env.move(src, dst)
                        reward = self.compute_action_reward(sim_env, src, dst)
                    else:
                        reward = -150.0 / self.hard_factor

                    next_env_action_idx = action_idx if current_env.depth == 0 else current_env.action_idx
                    next_env_list.append(
                        CurrentEnv(
                            sim_env, current_env.depth + 1, reward * masked_probs[action_idx], current_env.original_env_idx, next_env_action_idx
                        )
                    )
                return next_env_list

            # Run parallel environment processing
            mcts_start_time = time.time_ns()
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_env,
                        current_env,
                        current_env.env.valid_actions,
                        np.array([self.action_to_index[action] for action in current_env.env.valid_actions]),
                        probs[idx]
                    )
                    for idx, current_env in enumerate(current_env_list)
                ]
                next_env_list = list(itertools.chain.from_iterable(future.result() for future in futures))
            mcts_time += time.time_ns() - mcts_start_time

            for idx, next_env in enumerate(next_env_list):
                action_rewards[next_env.original_env_idx][next_env.action_idx] += next_env.reward * discount_factor ** next_env.depth

            current_env_list = []
            if dd < mcts_depth:
                for idx, current_env in enumerate(next_env_list):
                    if current_env.env is None or len(current_env.env.valid_actions) == 0:
                        continue
                    if not current_env.env.is_done:
                        current_env_list.append(current_env)

        action_indices = []
        for env_idx, env in enumerate(self.envs):
            if env.is_done or len(action_rewards[env_idx]) == 0:
                action_indices.append(None)
            else:
                action_indices.append(max(action_rewards[env_idx].items(), key=lambda x: x[1])[0])

        return action_indices, probs_time, mcts_time, num_envs_in_depth

    def compute_action_reward(self, env, src: int, dst: int) -> float:
        reward = -35.0 / self.hard_factor
        if env.is_solved:
            return self.hard_factor * 10.0
        reward += self._compute_tube_rewards(env, src, dst)
        reward += self._compute_state_history_penalty(env)
        return reward

    def _compute_tube_rewards(self, env, src: int, dst: int) -> float:
        dst_top_color_streak = env.get_top_color_streak(dst)
        reward = 0.0
        if env.is_completed_tube(dst):
            reward += self.tube_capacity
        else:
            if env.state_key not in env.state_history and dst_top_color_streak >= self.tube_capacity - 2:
                if dst_top_color_streak == self.tube_capacity - 1:
                    reward += self.tube_capacity / 2.0
                elif dst_top_color_streak == self.tube_capacity - 2:
                    reward += self.tube_capacity / 4.0

        src_top_color_streak = env.get_top_color_streak(src)
        if env.state_key not in env.state_history and src_top_color_streak >= self.tube_capacity - 2:
            if src_top_color_streak == self.tube_capacity - 1:
                reward += self.tube_capacity / 2.0
            elif src_top_color_streak == self.tube_capacity - 2:
                reward += self.tube_capacity / 4.0
        return reward

    def _compute_state_history_penalty(self, env) -> float:
        penalty = 0.0
        if len(set(env.recent_state_keys)) < 6:
            penalty -= 0.1 * env.get_move_count() / self.hard_factor
        return penalty

    def _check_recursive_moves(self, env) -> bool:
        recursive_move_threshold = self.num_tubes * self.tube_capacity * 2
        reach_threshold = env.state_history[env.state_key] >= recursive_move_threshold
        count_reach_threshold = sum([1 for _, v in env.state_history.items() if v >= recursive_move_threshold])
        if reach_threshold and count_reach_threshold >= 3:
            return True
        return False

    def compute_rewards_and_dones(self, actions):
        states = np.array([env.state.copy() for env in self.envs])
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        step_dones = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if env.is_done:
                step_dones[i] = True
                continue

            if not step_dones[i] and env.get_move_count() >= self.max_step_count:
                step_dones[i] = True
                rewards[i] = -350.0 / self.hard_factor
                continue

            src, dst = actions[i]
            if not env.is_valid_move(src, dst):
                env.is_out_of_moves = True
                rewards[i] = -200.0 / self.hard_factor
                step_dones[i] = True
            else:
                env.move(src, dst)
                if self._check_recursive_moves(env):
                    env.is_in_recursive_moves = True
                    rewards[i] = -250.0 / self.hard_factor
                    step_dones[i] = True
                else:
                    rewards[i] += self.compute_action_reward(env, src, dst)
                    step_dones[i] = env.is_solved
        next_states = np.array([env.state.copy() for env in self.envs])
        return states, next_states, rewards, step_dones

    def store_transitions(self, states, action_indices, rewards, next_states, step_dones):
        for i, env in enumerate(self.envs):
            if env.is_done:
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

    def logging_progress(self, episode, step_count, total_rewards, cum_time_dict, num_envs_in_depth):
        solved_envs_count = np.sum([env.is_solved for env in self.envs])
        unsolved_envs_count = self.num_envs - solved_envs_count
        solve_rate = solved_envs_count / self.num_envs
        avg_solve_reward = np.sum([total_rewards[i] for i in range(self.num_envs) if self.envs[i].is_solved]) / solved_envs_count if solved_envs_count > 0 else 0.0
        avg_unsolved_reward = np.sum([total_rewards[i] for i in range(self.num_envs) if self.envs[i].is_solved == False]) / unsolved_envs_count if unsolved_envs_count > 0 else 0.0
        out_of_move_rate = np.sum([1 for env in self.envs if env.is_solved == False and env.is_out_of_moves]) / self.num_envs
        reach_recursive_moves_rate = np.sum([1 for env in self.envs if env.is_solved == False and env.is_in_recursive_moves]) / self.num_envs
        done_rate = np.sum([1 for env in self.envs if env.is_done]) / self.num_envs
        self.logger.info(
            f"Episode {episode: 03d} | Step {step_count: 03d} | Solve Rate: {solve_rate * 100.0: 6.2f}% "
            f"| Avg. Solve Rewards: {avg_solve_reward: 6.2f} | Avg. Unsolved Rewards: {avg_unsolved_reward: 6.2f} "
            f"| Reach Recursive Moves Rate: {reach_recursive_moves_rate * 100.0: 6.2f}% "
            f"| Out of Move Rate: {out_of_move_rate * 100.0: 6.2f}% | Done Rate: {done_rate * 100.0: 6.2f}%"
        )

        self.logger.info(
            f"Episode {episode: 03d} | Elapsed Time "
            f"| Torch Prob actions: {cum_time_dict["probs_time"] / 10**9:.2f} seconds "
            f"| MCTS actions: {cum_time_dict["mcts_time"] / 10**9:.2f} seconds "
            f"| Select actions: {cum_time_dict["select_actions"] / 10**9:.2f} seconds "
            f"| Compute rewards: {cum_time_dict["compute_rewards"] / 10**9:.2f} seconds "
            f"| Train step: {cum_time_dict["train_step"] / 10**9:.2f} seconds "
            f"| Update target network: {cum_time_dict["update_target_network"] / 10**9:.2f} seconds"
        )
        for depth, count in num_envs_in_depth.items():
            self.logger.info(f"Episode {episode: 03d} | Depth {depth} | Number of environments: {count / step_count:.2f}")

    def train(self, num_episodes, mcts_depth=3, top_k=7, discount_factor=0.9, train_steps_per_move=2):
        recent_results = deque(maxlen=10)

        for episode in range(num_episodes):
            # Reset all environments
            for env in self.envs:
                env.reset()
            total_rewards = np.zeros(self.num_envs, dtype=np.float32)
            step_count = 0

            start_time = datetime.datetime.now()
            self.logger.info(f"Episode {episode: 03d} | Start training for Episode {episode: 03d}")

            # Initialize variables for tracking time
            cum_time_dict = defaultdict(float)
            num_envs_in_depth = defaultdict(int)
            while step_count < self.max_step_count:
                if step_count > 0 and step_count % 10 == 0:
                    self.logging_progress(episode, step_count, total_rewards, cum_time_dict, num_envs_in_depth)

                # Get valid actions and select actions for each environment
                select_actions_start = time.time_ns()
                action_indices, probs_time, mcts_time, num_envs_in_depth_output = self.select_actions(mcts_depth, top_k, discount_factor)
                cum_time_dict["select_actions"] += time.time_ns() - select_actions_start
                cum_time_dict["probs_time"] += probs_time
                cum_time_dict["mcts_time"] += mcts_time
                for depth, count in num_envs_in_depth_output.items():
                    num_envs_in_depth[depth] += count

                # Map action indices to actions
                actions = [self.index_to_action[idx] if idx is not None else (-1, -1) for idx in action_indices]

                # Compute rewards and update environments
                compute_rewards_start = time.time_ns()
                states, next_states, rewards, step_dones = self.compute_rewards_and_dones(actions)
                cum_time_dict["compute_rewards"] += time.time_ns() - compute_rewards_start

                # Store transitions in replay memory
                self.store_transitions(states, action_indices, rewards, next_states, step_dones)
                for i, env in enumerate(self.envs):
                    if step_dones[i]:
                        env.is_done = True

                # Train the agent
                train_step_start = time.time_ns()
                for _ in range(train_steps_per_move):
                    self.agent.train_step()
                cum_time_dict["train_step"] += time.time_ns() - train_step_start

                # Update states and done flags
                total_rewards += rewards
                step_count += 1

                if np.sum([1 for env in self.envs if env.is_done]) == self.num_envs:
                    break

            for i, env in enumerate(self.envs):
                logging.info(
                    f"Env {i} | Move count: {env.get_move_count()} | Is solved: {env.is_solved} "
                    f"| Is Out of Moves: {env.is_out_of_moves} | Is In Recursive Moves: {env.is_in_recursive_moves} "
                    f"| State:\n{env.state}"
                )

            # Update the target network periodically
            update_target_network_start = time.time_ns()
            if episode % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            cum_time_dict["update_target_network"] = time.time_ns() - update_target_network_start

            # Log training progress
            solved_envs_count = np.sum([env.is_solved for env in self.envs])
            avg_solve_step = np.sum([env.get_move_count() for env in self.envs if env.is_solved]) / solved_envs_count if solved_envs_count > 0 else 0.0
            recent_results.append(solved_envs_count / self.num_envs)

            elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
            self.logging_progress(episode, step_count, total_rewards, cum_time_dict, num_envs_in_depth)
            self.logger.info(
                f"Episode {episode: 03d} | Last 10 Solve Rate: {np.mean(recent_results) * 100: 6.1f}% "
                f"| Avg. Solve Steps: {avg_solve_step: 6.1f} "
                f"| Total elapsed time for Episode {episode: 03d}: {elapsed_seconds} seconds"
            )

            # Save the model periodically
            if episode > 0 and episode % 5 == 0:
                model_save_path = save_model(self.agent, self.num_colors, self.tube_capacity, episode)
                logging.info(f"Model for the checkpoint at episode {episode: 03d} is saved to {model_save_path}.")
