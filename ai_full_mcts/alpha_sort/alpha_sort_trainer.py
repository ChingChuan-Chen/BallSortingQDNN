import time
import random
import datetime
import logging
import itertools
from collections import deque, defaultdict, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from alpha_sort.lib.ball_sort_env import BallSortEnv
from alpha_sort.utils import save_model, hash_state

CurrentEnv = namedtuple('CurrentEnv', ['env', 'depth', 'reward', 'original_env_idx', 'action_idx'])


class AlphaSortTrainer:
    def __init__(self, envs, agent, max_num_colors, max_tube_capacity):
        self.envs = envs
        self.agent = agent
        self.max_num_colors = max_num_colors
        self.max_tube_capacity = max_tube_capacity

        # Initialize a random number generator
        self.rng = np.random.default_rng(random.randint(0, 2**31 - 1))

        # Environment properties
        self.num_colors = envs[0].get_num_colors()
        self.tube_capacity = envs[0].get_tube_capacity()
        self.num_empty_tubes = envs[0].get_num_empty_tubes()
        self.num_tubes = self.num_colors + self.num_empty_tubes
        self.num_envs = len(envs)
        self.max_tubes = max_num_colors + self.num_empty_tubes
        self.max_step_count = 75 * envs[0].get_num_colors()
        self.recursive_move_threshold = self.num_tubes * self.tube_capacity * 2
        self.hard_factor = self.tube_capacity * self.num_colors ** 2

        # Precompute all possible actions
        self.all_possible_actions = [
            (i, j) for i in range(self.num_tubes) for j in range(self.num_tubes) if i != j
        ]
        self.action_to_index = {a: idx for idx, a in enumerate(self.all_possible_actions)}
        self.index_to_action = {idx: a for a, idx in self.action_to_index.items()}

        # initialize logger
        self.logger = logging.getLogger()

    def state_encode_helper(self, env):
        # Encode the state using the provided function
        return env.get_encoded_state(self.max_num_colors, self.num_empty_tubes, self.max_tube_capacity)

    def select_actions(self, mcts_depth, top_k, discount_factor=0.9):
        # Initialize the list of current environments and their corresponding rewards
        current_env_list = []
        for i, env in enumerate(self.envs):
            if env.get_is_done() or env.have_valid_moves() == False:
                continue
            current_env_list.append(CurrentEnv(env.clone(), 0, 0.0, i, 0))
        action_rewards = [defaultdict(float) for _ in range(self.num_envs)]

        probs_time = 0.0
        mcts_time = 0.0
        num_envs_in_depth = defaultdict(int)
        for dd in range(mcts_depth + 1):
            num_envs_in_depth[dd] += len(current_env_list)

            state_inputs = []
            for idx, current_env in enumerate(current_env_list):
                state_inputs.append(self.state_encode_helper(current_env.env))

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
                    nonzero_count = np.sum(masked_probs > 1e-2)
                    if nonzero_count < top_k:
                        logging.warning(f"Warning: only {nonzero_count} valid actions found, using all of them.")
                        top_k_indices = valid_action_indices[masked_probs[valid_action_indices] > 0.01]
                    else:
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
                    sim_env.move(src, dst)
                    reward = self.compute_action_reward(sim_env, src, dst)

                    next_env_action_idx = action_idx if current_env.depth == 0 else current_env.action_idx
                    next_env_list.append(
                        CurrentEnv(
                            sim_env, current_env.depth + 1, reward * masked_probs[action_idx], current_env.original_env_idx, next_env_action_idx
                        )
                    )
                return next_env_list

            # Run parallel environment processing
            mcts_start_time = time.time_ns()
            next_env_list = list(itertools.chain.from_iterable(
                process_env(
                    current_env,
                    current_env.env.get_valid_moves(),
                    np.array([self.action_to_index[action] for action in current_env.env.get_valid_moves()]),
                    probs[idx]
                )
                for idx, current_env in enumerate(current_env_list)
            ))
            mcts_time += time.time_ns() - mcts_start_time

            for idx, next_env in enumerate(next_env_list):
                action_rewards[next_env.original_env_idx][next_env.action_idx] += next_env.reward * discount_factor ** next_env.depth

            current_env_list = []
            if dd < mcts_depth:
                for idx, cur_env in enumerate(next_env_list):
                    if not cur_env.env.have_valid_moves():
                        continue
                    if not cur_env.env.get_is_done() and not cur_env.env.get_is_solved():
                        current_env_list.append(current_env)

        action_indices = []
        for env_idx, env in enumerate(self.envs):
            if env.get_is_done() or len(action_rewards[env_idx]) == 0:
                action_indices.append(None)
            else:
                action_indices.append(max(action_rewards[env_idx].items(), key=lambda x: x[1])[0])

        return action_indices, probs_time, mcts_time, num_envs_in_depth

    def compute_action_reward(self, env, src: int, dst: int) -> float:
        reward = -35.0 / self.hard_factor
        if env.get_is_solved():
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
            if not env.is_recent_state_key() and dst_top_color_streak >= self.tube_capacity - 2:
                if dst_top_color_streak == self.tube_capacity - 1:
                    reward += self.tube_capacity / 2.0
                elif dst_top_color_streak == self.tube_capacity - 2:
                    reward += self.tube_capacity / 4.0

        src_top_color_streak = env.get_top_color_streak(src)
        if not env.is_recent_state_key() and src_top_color_streak >= self.tube_capacity - 2:
            if src_top_color_streak == self.tube_capacity - 1:
                reward += self.tube_capacity / 2.0
            elif src_top_color_streak == self.tube_capacity - 2:
                reward += self.tube_capacity / 4.0
        return reward

    def _compute_state_history_penalty(self, env) -> float:
        penalty = 0.0
        if env.is_recent_state_key():
            penalty -= env.get_move_count() / self.hard_factor
        if env.get_current_state_count() > self.num_tubes:
            penalty -= -600.0 / self.hard_factor * env.get_current_state_count() / self.recursive_move_threshold
        if env.get_last_state_count() > self.num_tubes * 2:
            penalty -= -600.0 / self.hard_factor * env.get_last_state_count() / self.recursive_move_threshold
        return penalty

    def check_is_recursive_move(self, env) -> bool:
        if env.get_move_count() < self.recursive_move_threshold * 3:
            return False
        if env.get_current_state_count() > self.recursive_move_threshold and env.get_last_state_count() > self.recursive_move_threshold:
            return True
        return False

    def compute_rewards_and_dones(self, actions):
        encoded_states = np.array([self.state_encode_helper(env) for env in self.envs])
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        step_dones = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if env.get_is_done():
                step_dones[i] = True
                continue

            if not step_dones[i] and env.get_move_count() >= self.max_step_count:
                step_dones[i] = True
                rewards[i] = -350.0 / self.hard_factor
                continue

            src, dst = actions[i]
            env.move(src, dst)
            if self.check_is_recursive_move(env):
                env.set_is_recursive_move(True)
                rewards[i] = -400.0 / self.hard_factor
                step_dones[i] = True
            else:
                rewards[i] += self.compute_action_reward(env, src, dst)
                step_dones[i] = env.get_is_solved()
        encoded_next_states = np.array([self.state_encode_helper(env) for env in self.envs])
        return encoded_states, encoded_next_states, rewards, step_dones

    def store_transitions(self, encoded_states, action_indices, rewards, encoded_next_states, step_dones):
        for i, env in enumerate(self.envs):
            if env.get_is_done():
                continue
            if action_indices[i] is None:
                self.agent.store_transition(encoded_states[i], 0, -10.0, encoded_states[i], True)
            else:
                self.agent.store_transition(
                    encoded_states[i],
                    action_indices[i],
                    rewards[i],
                    encoded_next_states[i],
                    step_dones[i]
                )

    def logging_progress(self, episode, step_count, total_rewards, cum_time_dict, num_envs_in_depth):
        solved_envs_count = np.sum([env.get_is_solved() for env in self.envs])
        unsolved_envs_count = self.num_envs - solved_envs_count
        solve_rate = solved_envs_count / self.num_envs
        avg_solve_reward = np.sum([total_rewards[i] for i in range(self.num_envs) if self.envs[i].get_is_solved()]) / solved_envs_count if solved_envs_count > 0 else 0.0
        avg_unsolved_reward = np.sum([total_rewards[i] for i in range(self.num_envs) if not self.envs[i].get_is_solved()]) / unsolved_envs_count if unsolved_envs_count > 0 else 0.0
        out_of_move_rate = np.sum([1 for env in self.envs if not env.get_is_solved() and not env.have_valid_moves()]) / self.num_envs
        reach_recursive_moves_rate = np.sum([1 for env in self.envs if not env.get_is_solved() and env.get_is_recursive_move()]) / self.num_envs
        done_rate = np.sum([1 for env in self.envs if env.get_is_done()]) / self.num_envs
        self.logger.info(
            f"Episode {episode: 03d} | Step {step_count: 03d} | Solve Rate: {solve_rate * 100.0: 6.2f}% "
            f"| Avg. Solve Rewards: {avg_solve_reward: 6.2f} | Avg. Unsolved Rewards: {avg_unsolved_reward: 6.2f} "
            f"| Reach Recursive Moves Rate: {reach_recursive_moves_rate * 100.0: 6.2f}% "
            f"| Out of Move Rate: {out_of_move_rate * 100.0: 6.2f}% | Done Rate: {done_rate * 100.0: 6.2f}%"
        )

        self.logger.info(
            f"Episode {episode: 03d} | Elapsed Time "
            f"| Torch Prob actions: {cum_time_dict['probs_time'] / 10**9:.2f} seconds "
            f"| MCTS actions: {cum_time_dict['mcts_time'] / 10**9:.2f} seconds "
            f"| Select actions: {cum_time_dict['select_actions'] / 10**9:.2f} seconds "
            f"| Compute rewards: {cum_time_dict['compute_rewards'] / 10**9:.2f} seconds "
            f"| Train step: {cum_time_dict['train_step'] / 10**9:.2f} seconds "
            f"| Update target network: {cum_time_dict['update_target_network'] / 10**9:.2f} seconds"
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
                encoded_states, encoded_next_states, rewards, step_dones = self.compute_rewards_and_dones(actions)
                cum_time_dict["compute_rewards"] += time.time_ns() - compute_rewards_start

                # Store transitions in replay memory
                self.store_transitions(encoded_states, action_indices, rewards, encoded_next_states, step_dones)
                for i, env in enumerate(self.envs):
                    if step_dones[i]:
                        env.set_is_done(True)

                # Train the agent
                train_step_start = time.time_ns()
                for _ in range(train_steps_per_move):
                    self.agent.train_step()
                cum_time_dict["train_step"] += time.time_ns() - train_step_start

                # Update states and done flags
                total_rewards += rewards
                step_count += 1

                if np.sum([1 for env in self.envs if env.get_is_done()]) == self.num_envs:
                    break

            for i, env in enumerate(self.envs):
                logging.info(
                    f"Env {i} | Move count: {env.get_move_count()} | Is solved: {env.get_is_solved()} "
                    f"| Is Out of Moves: {not env.have_valid_moves()} | Is In Recursive Moves: {env.get_is_recursive_move()} "
                    f"| State:\n{env.get_state()}"
                )

            # Update the target network periodically
            update_target_network_start = time.time_ns()
            if episode % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            cum_time_dict["update_target_network"] = time.time_ns() - update_target_network_start

            # Log training progress
            solved_envs_count = np.sum([env.get_is_solved() for env in self.envs])
            avg_solve_step = np.sum([env.get_move_count() for env in self.envs if env.get_is_solved()]) / solved_envs_count if solved_envs_count > 0 else 0.0
            recent_results.append(solved_envs_count / self.num_envs)

            elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
            self.logging_progress(episode, step_count, total_rewards, cum_time_dict, num_envs_in_depth)
            self.logger.info(
                f"Episode {episode: 03d} | Last 10 Solve Rate: {np.mean(recent_results) * 100: 6.1f}% "
                f"| Avg. Solve Steps: {avg_solve_step: 6.1f} "
                f"| Total elapsed time for Episode {episode: 03d}: {elapsed_seconds} seconds"
            )

            # Save the model periodically
            if episode > 0 and episode % 10 == 0:
                model_save_path = save_model(self.agent, self.num_colors, self.tube_capacity, episode)
                logging.info(f"Model for the checkpoint at episode {episode: 03d} is saved to {model_save_path}.")
