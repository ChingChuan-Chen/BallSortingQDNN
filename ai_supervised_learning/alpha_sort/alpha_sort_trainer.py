import time
import random
import datetime
import logging
from collections import deque, defaultdict, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from alpha_sort.lib.ball_sort_env import BallSortEnv
from alpha_sort.utils import save_model

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
        self.recursive_move_threshold = self.num_tubes * self.tube_capacity * 0.35
        self.hard_factor = self.num_colors / self.tube_capacity * (1.0 + self.num_empty_tubes / self.num_colors)

        # Precompute all possible actions
        self.all_possible_actions = [
            (i, j) for i in range(self.num_tubes) for j in range(self.num_tubes) if i != j
        ]
        self.action_to_index = {a: idx for idx, a in enumerate(self.all_possible_actions)}
        self.index_to_action = {idx: a for a, idx in self.action_to_index.items()}

        # initialize logger
        self.logger = logging.getLogger()
        self.tree = {}  # Initialize the shared tree dictionary
        self.encoded_state_cache = {}  # Cache for encoded states

    def state_encode_helper(self, env):
        if env.get_state_key() in self.encoded_state_cache:
            return self.encoded_state_cache[env.get_state_key()]
        # Encode the state using the provided function
        encoded_state = env.get_encoded_state(self.max_num_colors, self.num_empty_tubes, self.max_tube_capacity)
        return np.array(encoded_state, dtype=np.float32)

    def select_actions(self, mcts_depth, discount_factor=0.9, c_puct=2.0):
        def get_tree_node(state_hash):
            if state_hash not in self.tree:
                self.tree[state_hash] = {
                    "visit_counts": defaultdict(int),
                    "q_values": defaultdict(float),
                    "policy_probs": None,
                }
            return self.tree[state_hash]

        def calculate_puct_score(node, action_idx, total_visits, c_puct):
            visit_count = node["visit_counts"][action_idx]
            q_value = node["q_values"][action_idx]
            policy_prob = node["policy_probs"][action_idx]
            return q_value + c_puct * policy_prob * np.sqrt(total_visits) / (1 + visit_count)

        current_env_list = [
            CurrentEnv(env.clone(), 0, 0.0, i, 0)
            for i, env in enumerate(self.envs)
            if not env.get_is_done() and env.have_valid_moves()
        ]

        network_time = 0.0
        mcts_time = 0.0
        for dd in range(mcts_depth + 1):
            state_inputs = [self.state_encode_helper(env.env) for env in current_env_list]
            if not state_inputs:
                break

            # Batch inference for policy and value networks
            network_start_time = time.time_ns()
            with torch.no_grad():
                logits = self.agent.policy_net(torch.tensor(np.array(state_inputs), dtype=torch.float32).to(self.agent.device))
                probs = F.softmax(logits, dim=1).cpu().numpy()
                state_values = self.agent.value_net(torch.tensor(np.array(state_inputs), dtype=torch.float32).to(self.agent.device)).cpu().numpy()
            network_time += time.time_ns() - network_start_time

            mcts_start_time = time.time_ns()
            next_env_list = []
            for idx, current_env in enumerate(current_env_list):
                if current_env.env.get_is_done() or not current_env.env.have_valid_moves():
                    continue

                state_hash = current_env.env.get_state_key()
                node = get_tree_node(state_hash)

                # Initialize policy probabilities if not already set
                if node["policy_probs"] is None:
                    node["policy_probs"] = probs[idx]

                valid_actions = current_env.env.get_valid_moves()
                valid_action_indices = [self.action_to_index[action] for action in valid_actions]

                # Calculate PUCT scores for valid actions
                total_visits = sum(node["visit_counts"].values()) + 1
                puct_scores = {
                    action_idx: calculate_puct_score(node, action_idx, total_visits, c_puct)
                    for action_idx in valid_action_indices
                }

                # Select the action with the highest PUCT score
                best_action_idx = max(puct_scores, key=puct_scores.get)
                best_action = self.index_to_action[best_action_idx]

                # move the best action
                src, dst = best_action
                current_env.env.move(src, dst)
                reward = self.compute_action_reward(current_env.env, src, dst)

                # Update Q-values and visit counts
                node["visit_counts"][best_action_idx] += 1
                node["q_values"][best_action_idx] += (reward + discount_factor * state_values[idx] - node["q_values"][best_action_idx]) / node["visit_counts"][best_action_idx]

                # Add the next environment to the list
                next_env_list.append(
                    CurrentEnv(
                        current_env.env, current_env.depth + 1, reward, current_env.original_env_idx, best_action_idx
                    )
                )
            mcts_time += time.time_ns() - mcts_start_time

            current_env_list = next_env_list

        def stable_softmax(x):
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            prob = exp_x / np.sum(exp_x)
            prob[np.isnan(prob)] = 0.0
            prob = prob / np.sum(prob)  # Normalize to sum to 1
            return prob

        # Select the final action for each environment
        decide_next_action_start_time = time.time_ns()
        action_indices = []
        for env_idx, env in enumerate(self.envs):
            if env.get_is_done():
                action_indices.append(None)
                continue

            state_hash = current_env.env.get_state_key()
            node = get_tree_node(state_hash)

            if not node["visit_counts"]:
                # Fallback: Select a random valid action if no actions were explored
                valid_actions = env.get_valid_moves()
                if valid_actions:
                    random_action = random.choice(valid_actions)
                    action_indices.append(self.action_to_index[random_action])
                else:
                    action_indices.append(None)  # No valid actions available
            else:
                # Use a softmax over visit counts
                visit_counts = np.array([node["visit_counts"][action_idx] for action_idx in node["visit_counts"]])
                visit_probs = stable_softmax(visit_counts)
                selected_action_idx = self.rng.choice(list(node["visit_counts"].keys()), p=visit_probs)
                action_indices.append(selected_action_idx)
        decide_next_action_time = time.time_ns() - decide_next_action_start_time

        eplased_time_dict = {
            "network_time": network_time,
            "mcts_time": mcts_time,
            "decide_next_action": decide_next_action_time,
        }
        return action_indices, eplased_time_dict

    def compute_action_reward(self, env, src: int, dst: int) -> float:
        reward = -0.5 / self.hard_factor
        if env.get_is_solved():
            return 600.0 * self.hard_factor
        reward += self._compute_tube_rewards(env, src, dst)
        reward += self._compute_state_history_penalty(env)
        return reward

    def _compute_tube_rewards(self, env, src: int, dst: int) -> float:
        dst_top_color_streak = env.get_top_color_streak(dst)
        reward = 0.0
        if env.is_completed_tube(dst):
            reward += self.hard_factor * 2.0
        else:
            if dst_top_color_streak == self.tube_capacity - 1:
                reward += self.hard_factor
            elif dst_top_color_streak == self.tube_capacity - 2:
                reward += self.hard_factor / 3.0

        src_top_color_streak = env.get_top_color_streak(src)
        if src_top_color_streak == self.tube_capacity - 1:
            reward += self.hard_factor
        elif src_top_color_streak == self.tube_capacity - 2:
            reward += self.hard_factor / 3.0
        return reward

    def _compute_state_history_penalty(self, env) -> float:
        penalty = 0.0
        step_factor = 1.0 + env.get_move_count() / self.max_step_count * 2.0
        recent_count = env.get_recent_count()
        if recent_count > 1:
            penalty -= 5.0 / self.hard_factor * (1 + recent_count / env.get_move_count()) * step_factor
        if env.get_current_state_count() > self.num_colors:
            penalty -= 25.0 / self.hard_factor * env.get_current_state_count() / self.recursive_move_threshold * step_factor
        if env.get_last_state_count() > self.num_colors:
            penalty -= 25.0 / self.hard_factor * env.get_last_state_count() / self.recursive_move_threshold * step_factor
        return penalty

    def check_is_recursive_move(self, env) -> bool:
        if env.get_move_count() < self.recursive_move_threshold * 2:
            return False
        if env.get_current_state_count() > self.recursive_move_threshold and env.get_last_state_count() > self.recursive_move_threshold:
            return True
        return False

    def compute_rewards_and_dones(self, action_indices):
        encoded_states = np.array([self.state_encode_helper(env) for env in self.envs])
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        step_dones = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if env.get_is_done() or action_indices[i] is None:
                step_dones[i] = True
                continue

            if not step_dones[i] and env.get_move_count() >= self.max_step_count:
                step_dones[i] = True
                rewards[i] = -100.0 / self.hard_factor
                continue

            src, dst = self.index_to_action[action_indices[i]]
            env.move(src, dst)
            if self.check_is_recursive_move(env):
                env.set_is_recursive_move(True)
                rewards[i] = -500.0 / self.hard_factor
                step_dones[i] = True
            else:
                rewards[i] += self.compute_action_reward(env, src, dst)
                step_dones[i] = env.get_is_solved()
        encoded_next_states = np.array([self.state_encode_helper(env) for env in self.envs])
        return encoded_states, encoded_next_states, rewards, step_dones

    def store_transitions(self, encoded_states, action_indices, rewards, encoded_next_states, step_dones):
        for i, env in enumerate(self.envs):
            if env.get_is_done() or action_indices[i] is None:
                continue

            priority = 1.0 if env.get_recent_count() > 1 else 0.35
            self.agent.store_transition(
                encoded_states[i],
                action_indices[i],
                rewards[i] * priority,
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
            f"Episode {episode: 03d} | Step {step_count: 03d} | Solved Rate: {solve_rate * 100.0: 6.2f}% "
            f"| Avg. Solved/Unsolved Rewards: {avg_solve_reward: 6.2f} / {avg_unsolved_reward: 6.2f} "
            f"| Recursive Moves Rate: {reach_recursive_moves_rate * 100.0: 6.2f}% "
            f"| Out of Move Rate: {out_of_move_rate * 100.0: 6.2f}% | Done Rate: {done_rate * 100.0: 6.2f}%"
        )

        self.logger.info(
            f"Episode {episode: 03d} | Elapsed Time "
            f"| Evaluate Networks: {cum_time_dict['network_time'] / 10**9:.2f} seconds "
            f"| MCTS actions: {cum_time_dict['mcts_time'] / 10**9:.2f} seconds "
            f"| Decide next action: {cum_time_dict['decide_next_action'] / 10**9:.2f} seconds "
            f"| Select actions: {cum_time_dict['select_actions'] / 10**9:.2f} seconds "
            f"| Compute rewards: {cum_time_dict['compute_rewards'] / 10**9:.2f} seconds "
            f"| Train step: {cum_time_dict['train_step'] / 10**9:.2f} seconds "
            f"| Update target network: {cum_time_dict['update_target_network'] / 10**9:.2f} seconds"
        )
        for depth, count in num_envs_in_depth.items():
            self.logger.info(f"Episode {episode: 03d} | Depth {depth} | Number of environments: {count / step_count:.2f}")

    def train(self, num_episodes, mcts_depth=8, discount_factor=0.9, train_steps_per_move=1, print_each_env=True):
        recent_results = deque(maxlen=10)

        for episode in range(num_episodes):
            # Reset all environments
            for env in self.envs:
                env.reset()

            # Initialize variables for tracking time
            cum_time_dict = defaultdict(float)
            num_envs_in_depth = defaultdict(int)
            total_rewards = np.zeros(self.num_envs, dtype=np.float32)
            step_count = 0

            start_time = datetime.datetime.now()
            self.logger.info(f"Episode {episode: 03d} | Start training for Episode {episode: 03d}")

            while step_count < self.max_step_count:
                if step_count > 0 and step_count % 20 == 0:
                    self.logging_progress(episode, step_count, total_rewards, cum_time_dict, num_envs_in_depth)

                # Get valid actions and select actions for each environment
                # logging.info(f"Episode {episode: 03d} | Step {step_count: 03d} | Select actions")
                select_actions_start = time.time_ns()
                action_indices, eplased_time_dict = self.select_actions(mcts_depth, discount_factor)
                cum_time_dict["select_actions"] += time.time_ns() - select_actions_start
                cum_time_dict["network_time"] += eplased_time_dict["network_time"]
                cum_time_dict["mcts_time"] += eplased_time_dict["mcts_time"]
                cum_time_dict["decide_next_action"] += eplased_time_dict["decide_next_action"]

                # Compute rewards and update environments
                # logging.info(f"Episode {episode: 03d} | Step {step_count: 03d} | Compute rewards and dones")
                compute_rewards_start = time.time_ns()
                encoded_states, encoded_next_states, rewards, step_dones = self.compute_rewards_and_dones(action_indices)
                cum_time_dict["compute_rewards"] += time.time_ns() - compute_rewards_start

                # Store transitions in replay memory
                # logging.info(f"Episode {episode: 03d} | Step {step_count: 03d} | Store transitions")
                self.store_transitions(encoded_states, action_indices, rewards, encoded_next_states, step_dones)
                for i, env in enumerate(self.envs):
                    if step_dones[i]:
                        env.set_is_done(True)

                # Train the agent
                # logging.info(f"Episode {episode: 03d} | Step {step_count: 03d} | Train agent")
                train_step_start = time.time_ns()
                for _ in range(train_steps_per_move):
                    self.agent.train_step()
                cum_time_dict["train_step"] += time.time_ns() - train_step_start

                # Update states and done flags
                total_rewards += rewards
                step_count += 1

                if np.sum([1 for env in self.envs if env.get_is_done()]) == self.num_envs:
                    break

            if print_each_env:
                for i, env in enumerate(self.envs):
                    logging.info(
                        f"Env {i} | Move count: {env.get_move_count()} | Is solved: {env.get_is_solved()} "
                        f"| Is Out of Moves: {not env.have_valid_moves()} | Is In Recursive Moves: {env.get_is_recursive_move()} "
                        f"| State:\n{env.get_state()}"
                    )

            # Update the target network periodically
            update_target_network_start = time.time_ns()
            if episode % self.agent.target_update_freq == 0:
                self.agent.update_target_networks()
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
            if episode > 0 and episode % 5 == 0 and episode == num_episodes - 1:
                model_save_path = save_model(self.agent, self.num_colors, self.tube_capacity, episode)
                logging.info(f"Model for the checkpoint at episode {episode: 03d} is saved to {model_save_path}.")

    def train_supervised(self, num_epochs=10):
        for epoch in range(num_epochs):
            epoch_start_time = time.time_ns()
            find_solution_path_time = 0.0
            move_time = 0.0

            # Clear the replay memory for each epoch
            self.agent.memory.clear()

            num_solutions = 0
            solved_envs_count = 0
            solved_step_count = 0
            max_solved_step_count = 0
            min_solved_step_count = float('inf')
            for env in self.envs:
                env.reset()  # Reset the environment to its initial state

                # Generate the solution path using DFS
                find_solution_path_start_time = time.time_ns()
                solution_paths = env.find_solution_paths()
                find_solution_path_time += time.time_ns() - find_solution_path_start_time
                path_lens = np.array([len(path) for path in solution_paths])

                if len(path_lens) <= 0 or path_lens[0] == 0:
                    continue

                solved_envs_count += 1
                num_solutions += len(solution_paths)
                solved_step_count += np.sum(path_lens)
                max_solved_step_count = max(max_solved_step_count, np.max(path_lens))
                min_solved_step_count = min(min_solved_step_count, np.min(path_lens))

                move_start_time = time.time_ns()
                for path in solution_paths:
                    sim_env = env.clone()
                    state = self.state_encode_helper(sim_env)
                    for move in path:
                        # Get the action index for the move
                        action_idx = self.action_to_index[move]

                        # Apply the move and encode the next state
                        sim_env.move(*move)
                        next_state = self.state_encode_helper(sim_env)

                        # Store the transition in replay memory
                        done = sim_env.get_is_solved()
                        self.agent.store_transition(state, action_idx, 1.0, next_state, done)

                        # Update the current state
                        state = next_state
                move_time += time.time_ns() - move_start_time

            # Train the agent using the stored transitions
            num_train_steps = min(1, solved_step_count // self.agent.batch_size)
            train_step_start_time = time.time_ns()
            for _ in range(num_train_steps):
                self.agent.train_step()
            train_step_time = time.time_ns() - train_step_start_time

            epoch_end_time = time.time_ns()
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} | Solved Rate: {solved_envs_count / len(self.envs) * 100.0: 7.2f}% "
                f"| Average steps: {solved_step_count / num_solutions if num_solutions > 0 else 0} "
                f"| Max/Min steps: {max_solved_step_count} / {min_solved_step_count} "
                f"| Total solution path count: {solved_step_count}"
            )
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} | Total elapsed time: {(epoch_end_time - epoch_start_time) / 10**9:.2f} seconds"
                f" | Find solution path time: {find_solution_path_time / 10**9:.2f} seconds"
                f" | Move time: {move_time / 10**9:.2f} seconds"
                f" | Train step time: {train_step_time / 10**9:.2f} seconds"
            )

            # Save the model periodically
            if epoch > 0 and epoch % 5 == 0 or epoch == num_epochs - 1:
                model_save_path = save_model(self.agent, self.num_colors, self.tube_capacity, epoch, save_dir="pretrained_models")
                logging.info(f"Model for the checkpoint at episode {epoch: 03d} is saved to {model_save_path}.")
