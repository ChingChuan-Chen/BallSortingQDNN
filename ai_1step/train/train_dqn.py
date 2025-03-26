from collections import deque
import numpy as np
import torch


def train_dqn(env, agent, num_episodes=1000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, train_steps_per_move=4):
    max_colors = 12
    max_capacity = 8
    max_tubes = max_colors + 2

    def encode(state):
        """
        Encode a [num_tubes][capacity] integer list state into a fixed-length one-hot vector.
        Each ball value is one-hot encoded with (max_colors + 1) size.
        The entire array is zero-padded if num_tubes or capacity is smaller than max.
        """
        one_hot_size = max_colors + 1
        encoded = np.zeros((max_tubes * max_capacity, one_hot_size), dtype=np.float32)

        for tube_idx, tube in enumerate(state):
            for pos_idx, ball in enumerate(tube):
                flat_index = tube_idx * max_capacity + pos_idx
                encoded[flat_index, ball] = 1
            # pad remaining positions in tube with 'empty'
            for pos_idx in range(len(tube), max_capacity):
                flat_index = tube_idx * max_capacity + pos_idx
                encoded[flat_index, 0] = 1

        # pad remaining tubes (if any) with 'empty'
        for tube_idx in range(len(state), max_tubes):
            for pos_idx in range(max_capacity):
                flat_index = tube_idx * max_capacity + pos_idx
                encoded[flat_index, 0] = 1

        return encoded.flatten().tolist()

    epsilon = epsilon_start
    recent_results = deque(maxlen=10)

    all_possible_actions = [
        (i, j) for i in range(env.num_tubes) for j in range(env.num_tubes) if i != j
    ]
    action_to_index = {a: idx for idx, a in enumerate(all_possible_actions)}
    index_to_action = {idx: a for a, idx in action_to_index.items()}

    for episode in range(num_episodes):
        state = env.reset()
        encoded_state = encode(state)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            valid_actions = env.get_valid_actions()
            valid_indices = [action_to_index[a] for a in valid_actions if a in action_to_index]
            action_idx = agent.select_action(encoded_state, valid_indices, epsilon)

            if action_idx is None:
                reward = -10.0
                agent.store_transition(encoded_state, 0, reward, encoded_state, True)
                total_reward += reward
                break

            src, dst = index_to_action[action_idx]
            next_state, reward, done = env.step((src, dst))
            encoded_next_state = encode(next_state)

            agent.store_transition(encoded_state, action_idx, reward, encoded_next_state, done)
            for _ in range(train_steps_per_move):
                agent.train_step()

            encoded_state = encoded_next_state
            total_reward += reward
            steps += 1

        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        solved = int(env.is_solved())
        recent_results.append(solved)
        recent_solve_rate = sum(recent_results) / len(recent_results)

        if episode % 20 == 0:
            model_name = f"dqn_{env.num_colors}c_{env.tube_capacity}cap_ep{episode:04d}.pth"
            torch.save(agent.policy_net.state_dict(), f"checkpoints/{model_name}")

        print(f"🎯 Episode {episode:03d} | Solved: {'Yes' if solved else ' No'} "
              f"| Steps: {steps:3d} | Total Reward: {total_reward:6.2f} "
              f"| Epsilon: {epsilon:.3f} | Last 10 Solve Rate: {recent_solve_rate:.1%}")


def parallel_train_dqn(parallel_env, agent, num_episodes=1000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, train_steps_per_move=2):
    max_colors = 12
    max_capacity = 8
    max_tubes = max_colors + 2

    def encode(states: list) -> np.ndarray:
        n_envs = len(states)
        one_hot_size = max_colors + 1
        flat_length = max_tubes * max_capacity * one_hot_size
        encoded_batch = np.zeros((n_envs, flat_length), dtype=np.float32)

        for env_idx, state in enumerate(states):
            encoded = np.zeros((max_tubes * max_capacity, one_hot_size), dtype=np.float32)
            for tube_idx, tube in enumerate(state):
                for pos_idx in range(max_capacity):
                    flat_index = tube_idx * max_capacity + pos_idx
                    if pos_idx < len(tube):
                        ball = tube[pos_idx]
                    else:
                        ball = 0  # padding
                    encoded[flat_index, ball] = 1

            # pad missing tubes
            for tube_idx in range(len(state), max_tubes):
                for pos_idx in range(max_capacity):
                    flat_index = tube_idx * max_capacity + pos_idx
                    encoded[flat_index, 0] = 1

            encoded_batch[env_idx] = encoded.flatten()
        return encoded_batch

    epsilon = epsilon_start
    recent_results = deque(maxlen=10)
    n_envs = parallel_env.n_envs

    all_possible_actions = [
        (i, j) for i in range(parallel_env.num_tubes) for j in range(parallel_env.num_tubes) if i != j
    ]
    action_to_index = {a: idx for idx, a in enumerate(all_possible_actions)}
    index_to_action = {idx: a for a, idx in action_to_index.items()}

    for episode in range(num_episodes):
        states = parallel_env.reset()
        encoded_states = encode(states)
        total_rewards = np.zeros(n_envs, dtype=np.float32)
        dones = np.zeros(n_envs, dtype=bool)

        while not np.all(dones):
            all_valid_actions = parallel_env.get_all_valid_actions()
            action_indices = []

            for i in range(n_envs):
                if dones[i] or len(all_valid_actions[i]) == 0:
                    action_indices.append(None)
                    continue
                valid_idxs = [action_to_index[a] for a in all_valid_actions[i]]
                idx = agent.select_action(encoded_states[i], valid_idxs, epsilon)
                action_indices.append(idx)

            actions = [index_to_action[idx] if idx is not None else (-1, -1) for idx in action_indices]

            next_states, rewards, step_dones = parallel_env.step(actions)
            encoded_next_states = encode(next_states)

            for i in range(n_envs):
                if dones[i]:
                    continue
                if action_indices[i] is None:
                    agent.store_transition(encoded_states[i], 0, -10.0, encoded_states[i], True)
                    total_rewards[i] += -10.0
                else:
                    agent.store_transition(
                        encoded_states[i],
                        action_indices[i],
                        rewards[i],
                        encoded_next_states[i],
                        step_dones[i]
                    )
                    total_rewards[i] += rewards[i]

            for _ in range(train_steps_per_move):
                agent.train_step()

            encoded_states = encoded_next_states
            dones |= step_dones

        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        solve_rate = np.average([parallel_env.is_solved(i) for i in range(n_envs)])
        avg_solve_steps = np.average([parallel_env.step_counts[i] for i in range(n_envs) if parallel_env.is_solved(i)])
        avg_rewards = np.average(total_rewards)
        recent_results.append(solve_rate)
        recent_solve_rate = sum(recent_results) / len(recent_results)

        if episode % 50 == 0:
            model_name = f"dqn_{parallel_env.num_colors}c_{parallel_env.tube_capacity}cap_ep{episode:04d}.pth"
            torch.save(agent.policy_net.state_dict(), f"checkpoints/{model_name}")

        print(f"🎯 Episode {episode: 03d} | Solve Rate: {solve_rate * 100: 6.1f}% "
              f"| Avg. Steps to Solve: {avg_solve_steps: 5.1f} | Avg. Rewards: {avg_rewards: 6.1f}"
              f"| Epsilon: {epsilon: .3f} | Last 10 Solve Rate: {recent_solve_rate * 100: 6.1f}%")

