from collections import deque
import numpy as np
import torch


def parallel_train_dqn_2d(parallel_env, agent, max_number_colors, max_tube_capacity, num_episodes=1000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.975, train_steps_per_move=2):
    max_number_tubes = max_number_colors + 2

    def encode(states: list) -> np.ndarray:
        encoded_batch = np.zeros((len(states), max_number_colors + 1, max_number_tubes, max_tube_capacity), dtype=np.float32)
        for env_idx, state in enumerate(states):
            for tube_idx, tube in enumerate(state):
                for pos_idx, ball in enumerate(tube):
                    encoded_batch[env_idx, ball, tube_idx, pos_idx] = 1  # One-hot encoding

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

        if episode > 0 and episode % 10 == 0:
            model_name = f"dqn_{parallel_env.num_colors}c_{parallel_env.tube_capacity}cap_ep{episode:04d}_resnet.pth"
            torch.save(agent.policy_net.state_dict(), f"checkpoints/{model_name}")

        print(f"🎯 Episode {episode: 03d} | Solve Rate: {solve_rate * 100: 6.1f}% "
              f"| Avg. Steps to Solve: {avg_solve_steps: 5.1f} | Avg. Rewards: {avg_rewards: 6.1f}"
              f"| Epsilon: {epsilon: .3f} | Last 10 Solve Rate: {recent_solve_rate * 100: 6.1f}%")
