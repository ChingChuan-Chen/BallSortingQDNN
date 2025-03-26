from collections import deque
import numpy as np
import torch
from env.ball_sort_env import BallSortEnv


def save_model(agent, num_colors, tube_capacity, episode, save_dir="checkpoints"):
    """
    Save the model's state dictionary to a file.

    Args:
        agent (AlphaSortAgent): The agent whose model is to be saved.
        num_colors (int): Number of colors in the puzzle.
        tube_capacity (int): Maximum capacity of each tube.
        episode (int): Current training episode.
        save_dir (str): Directory to save the model.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the policy network
    policy_model_name = f"alphasort_policy_{num_colors}c_{tube_capacity}cap_ep{episode:04d}.pth"
    policy_model_path = os.path.join(save_dir, policy_model_name)
    torch.save(agent.policy_net.state_dict(), policy_model_path)

    # Save the value network
    value_model_name = f"alphasort_value_{num_colors}c_{tube_capacity}cap_ep{episode:04d}.pth"
    value_model_path = os.path.join(save_dir, value_model_name)
    torch.save(agent.value_net.state_dict(), value_model_path)

    print(f"✅ Models saved to {policy_model_path} and {value_model_path}")


def encode(states: list, max_num_colors: int, max_tubes: int, max_tube_capacity: int) -> np.ndarray:
    """
    Encode states into one-hot representations.

    Args:
        states (list): List of states to encode.
        max_num_colors (int): Maximum number of colors.
        max_tubes (int): Maximum number of tubes.
        max_tube_capacity (int): Maximum capacity of each tube.

    Returns:
        np.ndarray: Encoded states as one-hot representations.
    """
    encoded_batch = np.zeros((len(states), max_num_colors + 1, max_tubes, max_tube_capacity), dtype=np.float32)
    for env_idx, state in enumerate(states):
        for tube_idx, tube in enumerate(state):
            for pos_idx, ball in enumerate(tube):
                encoded_batch[env_idx, ball, tube_idx, pos_idx] = 1  # One-hot encoding
    return encoded_batch


def select_actions(agent, encoded_states, all_valid_actions, action_to_index, epsilon, dones, mcts_simulations=10):
    """
    Select actions for all environments using the MCTS strategy.

    Args:
        agent (AlphaSortAgent): The agent to select actions.
        encoded_states (np.ndarray): Encoded states of all environments.
        all_valid_actions (list): List of valid actions for each environment.
        action_to_index (dict): Mapping of actions to indices.
        epsilon (float): Epsilon for epsilon-greedy policy.
        dones (np.ndarray): Done flags for each environment.
        mcts_simulations (int): Number of MCTS simulations to run.

    Returns:
        list: List of selected action indices.
    """
    action_indices = []

    for i in range(len(encoded_states)):
        if dones[i] or len(all_valid_actions[i]) == 0:
            action_indices.append(None)
            continue

        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            valid_idxs = [action_to_index[a] for a in all_valid_actions[i]]
            action_indices.append(np.random.choice(valid_idxs))
            continue

        # MCTS-based action selection
        best_action = None
        best_value = float('-inf')

        for action in all_valid_actions[i]:
            total_value = 0.0

            for _ in range(mcts_simulations):
                # Simulate the action
                simulated_env = BallSortEnv(
                    num_colors=agent.num_colors,
                    tube_capacity=agent.max_tube_capacity,
                    num_empty_tubes=agent.num_colors + 2 - agent.num_colors,
                    state=encoded_states[i].copy()
                )
                src, dst = action
                if simulated_env.is_valid_move(src, dst):
                    simulated_env.move(src, dst)

                # Evaluate the resulting state
                simulated_state = simulated_env.state
                encoded_simulated_state = encode(
                    [simulated_state],
                    max_num_colors=agent.max_colors,
                    max_tubes=agent.num_colors + 2,
                    max_tube_capacity=agent.max_tube_capacity
                )[0]
                value = agent.evaluate_state(encoded_simulated_state)
                total_value += value

            # Average the value over simulations
            avg_value = total_value / mcts_simulations

            # Update the best action
            if avg_value > best_value:
                best_value = avg_value
                best_action = action

        # Convert the best action to its index
        action_indices.append(action_to_index[best_action])

    return action_indices


def compute_rewards_and_dones(envs, actions, states, dones, tube_capacity):
    """
    Compute rewards and update done flags for all environments.

    Args:
        envs (list): List of BallSortEnv environments.
        actions (list): List of actions (src, dst) for each environment.
        states (list): Current states of the environments.
        dones (np.ndarray): Done flags for each environment.
        tube_capacity (int): Maximum capacity of each tube.

    Returns:
        tuple: (next_states, rewards, step_dones)
    """
    next_states = []
    rewards = []
    step_dones = []

    for i, env in enumerate(envs):
        if dones[i]:
            next_states.append(states[i])
            rewards.append(0.0)
            step_dones.append(True)
            continue

        src, dst = actions[i]
        if not env.is_valid_move(src, dst):
            rewards.append(-5.0)  # Penalty for invalid move
            next_states.append(states[i])
            step_dones.append(False)
        else:
            env.move(src, dst)
            next_states.append(env.state.copy())
            rewards.append(-0.1)  # Small penalty for each step
            if env.num_balls_per_tube[dst] == tube_capacity and np.all(env.state[dst, :] == env.state[dst, 0]):
                rewards[-1] += 2.0  # Bonus for completing a tube
            step_dones.append(env.is_solved())

    return next_states, rewards, step_dones


def store_transitions(agent, encoded_states, action_indices, rewards, encoded_next_states, step_dones, dones, num_envs):
    """
    Store transitions in the replay memory for all environments.

    Args:
        agent (AlphaSortAgent): The agent to train.
        encoded_states (np.ndarray): Encoded current states.
        action_indices (list): List of action indices for each environment.
        rewards (list): List of rewards for each environment.
        encoded_next_states (np.ndarray): Encoded next states.
        step_dones (list): Done flags for each environment.
        dones (np.ndarray): Done flags for each environment.
        num_envs (int): Number of environments.
    """
    for i in range(num_envs):
        if dones[i]:
            continue
        if action_indices[i] is None:
            agent.store_transition(encoded_states[i], 0, -10.0, encoded_states[i], True)
        else:
            agent.store_transition(
                encoded_states[i],
                action_indices[i],
                rewards[i],
                encoded_next_states[i],
                step_dones[i]
            )


def train_alpha_sort(envs, agent, max_num_colors, max_tube_capacity, num_episodes=1000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.975, train_steps_per_move=2):
    """
    Train the AlphaSortAgent using multiple BallSortEnv environments.

    Args:
        envs (list): List of BallSortEnv environments.
        agent (AlphaSortAgent): The agent to train.
        max_num_colors (int): Number of colors in the puzzle.
        max_tube_capacity (int): Maximum capacity of each tube.
        num_episodes (int): Number of training episodes.
        epsilon_start (float): Initial epsilon for epsilon-greedy policy.
        epsilon_end (float): Minimum epsilon for epsilon-greedy policy.
        epsilon_decay (float): Decay rate for epsilon.
        train_steps_per_move (int): Number of training steps per move.
    """
    num_colors = envs[0].num_colors
    tube_capacity = envs[0].tube_capacity
    num_empty_tubes = envs[0].num_empty_tubes
    num_envs = len(envs)
    max_tubes = max_num_colors + num_empty_tubes

    epsilon = epsilon_start
    recent_results = deque(maxlen=10)

    # Precompute all possible actions
    all_possible_actions = [
        (i, j) for i in range(num_colors + num_empty_tubes) for j in range(num_colors + num_empty_tubes) if i != j
    ]
    action_to_index = {a: idx for idx, a in enumerate(all_possible_actions)}
    index_to_action = {idx: a for a, idx in action_to_index.items()}

    for episode in range(num_episodes):
        # Reset all environments
        states = [env.reset() for env in envs]
        encoded_states = encode(states, max_num_colors, max_tubes, max_tube_capacity)
        total_rewards = np.zeros(num_envs, dtype=np.float32)
        dones = np.zeros(num_envs, dtype=bool)

        while not np.all(dones):
            all_valid_actions = [env.get_valid_actions() for env in envs]

            # Select actions for each environment
            action_indices = select_actions(agent, encoded_states, all_valid_actions, action_to_index, epsilon, dones)

            # Map action indices to actions
            actions = [index_to_action[idx] if idx is not None else (-1, -1) for idx in action_indices]

            # Compute rewards and update environments
            next_states, rewards, step_dones = compute_rewards_and_dones(envs, actions, states, dones, tube_capacity)

            # Encode next states
            encoded_next_states = encode(next_states, max_num_colors, max_tubes, max_tube_capacity)

            # Store transitions in replay memory
            store_transitions(agent, encoded_states, action_indices, rewards, encoded_next_states, step_dones, dones, num_envs)

            # Train the agent
            for _ in range(train_steps_per_move):
                agent.train_step()

            # Update states and done flags
            encoded_states = encoded_next_states
            dones |= step_dones

        # Update the target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        # Log training progress
        solve_rate = np.mean([env.is_solved() for env in envs])
        avg_rewards = np.mean(total_rewards)
        recent_results.append(solve_rate)
        recent_solve_rate = np.mean(recent_results)

        print(f"🎯 Episode {episode: 03d} | Solve Rate: {solve_rate * 100: 6.1f}% "
              f"| Avg. Rewards: {avg_rewards: 6.1f} | Epsilon: {epsilon: .3f} "
              f"| Last 10 Solve Rate: {recent_solve_rate * 100: 6.1f}%")

        # Save the model periodically
        if episode > 0 and episode % 10 == 0:
            save_model(agent, num_colors, tube_capacity, episode)
