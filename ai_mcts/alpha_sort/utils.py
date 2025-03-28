import os
import numpy as np
import torch
import hashlib

def hash_state(state):
    """
    Generate a hash for the given state.

    Args:
        state (np.ndarray): The state to hash.

    Returns:
        str: A unique hash for the state.
    """
    state_bytes = state.tobytes()  # Convert the state to bytes
    return hashlib.sha256(state_bytes).hexdigest()  # Generate a SHA-256 hash

def save_model(agent, num_colors, tube_capacity, episode=None, save_dir="checkpoints"):
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
    if episode is None:
        policy_model_name = f"alphasort_policy_{num_colors}c_{tube_capacity}cap.pth"
    else:
        policy_model_name = f"alphasort_policy_{num_colors}c_{tube_capacity}cap_ep{episode:04d}.pth"
    policy_model_path = os.path.join(save_dir, policy_model_name)
    torch.save(agent.policy_net.state_dict(), policy_model_path)

    print(f"✅ Policy Network is successfully saved to {policy_model_path}")


def state_encode(states: list, max_num_colors: int, num_empty_tubes: int, max_tube_capacity: int) -> np.ndarray:
    """
    Encode states into one-hot representations.

    Args:
        states (list): List of states to encode.
        max_num_colors (int): Maximum number of colors.
        num_empty_tubes (int): Number of empty tubes.
        max_tube_capacity (int): Maximum capacity of each tube.

    Returns:
        np.ndarray: Encoded states as one-hot representations.
    """
    encoded_batch = np.zeros((len(states), max_num_colors + 1, max_num_colors + num_empty_tubes, max_tube_capacity), dtype=np.float32)
    for env_idx, state in enumerate(states):
        for tube_idx, tube in enumerate(state):
            for pos_idx, ball in enumerate(tube):
                encoded_batch[env_idx, ball, tube_idx, pos_idx] = 1  # One-hot encoding
    return encoded_batch

def state_decode(encoded_states, num_colors: int, num_empty_tubes: int, tube_capacity: int) -> list:
    """
    Decode one-hot encoded states back to the original state representation.

    Args:
        encoded_states (np.ndarray): One-hot encoded states of shape
                                      (batch_size, num_colors + 1, num_colors + num_empty_tubes, tube_capacity).
        num_colors (int): number of colors.
        num_empty_tubes (int): Number of empty tubes.
        tube_capacity (int): Capacity of each tube.

    Returns:
        list: Decoded states as a list of 2D arrays, where each array represents a state.
    """
    batch_size = encoded_states.shape[0]
    decoded_states = []

    for i in range(batch_size):
        state = np.zeros((num_colors + num_empty_tubes, tube_capacity), dtype=np.int32)
        for tube_idx in range(num_colors + num_empty_tubes):
            for pos_idx in range(tube_capacity):
                # Find the color index (ball type) with the highest value in the one-hot encoding
                color = np.argmax(encoded_states[i, :, tube_idx, pos_idx])
                state[tube_idx, pos_idx] = color
        decoded_states.append(state)

    return decoded_states
