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

def save_model(agent, num_colors, tube_capacity, episode=None, save_dir="checkpoints") -> str:
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
    return policy_model_path
