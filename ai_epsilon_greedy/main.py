import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


# Function to transfer weights while reinitializing the output layer
def transfer_weights(old_model_path, new_model):
    old_state_dict = torch.load(old_model_path, map_location="cpu")
    new_state_dict = new_model.state_dict()

    # Check if action dimensions match
    old_action_dim = old_state_dict["fc4.weight"].shape[0]  # Output size of old model
    new_action_dim = new_state_dict["fc4.weight"].shape[0]  # Output size of new model

    # Copy matching layers
    for name, param in old_state_dict.items():
        if name in new_state_dict and param.shape == new_state_dict[name].shape:
            new_state_dict[name] = param  # Transfer weights

    # Reinitialize the output layer (`fc3`)

    # Reinitialize output layer if action_dim changed
    if old_action_dim != new_action_dim:
        new_state_dict["fc4.weight"] = torch.nn.init.xavier_uniform_(new_state_dict["fc4.weight"])
        new_state_dict["fc4.bias"] = torch.zeros_like(new_state_dict["fc4.bias"])

    new_model.load_state_dict(new_state_dict, strict=False)
    return new_model


if __name__ == "__main__":
    from env.ball_sort_env import BallSortEnv
    from env.parallel_sort_env import ParallelBallSortEnv
    from agent.dqn_agent import DQNAgent
    from train.train_dqn import train_dqn, parallel_train_dqn
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel = True
    max_capacity = 8
    max_colors = 12
    max_tubes = max_colors + 2
    state_dim = max_tubes * max_capacity * (max_colors + 1)
    previous_model_path = 'models/dqn_4c_4cap.pth'

    train_game_size = [
        # {"num_colors": 3, "capacity": 4, "episodes": 50},
        # {"num_colors": 4, "capacity": 4, "episodes": 100},
        {"num_colors": 5, "capacity": 4, "episodes": 100},
        {"num_colors": 6, "capacity": 4, "episodes": 200},
        {"num_colors": 7, "capacity": 4, "episodes": 400},
        {"num_colors": 8, "capacity": 4, "episodes": 400},
        {"num_colors": 9, "capacity": 4, "episodes": 500},
        # {"num_colors": 10, "capacity": 4, "episodes": 1500},
        # {"num_colors": 11, "capacity": 4, "episodes": 1800},
        # {"num_colors": 12, "capacity": 4, "episodes": 2200},
        # {"num_colors": 4, "capacity": 6, "episodes": 1500},
        # {"num_colors": 6, "capacity": 6, "episodes": 1500},
        # {"num_colors": 8, "capacity": 6, "episodes": 1500},
        # {"num_colors": 4, "capacity": 8, "episodes": 1500},
        # {"num_colors": 6, "capacity": 8, "episodes": 1500},
        # {"num_colors": 8, "capacity": 8, "episodes": 1500},
    ]

    for stage in train_game_size:
        num_colors = stage["num_colors"]
        action_dim = (num_colors + 2) * (num_colors + 1)
        capacity = stage["capacity"]
        episodes = stage["episodes"]

        print(f"\n🧪 Training on puzzle with {num_colors} colors, capacity {capacity}, for {episodes} episodes.")

        if parallel:
            env = ParallelBallSortEnv(n_envs=128, num_colors=num_colors, tube_capacity=capacity)
        else:
            env = BallSortEnv(num_colors=num_colors, tube_capacity=capacity)

        agent = DQNAgent(state_dim, action_dim, device=device, batch_size=256)

        # Create a fresh or shared agent (shared helps retain learning across stages)
        if previous_model_path is not None:
            print("Previous model exists, load the weights...")
            transfer_weights(previous_model_path, agent.policy_net)
            print("Weights is loaded")
        if parallel:
            parallel_train_dqn(env, agent, num_episodes=episodes, train_steps_per_move=2)
        else:
            train_dqn(env, agent, num_episodes=episodes, train_steps_per_move=4)

        previous_model_path = f"models/dqn_{num_colors}c_{capacity}cap.pth"
        torch.save(agent.policy_net.state_dict(), previous_model_path)
