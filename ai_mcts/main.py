import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

if __name__ == "__main__":
    from alpha_sort import BallSortEnv, AlphaSortAgent, AlphaSortTrainer, save_model
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_tube_capacity = 8
    max_num_colors = 12
    num_empty_tubes = 2
    max_number_tubes = max_num_colors + num_empty_tubes
    n_envs = 256
    previous_model_path = 'checkpoints/alphasort_policy_5c_4cap_ep0010.pth'

    train_game_size = [
        # {"num_colors": 4, "capacity": 4, "episodes": 50, "epsilon_decay":0.95},
        {"num_colors": 5, "capacity": 4, "episodes": 50, "epsilon_decay":0.95},
        {"num_colors": 6, "capacity": 4, "episodes": 100, "epsilon_decay":0.975},
        {"num_colors": 7, "capacity": 4, "episodes": 100, "epsilon_decay":0.975},
        {"num_colors": 8, "capacity": 4, "episodes": 200, "epsilon_decay":0.99},
        {"num_colors": 9, "capacity": 4, "episodes": 300, "epsilon_decay":0.9925},
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
        tube_capacity = stage["capacity"]
        episodes = stage["episodes"]
        epsilon_decay = stage.get("epsilon_decay", 0.995)

        print(f"\n🧪 Training on puzzle with {num_colors} colors, capacity {tube_capacity}, for {episodes} episodes on {device}.")

        envs = [BallSortEnv(num_colors, tube_capacity, num_empty_tubes) for _ in range(n_envs)]

        agent = AlphaSortAgent(num_colors, max_num_colors, max_tube_capacity, device, batch_size=512)

        # Create a fresh or shared agent (shared helps retain learning across stages)
        if previous_model_path is not None:
            print("Previous model exists, load the weights...")
            agent.load_pretrained_weights(previous_model_path)
            print("Weights is loaded")
        else:
            print("No previous model found, start fresh.")

        trainer = AlphaSortTrainer(envs, agent, max_num_colors, max_tube_capacity)
        trainer.train(episodes, mcts_simulations=10, mcts_depth=3, train_steps_per_move=2)

        previous_model_path = save_model(agent, num_colors, tube_capacity, save_dir="models")
