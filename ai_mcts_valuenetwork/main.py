import sys
import os
import logging
import traceback
import faulthandler
faulthandler.enable()

# Configure the logger
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Get the root logger
logger = logging.getLogger()

# Add a FileHandler to log messages to a file
file_handler = logging.FileHandler('training_log.txt', mode='a')  # Append mode
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

if __name__ == "__main__":
    from alpha_sort import BallSortEnv, AlphaSortAgent, AlphaSortTrainer, save_model
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_tube_capacity = 8
    max_num_colors = 12
    num_empty_tubes = 2
    max_number_tubes = max_num_colors + num_empty_tubes
    n_envs = 64
    previous_model_path = "models/alphasort_model_5c_4cap.pth"

    train_game_size = [
        {"num_colors": 4, "tube_capacity": 4, "episodes": 200},
        {"num_colors": 5, "tube_capacity": 4, "episodes": 250},
        {"num_colors": 7, "tube_capacity": 4, "episodes": 300},
        # {"num_colors": 9, "tube_capacity": 4, "episodes": 70},
        # {"num_colors": 11, "tube_capacity": 4, "episodes": 100},
        # {"num_colors": 12, "tube_capacity": 4, "episodes": 200},
        # {"num_colors": 6, "tube_capacity": 6, "episodes": 100},
        # {"num_colors": 8, "tube_capacity": 6, "episodes": 150},
        # {"num_colors": 10, "tube_capacity": 6, "episodes": 150},
        # {"num_colors": 12, "tube_capacity": 6, "episodes": 200},
        # {"num_colors": 6, "tube_capacity": 8, "episodes": 150},
        # {"num_colors": 8, "tube_capacity": 8, "episodes": 150},
        # {"num_colors": 10, "tube_capacity": 8, "episodes": 200},
        # {"num_colors": 12, "tube_capacity": 8, "episodes": 200},
    ]

    for stage in train_game_size:
        num_colors = stage["num_colors"]
        tube_capacity = stage["tube_capacity"]
        episodes = stage["episodes"]

        logger.info(f"Training on puzzle with {num_colors} colors, tube_capacity {tube_capacity}, for {episodes} episodes on {device}.")

        envs = [BallSortEnv(num_colors, tube_capacity, num_empty_tubes) for _ in range(n_envs)]

        agent = AlphaSortAgent(num_colors, max_num_colors, max_tube_capacity, device, batch_size=64)

        # Create a fresh or shared agent (shared helps retain learning across stages)
        if previous_model_path is not None:
            logger.info("Previous model exists, load the weights...")
            agent.load_pretrained_weights(previous_model_path)
            logger.info("Weights is loaded")
        else:
            logger.info("No previous model found, start fresh.")

        trainer = AlphaSortTrainer(envs, agent, max_num_colors, max_tube_capacity)

        try:
            trainer.train(episodes, mcts_depth=5, train_steps_per_move=1)
        except Exception as e:
            logger.error(f"An error occurred during training: {e} and the traceback is {traceback.format_exc()}")
            raise

        previous_model_path = save_model(agent, num_colors, tube_capacity, save_dir="models")
