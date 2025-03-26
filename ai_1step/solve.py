import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def solve_game(env, agent, encoder, render=True, max_steps=200):
    """
    Uses a trained agent to attempt solving a single game without exploration.
    """
    state = env.reset()
    encoded_state = encoder.encode(state)
    done = False
    total_reward = 0
    steps = 0

    all_possible_actions = [
      (i, j) for i in range(env.num_tubes) for j in range(env.num_tubes) if i != j
    ]
    action_to_index = {a: idx for idx, a in enumerate(all_possible_actions)}
    index_to_action = {idx: a for a, idx in action_to_index.items()}

    if render:
        print("Initial State:")
        for i, tube in enumerate(state):
            print(f"Tube {i + 1}: {tube}")
        print()

    while not done and steps < max_steps:
        valid_actions = env.get_valid_actions()
        valid_indices = [action_to_index[a] for a in valid_actions if a in action_to_index]

        action_idx = agent.select_action(encoded_state, valid_indices, epsilon=0.0)

        if action_idx is None:
            print("No valid moves left.")
            break

        src, dst = index_to_action[action_idx]
        next_state, reward, done = env.step((src, dst))
        encoded_state = encoder.encode(next_state)
        total_reward += reward
        steps += 1

        if render:
            print(f"Step {steps}: Move from Tube {src + 1} to Tube {dst + 1}")
            for i, tube in enumerate(next_state):
                print(f"Tube {i + 1}: {tube}")
            print()

        print(f"Finished in {steps} steps. Total reward: {total_reward:.2f}")
        print(f"🧩 Solved: {'Yes' if env.is_solved() else 'No'}")
        print(f"🔢 Steps taken: {steps}")
    if env.is_solved():
        print("🎉 Puzzle Solved!")
    else:
        print("❌ Puzzle not solved.")


if __name__ == "__main__":
    from env.ball_sort_env import BallSortEnv
    from agent.dqn_agent import DQNAgent
    from train.train_dqn import StateEncoder, train_dqn
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = BallSortEnv(num_colors=6, tube_capacity=4)  # any config
    encoder = StateEncoder(max_color=12, max_capacity=8)

    state_dim = encoder.encoding_size
    action_dim = env.num_tubes * (env.num_tubes - 1)

    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.policy_net.load_state_dict(torch.load("models/puzzle_dqn_weights.pth", map_location=device))
    agent.policy_net.eval()

    solve_game(env, agent, encoder, render=True)
