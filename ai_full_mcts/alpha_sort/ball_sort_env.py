import random
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple
from alpha_sort.lib._ball_sort_game import C_BallSortEnv
from alpha_sort.utils import hash_state


class BallSortEnv:
    rng = np.random.default_rng(random.randint(0, 2**31-1))
    def __init__(self, num_colors: int, tube_capacity: int = 4, num_empty_tubes: int = 2, state: np.ndarray = None):
        self.num_colors = num_colors
        self.tube_capacity = tube_capacity
        self.num_empty_tubes = num_empty_tubes
        self.num_tubes = num_colors + num_empty_tubes

        # Initialize the state as NumPy arrays
        self.state = np.zeros((self.num_tubes, self.tube_capacity), dtype=np.int8)

        # initialize state related variables
        self.state_key = None
        self.state_history = defaultdict(int)
        self.recent_state_keys = deque(maxlen=20)
        self.is_out_of_moves = False
        self.is_in_recursive_moves = False
        self.is_done = False
        self.is_solved = False
        self.valid_actions = None

        if state is not None:
            is_valid, reason = self.is_valid_state(state)
            if not is_valid:
                raise ValueError(f"State is invalid since {reason}.")
            self.state = state
            self.state_key = hash_state(self.state)

        # Initialize the Cython environment with memory views
        self._env = C_BallSortEnv(
            tube_capacity=self.tube_capacity,
            num_colors=self.num_colors,
            num_empty_tubes=self.num_empty_tubes,
            state=self.state
        )

        if state is not None:
            self.state_key = hash_state(self.state)
            self.valid_actions = self._env.get_valid_actions()
            self.is_out_of_moves = len(self.valid_actions) == 0
            self.is_solved = self._env.is_solved()
            self.is_done = self.is_solved
        else:
            self.reset()

    def reset(self):
        # Reset the Cython environment
        self._env.reset()

        # Clear the state
        self.state.fill(0)

        # Generate shuffled balls
        balls = np.repeat(np.arange(1, self.num_colors + 1), self.tube_capacity)
        self.rng.shuffle(balls)

        # Fill the tubes with balls
        filled_tubes = self.num_tubes - self.num_empty_tubes  # Leave empty tubes
        for i in range(filled_tubes):
            self.state[i, :] = balls[i * self.tube_capacity:(i + 1) * self.tube_capacity]

        # update state_key and other state-related variables
        self.state_key = hash_state(self.state)
        self.valid_actions = self._env.get_valid_actions()
        self.is_out_of_moves = len(self.valid_actions) == 0
        self.is_solved = self._env.is_solved()
        self.state_history.clear()
        self.state_history[self.state_key] = 1
        self.recent_state_keys.clear()
        self.is_done = False
        self.is_in_recursive_moves = False
        self._env.set_move_count(0)

    def is_valid_state(self, state: np.ndarray) -> Tuple[bool, str]:
        # Check shape
        if state.shape != (self.num_tubes, self.tube_capacity):
            return False, "the shape is not correct"

        # Check for valid ball colors and empty slots
        if not np.all((state >= 0) & (state <= self.num_colors)):
            return False, "there are invalid ball colors"

        # Count total balls and validate color counts
        total_balls = np.count_nonzero(state)
        if total_balls != self.num_colors * self.tube_capacity:
            return False, "the total number of balls is incorrect"

        # Validate that each color appears exactly tube_capacity times
        color_counts = np.bincount(state[state > 0].flatten(), minlength=self.num_colors + 1)
        for color in range(1, self.num_colors + 1):
            if color_counts[color] != self.tube_capacity:
                return False, f"color {color} appears {color_counts[color]} times"
        return True, None

    def is_full_tube(self, tube_idx: int) -> bool:
        return self._env.is_full_tube(tube_idx)

    def is_empty_tube(self, tube_idx: int) -> bool:
        return self._env.is_empty_tube(tube_idx)

    def is_completed_tube(self, tube_idx: int) -> bool:
        return self._env.is_completed_tube(tube_idx)

    def top_index(self, tube: int) -> int:
        return self._env.top_index(tube)

    def get_top_color_streak(self, tube: int) -> int:
        return self._env.get_top_color_streak(tube)

    def move(self, src: int, dst: int) -> None:
        self.state_history[self.state_key] += 1
        self.recent_state_keys.append(self.state_key)
        self._env.move(src, dst)
        self.state_key = hash_state(self.state)
        self.valid_actions = self._env.get_valid_actions()
        self.is_out_of_moves = len(self.valid_actions) == 0
        self.is_solved = self._env.is_solved()
        self.is_done |= self.is_solved

    def undo_move(self, src: int, dst: int) -> None:
        self.state_history[self.state_key] -= 1
        if self.state_history[self.state_key] == 0:
            del self.state_history[self.state_key]
        self.recent_state_keys.pop()
        self._env.undo_move(src, dst)
        self.state_key = hash_state(self.state)
        self.valid_actions = self._env.get_valid_actions()
        self.is_out_of_moves = len(self.valid_actions) == 0
        self.is_solved = self._env.is_solved()
        self.is_done |= self.is_solved

    def get_move_count(self) -> int:
        return self._env.get_move_count()

    def is_valid_move(self, src: int, dst: int) -> bool:
        return self._env.is_valid_move(src, dst)

    def clone(self):
        new_env = BallSortEnv(
            num_colors=self.num_colors,
            tube_capacity=self.tube_capacity,
            num_empty_tubes=self.num_empty_tubes,
            state=self.state.copy()
        )
        new_env.state_history = defaultdict(int, self.state_history)
        new_env.recent_state_keys = deque(self.recent_state_keys, maxlen=1000)
        new_env.is_done = self.is_done
        new_env.state_key = self.state_key
        new_env.valid_actions = self.valid_actions.copy()
        new_env.is_out_of_moves = self.is_out_of_moves
        new_env.is_solved = self.is_solved
        new_env.is_done = self.is_done
        new_env._env.set_move_count(self.get_move_count())

        return new_env