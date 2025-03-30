import os
import sys
import numpy as np
import pytest

# Add the lib directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the BallSortEnv class from the Cython extension
from alpha_sort import BallSortEnv
from alpha_sort.lib._ball_sort_game import C_BallSortEnv


@pytest.fixture
def env():
    """Fixture to create a BallSortEnv instance."""
    return BallSortEnv(num_colors=4, tube_capacity=4, num_empty_tubes=2)


def test_reset(env):
    """Test the reset method."""
    env.reset()

    # Validate the state shape
    assert env.state.shape == (env.num_tubes, env.tube_capacity)

    # Validate the number of balls per tube
    filled_tubes = env.num_tubes - env.num_empty_tubes
    for i in range(filled_tubes):
        assert np.all(env.state[i] > 0)  # Filled tubes should have balls
        assert env.num_balls_per_tube[i] == env.tube_capacity

    for i in range(filled_tubes, env.num_tubes):
        assert np.all(env.state[i] == 0)  # Empty tubes should have no balls
        assert env.num_balls_per_tube[i] == 0

    # validate Cython environment
    assert env.is_moved() == True
    assert env.is_solved() == False
    assert env.get_move_count() == 0


def test_is_valid_state(env):
    """Test the is_valid_state method."""
    # Valid state
    valid_state = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    is_valid, reason = env.is_valid_state(valid_state)
    assert is_valid
    assert reason is None

    # Invalid state (wrong shape)
    invalid_state_shape = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    is_valid, reason = env.is_valid_state(invalid_state_shape)
    assert not is_valid
    assert reason == "the shape is not correct"

    # Invalid state (wrong ball count)
    invalid_state_count = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    is_valid, reason = env.is_valid_state(invalid_state_count)
    assert not is_valid
    assert reason == "the total number of balls is incorrect"


def test_update_num_balls_per_tube(env):
    """Test the update_num_balls_per_tube method."""
    state = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 0],
        [3, 3, 0, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    env.state = state
    env.update_num_balls_per_tube()

    # Validate num_balls_per_tube
    expected_num_balls = [4, 3, 2, 1, 0, 0]
    assert np.array_equal(env.num_balls_per_tube, expected_num_balls)


def test_memory_view_access(env):
    """Test if state and num_balls_per_tube are accessed by C_BallSortEnv with memory view."""
    # do a move and update num_balls_per_tube. Then check if the num_balls_per_tube is updated
    env.move(0, 5)
    env.update_num_balls_per_tube()
    assert env._env.get_num_balls_per_tube()[0] == 3
    assert env._env.get_num_balls_per_tube()[5] == 1

    # Modify state and num_balls_per_tube directly
    env.state[0, 0] = 99
    env.num_balls_per_tube[0] = 111

    # Access state and num_balls_per_tube from C_BallSortEnv
    assert env._env.get_state()[0, 0] == 99
    assert env._env.get_num_balls_per_tube()[0] == 111
