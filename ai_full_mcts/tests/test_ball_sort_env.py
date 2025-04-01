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

    for i in range(filled_tubes, env.num_tubes):
        assert np.all(env.state[i] == 0)  # Empty tubes should have no balls

    # validate Cython environment
    assert env.is_solved == False
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


def test_memory_view_access(env):
    env.state[0, 0] = 99
    assert env._env.get_state()[0, 0] == 99


def test_clone(env):
    # Set up the initial state
    env.reset()
    env.move(0, 5)  # Perform a move to modify the state

    # Clone the environment
    cloned_env = env.clone()

    # Validate the get_valid_actions method
    valid_actions = env.valid_actions
    cloned_valid_actions = cloned_env.valid_actions
    assert len(valid_actions) == len(cloned_valid_actions), "Valid actions count does not match between original and cloned environments."

    # Validate that the cloned environment has the same state
    assert np.array_equal(env.state, cloned_env.state), "Cloned state does not match the original state."

    # Validate that the cloned environment is independent (deep copy)
    cloned_env.state[0, 0] = 99
    assert env.state[0, 0] != 99, "Original state was modified when changing the cloned state."

    # Validate that the cloned environment's move count matches the original
    assert env.get_move_count() == cloned_env.get_move_count(), "Move count does not match between original and cloned environments."

    # Validate that the cloned environment's solved status matches the original
    assert env.is_solved == cloned_env.is_solved, "Solved status does not match between original and cloned environments."

    # Validate that the environment's move count is updated correctly and independently
    env.move(0, 4)  # Perform another move in the original environment
    assert env.get_move_count() != cloned_env.get_move_count(), "Move count should not match after modifying the original environment."


def test_get_valid_actions(env):
    """Test the get_valid_actions method."""
    env.reset()
    valid_actions = env.valid_actions

    assert len(valid_actions) > 0, "There should be valid actions available."
    assert env.is_done == False, "The environment should not be done after getting valid actions."
    assert env.is_out_of_moves == False, "The environment should not be out of moves after getting valid actions."
