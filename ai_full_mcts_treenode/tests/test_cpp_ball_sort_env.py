import os
import sys
import pytest
import numpy as np

# Add the lib directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the BallSortEnv class from the C++ extension
from alpha_sort.lib.ball_sort_env import BallSortEnv

@pytest.fixture
def env():
    state = np.array([
        [1, 1, 1, 1],
        [2, 3, 2, 2],
        [2, 3, 3, 3],
        [4, 4, 4, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    return BallSortEnv(tube_capacity=4, num_colors=4, num_empty_tubes=2, state=state)

@pytest.fixture
def solved_env():
    state = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    return BallSortEnv(tube_capacity=4, num_colors=4, num_empty_tubes=2, state=state)

@pytest.fixture
def imbalanced_env():
    state = np.array([
        [1, 1, 2, 0],
        [2, 2, 2, 0],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    return BallSortEnv(tube_capacity=4, num_colors=4, num_empty_tubes=2, state=state)

@pytest.fixture
def random_env():
    return BallSortEnv(tube_capacity=4, num_colors=4, num_empty_tubes=2)

def test_clone(env):
    # Clone the environment
    cloned_env = env.clone()

    # Ensure the cloned environment is not the same object
    assert cloned_env is not env

    # Ensure the state is identical
    np.testing.assert_array_equal(cloned_env.get_state(), env.get_state())

    # Ensure other attributes are identical
    assert cloned_env.get_move_count() == env.get_move_count()
    assert cloned_env.get_is_done() == env.get_is_done()
    assert cloned_env.get_is_moved() == env.get_is_moved()
    assert cloned_env.get_is_solved() == env.get_is_solved()

    # Modify the cloned environment and ensure it does not affect the original
    cloned_env.move(1, 4)
    assert not np.array_equal(cloned_env.get_state(), env.get_state())
    assert cloned_env.get_valid_moves() != env.get_valid_moves()
    assert cloned_env.is_valid_move(2, 4) == False
    assert env.is_valid_move(2, 4) == True
    assert cloned_env.get_move_count() != env.get_move_count()

    # Modify the environment and ensure it does not affect the original
    env.move(0, 4)
    assert not np.array_equal(cloned_env.get_state(), env.get_state())
    assert cloned_env.get_valid_moves() != env.get_valid_moves()
    assert cloned_env.get_move_count() == env.get_move_count()

def test_top_index(env, imbalanced_env):
    for i in range(4):
        assert env.top_index(i) == 3
    for i in range(4, 7):
        assert env.top_index(i) == -1

    assert env.top_index(-1) == -1
    assert imbalanced_env.top_index(-1) == -1

    assert imbalanced_env.top_index(0) == 2
    assert imbalanced_env.top_index(1) == 2
    assert imbalanced_env.top_index(2) == 3
    assert imbalanced_env.top_index(3) == 3
    assert imbalanced_env.top_index(4) == 1
    assert imbalanced_env.top_index(5) == -1
    assert imbalanced_env.top_index(6) == -1

def test_get_top_color_streak(env, imbalanced_env):
    assert env.get_top_color_streak(0) == 4
    assert env.get_top_color_streak(1) == 2
    assert env.get_top_color_streak(2) == 3
    assert env.get_top_color_streak(3) == 4
    assert env.get_top_color_streak(4) == 0
    assert env.get_top_color_streak(5) == 0

    assert imbalanced_env.get_top_color_streak(0) == 1
    assert imbalanced_env.get_top_color_streak(1) == 3
    assert imbalanced_env.get_top_color_streak(2) == 4
    assert imbalanced_env.get_top_color_streak(3) == 4
    assert imbalanced_env.get_top_color_streak(4) == 2
    assert imbalanced_env.get_top_color_streak(5) == 0

def test_is_valid_move(env, imbalanced_env, random_env):
    assert env.is_valid_move(0, 4)  # Move from tube 0 to empty tube 4
    assert env.is_valid_move(1, 5)  # Move from tube 1 to empty tube 5
    assert not env.is_valid_move(0, 0)  # Cannot move to the same tube
    assert not env.is_valid_move(0, 1)  # Cannot move to a non-empty tube
    assert not env.is_valid_move(4, 0)  # Cannot move from an empty tube

    assert imbalanced_env.is_valid_move(0, 1)  # Can move to matched color
    assert not imbalanced_env.is_valid_move(0, 4)  # Cannot move to non-matched color

    assert not random_env.is_valid_move(0, 1)
    assert random_env.is_valid_move(0, 4)

def test_is_solved(env, solved_env, imbalanced_env):
    assert env.get_is_solved() == False
    assert solved_env.get_is_solved() == True
    assert imbalanced_env.get_is_solved() == False

def test_get_valid_moves(env, solved_env, imbalanced_env):
    assert sorted(env.get_valid_moves()) == [(1, 4), (1, 5), (2, 4), (2, 5)]
    assert sorted(imbalanced_env.get_valid_moves()) == [(0, 1), (0, 5), (1, 0), (1, 5), (4, 5)]
    assert solved_env.get_valid_moves() == []

def test_get_state(env):
    """Test that get_state returns the correct NumPy array."""
    state = env.get_state()
    expected_state = np.array([
        [1, 1, 1, 1],
        [2, 3, 2, 2],
        [2, 3, 3, 3],
        [4, 4, 4, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    np.testing.assert_array_equal(state, expected_state)

def test_move(env):
    """Test the move method and its effect on state."""
    env.move(1, 4)  # Move from tube 0 to tube 4
    expected_state = np.array([
        [1, 1, 1, 1],
        [2, 3, 2, 0],
        [2, 3, 3, 3],
        [4, 4, 4, 4],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)

    # Check state
    np.testing.assert_array_equal(env.get_state(), expected_state)
    assert env.get_move_count() == True
    assert env.get_is_moved() == True

def test_moved_solved(env):
    assert env.get_is_moved() == True
    env.move(1, 4)
    assert env.get_is_moved() == True
    assert env.get_is_solved() == False
    assert env.get_is_moved() == False
    env.move(1, 4)
    assert env.get_is_solved() == False
    env.move(2, 5)
    assert env.get_is_solved() == False
    env.move(2, 5)
    assert env.get_is_solved() == False
    env.move(2, 5)
    assert env.get_is_solved() == False
    env.move(1, 5)
    assert env.is_completed_tube(5)
    assert env.get_is_solved() == False
    env.move(1, 4)
    assert env.get_is_solved() == False
    env.move(2, 4)
    assert env.is_completed_tube(4)
    assert env.get_is_solved() == True
    assert env.get_move_count() == 8

def test_is_full_tube(env, imbalanced_env):
    """Test the is_full_tube method."""
    assert env.is_full_tube(0)
    assert env.is_full_tube(1)
    assert env.is_full_tube(2)
    assert env.is_full_tube(3)
    assert not env.is_full_tube(4)
    assert not env.is_full_tube(5)

    assert not imbalanced_env.is_full_tube(0)
    assert not imbalanced_env.is_full_tube(1)
    assert imbalanced_env.is_full_tube(2)
    assert imbalanced_env.is_full_tube(3)
    assert not imbalanced_env.is_full_tube(4)
    assert not imbalanced_env.is_full_tube(5)

def test_is_empty_tube(env, imbalanced_env):
    """Test the is_empty_tube method."""
    assert not env.is_empty_tube(0)
    assert not env.is_empty_tube(1)
    assert not env.is_empty_tube(2)
    assert not env.is_empty_tube(3)
    assert env.is_empty_tube(4)
    assert env.is_empty_tube(5)

    assert not imbalanced_env.is_empty_tube(0)
    assert not imbalanced_env.is_empty_tube(1)
    assert not imbalanced_env.is_empty_tube(2)
    assert not imbalanced_env.is_empty_tube(3)
    assert not imbalanced_env.is_empty_tube(4)
    assert env.is_empty_tube(5)

def test_is_completed_tube(env, solved_env, imbalanced_env):
    """Test the is_completed_tube method."""
    assert env.is_completed_tube(0)
    assert not env.is_completed_tube(1)
    assert not env.is_completed_tube(2)
    assert env.is_completed_tube(3)
    assert not env.is_completed_tube(4)
    assert not env.is_completed_tube(5)

    assert solved_env.is_completed_tube(0)
    assert solved_env.is_completed_tube(1)
    assert solved_env.is_completed_tube(2)
    assert solved_env.is_completed_tube(3)
    assert not solved_env.is_completed_tube(4)
    assert not solved_env.is_completed_tube(5)

    assert not imbalanced_env.is_completed_tube(0)
    assert not imbalanced_env.is_completed_tube(1)
    assert imbalanced_env.is_completed_tube(2)
    assert imbalanced_env.is_completed_tube(3)
    assert not imbalanced_env.is_completed_tube(4)
    assert not imbalanced_env.is_completed_tube(5)

def test_reset(env):
    env.move(0, 5)
    env.reset()
    assert env.get_is_solved() == False
    assert env.get_is_moved() == False
    assert env.get_move_count() == 0

def test_undo_move(env):
    """Test the undo_move method and its effect on state."""
    # Initial state
    initial_state = env.get_state().copy()

    # Perform a move
    env.move(1, 4)  # Move from tube 1 to tube 4
    moved_state = env.get_state().copy()
    assert not np.array_equal(moved_state, initial_state)
    assert env.get_move_count() == 1
    assert env.get_is_moved() == True

    # Undo the move
    env.undo_move(1, 4)
    assert env.get_is_moved() == True
    assert np.array_equal(env.get_state(), initial_state)
    assert env.get_move_count() == 0
    assert env.get_is_moved() == True

def test_state_key(env):
    # Get the state key
    state_key = env.get_state_key()
    print("State Key:", state_key)

    # Ensure the state key is consistent
    assert isinstance(state_key, str)
    assert len(state_key) > 0

def test_is_recent_state_key(env):
    assert env.is_recent_state_key() is False
    for _ in range(6):
        env.move(1, 4)
        env.move(4, 1)
    assert env.is_recent_state_key() is True
    assert env.is_in_recursive_move() is False
    env.move(1, 4)
    assert env.is_recent_state_key() is True
    assert env.is_in_recursive_move() is False
    env.move(4, 1)

    # check if it is in recursive move
    for _ in range(20):
        env.move(1, 4)
        env.move(4, 1)

    assert env.is_recent_state_key() is True
    assert env.is_in_recursive_move() is True

def test_get_encoded_state(env):
    max_num_colors = 12
    num_empty_tubes = 2
    max_tube_capacity = 8

    # Encode the state
    encoded_state = env.get_encoded_state(max_num_colors, num_empty_tubes, max_tube_capacity)

    expected_encoded_state = np.array([
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
        ]], dtype=np.int8)

    assert np.array_equal(encoded_state, expected_encoded_state), f"Mismatch: {encoded_state} != {expected_encoded_state}"
