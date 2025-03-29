import numpy as np
from alpha_sort.lib._state_utils import state_encode, state_decode

def test_state_encode_decode():
    max_num_colors = 12
    num_empty_tubes = 2
    max_tube_capacity = 8

    # Example state
    state = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 0],
        [3, 3, 0, 0],
        [3, 3, 2, 0],
        [0, 0, 0, 0]
    ], dtype=np.int8)

    # Encode the state
    encoded_state = state_encode(state, max_num_colors, num_empty_tubes, max_tube_capacity)

    # Decode the state
    num_colors = 3
    tube_capacity = 4
    decoded_state = state_decode(encoded_state, num_colors, num_empty_tubes, tube_capacity)
    print("decoded_state:\n", decoded_state)

    # Verify that the decoded state matches the original state
    assert np.array_equal(state, decoded_state), f"Mismatch: {state} != {decoded_state}"

def test_full_tubes():
    max_num_colors = 12
    num_empty_tubes = 2
    max_tube_capacity = 8

    # State with all tubes full of the same color
    state = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.int8)

    # Encode and decode
    encoded_state = state_encode(state, max_num_colors, num_empty_tubes, max_tube_capacity)
    num_colors = 4
    tube_capacity = 4
    decoded_state = state_decode(encoded_state, num_colors, num_empty_tubes, tube_capacity)

    # Verify
    assert np.array_equal(state, decoded_state), f"Mismatch: {state} != {decoded_state}"
