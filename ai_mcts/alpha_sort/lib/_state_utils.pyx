import numpy as np
cimport numpy as cnp

def state_encode(cnp.ndarray[cnp.int8_t, ndim=2] state, int max_num_colors, int num_empty_tubes, int max_tube_capacity):
    """
    Encode a single state into a one-hot representation.

    Args:
        state (np.ndarray): State to encode of shape (max_num_colors + num_empty_tubes, max_tube_capacity).
        max_num_colors (int): Maximum number of colors.
        num_empty_tubes (int): Number of empty tubes.
        max_tube_capacity (int): Maximum capacity of each tube.

    Returns:
        np.ndarray: Encoded state as a one-hot representation of shape
                    (max_num_colors + 1, max_num_colors + num_empty_tubes, max_tube_capacity).
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=3] encoded_state = np.zeros(
        (max_num_colors + 1, max_num_colors + num_empty_tubes, max_tube_capacity),
        dtype=np.float32
    )
    cdef int tube_idx, pos_idx, ball
    for tube_idx in range(state.shape[0]):
        for pos_idx in range(state.shape[1]):
            ball = state[tube_idx, pos_idx]
            if ball >= 0:
                encoded_state[ball, tube_idx, pos_idx] = 1.0
    return encoded_state

def state_decode(cnp.ndarray[cnp.float32_t, ndim=3] encoded_state, int num_colors, int num_empty_tubes, int tube_capacity):
    """
    Decode a one-hot encoded state back to the original state representation.

    Args:
        encoded_state (np.ndarray): One-hot encoded state of shape
                                    (num_colors + 1, num_colors + num_empty_tubes, tube_capacity).
        num_colors (int): Number of colors.
        num_empty_tubes (int): Number of empty tubes.
        tube_capacity (int): Capacity of each tube.

    Returns:
        np.ndarray: Decoded state of shape (num_colors + num_empty_tubes, tube_capacity).
    """
    cdef cnp.ndarray[cnp.int8_t, ndim=2] state = np.zeros(
        (num_colors + num_empty_tubes, tube_capacity),
        dtype=np.int8
    )
    cdef int tube_idx, pos_idx, color
    for tube_idx in range(state.shape[0]):
        for pos_idx in range(state.shape[1]):
            color = np.argmax(encoded_state[:, tube_idx, pos_idx])
            state[tube_idx, pos_idx] = color
    return state
