import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cython cimport boundscheck, wraparound

cdef class C_BallSortEnv:
    cdef np.int8_t[:, ::1] state  # Memory view for state
    cdef np.int8_t[:] num_balls_per_tube  # Memory view for number of balls in each tube
    cdef int tube_capacity
    cdef int num_colors
    cdef int num_empty_tubes
    cdef int num_tubes

    def __init__(self, int tube_capacity, int num_colors, int num_empty_tubes, np.ndarray[np.int8_t, ndim=2] state, np.ndarray[np.int8_t] num_balls_per_tube):
        self.tube_capacity = tube_capacity
        self.num_colors = num_colors
        self.num_empty_tubes = num_empty_tubes
        self.num_tubes = num_colors + num_empty_tubes

        # Allocate memory for state and num_balls_per_tube
        self.state = state
        self.num_balls_per_tube = num_balls_per_tube

    def top_index(self, tube_idx: int) -> int:
        if tube_idx < 0 or tube_idx >= self.num_tubes:
            return -1
        return self.num_balls_per_tube[tube_idx] - 1

    def get_top_color_streak(self, tube_idx: int) -> int:
        cdef int top_idx = self.top_index(tube_idx)
        if top_idx == -1:  # If the tube is empty
            return 0
        cdef int longest_streak = 1
        cdef int top_color = self.state[tube_idx, top_idx]
        for i in range(top_idx - 1, -1, -1):
            if self.state[tube_idx, i] == top_color:
                longest_streak += 1
            else:
                break
        return longest_streak

    def is_valid_move(self, src: int, dst: int) -> bint:
        if src == dst:
            return 0
        if self.is_empty_tube(src) == 1:
            return 0
        if self.is_full_tube(dst) == 1:
            return 0

        dst_top_idx = self.top_index(dst)
        if dst_top_idx == -1:
            return 1

        src_top_idx = self.top_index(src)
        return self.state[src, src_top_idx] == self.state[dst, dst_top_idx]

    def is_full_tube(self, tube_idx: int) -> bint:
        if tube_idx < 0 or tube_idx >= self.num_tubes:
            return 0
        return self.num_balls_per_tube[tube_idx] == self.tube_capacity

    def is_empty_tube(self, tube_idx: int) -> bint:
        if tube_idx < 0 or tube_idx >= self.num_tubes:
            return 0
        return self.num_balls_per_tube[tube_idx] == 0

    def is_solved(self) -> bint:
        cdef int i, j
        cdef int empty_tubes = 0
        for i in range(self.num_tubes):
            if self.is_empty_tube(i) != 1:
                for j in range(1, self.tube_capacity):
                    if self.state[i, j] != self.state[i, 0]:
                        return 0
            else:
                empty_tubes += 1
        return empty_tubes == self.num_empty_tubes

    def get_valid_actions(self):
        cdef int i, j
        actions = []
        for i in range(self.num_tubes):
            for j in range(self.num_tubes):
                if i != j and self.is_valid_move(i, j) == 1:
                    actions.append((i, j))
        return actions

    def move(self, src: int, dst: int):
        if not self.is_valid_move(src, dst):
            raise ValueError(f"Cannot move from tube {src} to tube {dst}")

        # Get the top indices of the source and destination tubes
        src_top_idx = self.top_index(src)
        dst_top_idx = self.top_index(dst)

        # Move the ball
        self.state[dst, dst_top_idx + 1] = self.state[src, src_top_idx]
        self.state[src, src_top_idx] = 0

        # Update the number of balls in each tube
        self.num_balls_per_tube[src] -= 1
        self.num_balls_per_tube[dst] += 1

    def get_state(self) -> np.ndarray:
        return np.array(self.state, copy=True)

    def get_num_balls_per_tube(self) -> np.ndarray:
        return np.array(self.num_balls_per_tube, copy=True)
