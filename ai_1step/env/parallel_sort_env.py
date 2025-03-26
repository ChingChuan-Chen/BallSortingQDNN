import numpy as np


class ParallelBallSortEnv:
    def __init__(self, n_envs: int, num_colors: int, tube_capacity: int = 4, num_empty_tubes: int = 2):
        self.n_envs = n_envs
        self.num_colors = num_colors
        self.tube_capacity = tube_capacity
        self.num_empty_tubes = num_empty_tubes
        self.num_tubes = num_colors + num_empty_tubes
        self.max_steps = num_colors * 75
        self.max_difficulty = self.num_colors + 2.0 * (self.tube_capacity - 1)**2 / self.tube_capacity

        self.state_shape = (n_envs, self.num_tubes, self.tube_capacity)
        self.state = np.zeros(self.state_shape, dtype=np.int8)
        self.empty_slots = self.tube_capacity - np.count_nonzero(self.state, axis=2)
        self.step_counts = np.zeros(n_envs, dtype=np.int32)
        self.dones = np.zeros(n_envs, dtype=bool)
        self.difficulty_scores = np.zeros(n_envs, dtype=np.float32)
        self.reset()

    def reset(self):
        self.state[:] = 0
        self.dones[:] = False
        self.step_counts[:] = 0
        for env_idx in range(self.n_envs):
            balls = np.repeat(np.arange(1, self.num_colors + 1), self.tube_capacity)
            np.random.shuffle(balls)
            filled = self.num_tubes - self.num_empty_tubes
            self.state[env_idx, :filled] = balls.reshape((filled, self.tube_capacity))
        self.update_empty_slots()
        self.update_difficulty_scores()
        return self.state.copy()

    def update_empty_slots(self):
        self.empty_slots = self.tube_capacity - np.count_nonzero(self.state, axis=2)

    def update_difficulty_scores(self):
        for env_idx in range(self.n_envs):
            # calculate mixed tubes and tube_scores
            mixed_tubes = 0
            top_balls = np.bincount(self.state[env_idx, :, self.tube_capacity-1])
            tube_scores = np.zeros(self.num_tubes, dtype=np.float32)
            for tube_idx in range(self.num_tubes):
                if self.is_empty_tube(env_idx, tube_idx) or self.is_tube_complete(env_idx, tube_idx):
                    continue
                tube = self.state[env_idx, tube_idx]
                if np.sum(np.bincount(tube[tube > 0]) > 0) > 1:
                    mixed_tubes += 1

                longest_streak = self.get_top_color_streak(env_idx, tube_idx)
                top_balls_with_same_color_count = top_balls[self.state[env_idx, tube_idx, self.tube_capacity-1]]
                tube_scores[tube_idx] = (self.tube_capacity - longest_streak) ** 2 + (self.tube_capacity - top_balls_with_same_color_count) ** 2
                tube_scores[tube_idx] /= self.tube_capacity

            # Compute actual difficulty score
            difficulty_score = mixed_tubes + np.sum(tube_scores) / self.num_colors

            # Normalize difficulty score between [0,1]
            self.difficulty_scores[env_idx] = difficulty_score / self.max_difficulty

    def top_index(self, env_idx: int, tube_idx: int):
        top_idx = self.tube_capacity - self.empty_slots[env_idx, tube_idx] - 1
        return top_idx

    def is_full_tube(self, env_idx: int, tube_idx: int):
        return self.empty_slots[env_idx, tube_idx] == 0

    def is_empty_tube(self, env_idx: int, tube_idx: int):
        return self.empty_slots[env_idx, tube_idx] == self.tube_capacity

    def get_top_color_streak(self, env_idx: int, tube_idx: int):
        top_idx = self.top_index(env_idx, tube_idx)
        if top_idx == -1: # If tube is empty
            return 0
        longest_streak = 1
        top_color = self.state[env_idx, tube_idx, top_idx]
        # Count continuous occurrences of the top color from the rightmost (top) position
        for i in range(top_idx - 1, -1, -1):
            if self.state[env_idx, tube_idx, i] == top_color:
                longest_streak += 1
            else:
                break
        return longest_streak

    def is_tube_complete(self, env_idx: int, tube_idx: int):
        tube = self.state[env_idx, tube_idx]
        if np.count_nonzero(tube) != self.tube_capacity:
            return False
        return np.all(tube == tube[0])

    def can_move(self, env_idx: int, src: int, dst: int):
        if src == dst or self.dones[env_idx]:
            return False
        if self.is_empty_tube(env_idx, src):
            return False
        if self.is_tube_complete(env_idx, src):
            return False
        if self.is_full_tube(env_idx, dst):
            return False
        if self.is_empty_tube(env_idx, dst):
            return True
        src_ball_top_idx = self.top_index(env_idx, src)
        dst_ball_top_idx = self.top_index(env_idx, dst)
        return self.state[env_idx, src, src_ball_top_idx] == self.state[env_idx, dst, dst_ball_top_idx]

    def step(self, actions):
        next_states = np.zeros_like(self.state)
        rewards = np.full(self.n_envs, -0.1, dtype=np.float32)
        dones = np.zeros_like(self.dones)

        for env_idx, (src, dst) in enumerate(actions):
            if self.dones[env_idx] or (src == -1 and dst == -1):
                next_states[env_idx] = self.state[env_idx]
                rewards[env_idx] = 0.0
                dones[env_idx] = True
                continue

            if not self.can_move(env_idx, src, dst):
                rewards[env_idx] = -5.0 * (1.0 - 0.75 * self.difficulty_scores[env_idx])
                next_states[env_idx] = self.state[env_idx]
                self.step_counts[env_idx] += 1
                continue

            src_ball_top_idx = self.top_index(env_idx, src)
            if self.empty_slots[env_idx, dst] == 0:
                dst_ball_top_idx = -1
            else:
                dst_ball_top_idx = self.top_index(env_idx, dst)
            ball = self.state[env_idx, src, src_ball_top_idx]
            self.state[env_idx, src, src_ball_top_idx] = 0
            insert_idx = dst_ball_top_idx + 1 if dst_ball_top_idx != -1 else 0
            self.state[env_idx, dst, insert_idx] = ball

            if self.is_solved(env_idx):
                rewards[env_idx] = 50.0
                dones[env_idx] = True

            next_states[env_idx] = self.state[env_idx]

            self.step_counts[env_idx] += 1
            if self.step_counts[env_idx] >= self.max_steps and not dones[env_idx]:
                rewards[env_idx] = -5.0 * (1.0 - 0.75 * self.difficulty_scores[env_idx])
                dones[env_idx] = True

        self.update_empty_slots()
        for env_idx, (src, dst) in enumerate(actions):
            if not dones[env_idx]:
                if self.is_tube_complete(env_idx, dst):
                    rewards[env_idx] += 4.0
                else:
                    streak_dst = self.get_top_color_streak(env_idx, dst)
                    for i in range(1, self.tube_capacity - 1):
                        if self.empty_slots[env_idx, dst] == i and streak_dst == self.tube_capacity - i:
                            rewards[env_idx] += 2.0 / i
                        elif streak_dst == self.tube_capacity - i:
                            rewards[env_idx] += 1.0 / i

                streak_src = self.get_top_color_streak(env_idx, src)
                for i in range(1, self.tube_capacity - 1):
                    if self.empty_slots[env_idx, src] == i and streak_src == self.tube_capacity - i:
                        rewards[env_idx] += 2.5 / i
                    elif streak_src == self.tube_capacity - i:
                        rewards[env_idx] += 1.25 / i

                # rewards if it fills into a empty tube
                if self.empty_slots[env_idx, dst] == self.tube_capacity - 1:
                    rewards[env_idx] += 0.5

                # rewards if it clears out a tube
                if self.empty_slots[env_idx, src] == self.tube_capacity:
                    rewards[env_idx] += 5.0

        self.dones[:] = dones
        return next_states, rewards, dones

    def is_solved(self, env_idx: int):
        empty_tube_count = np.sum(self.empty_slots[env_idx] == self.tube_capacity)
        if empty_tube_count != self.num_empty_tubes:
            return False
        for i in np.argwhere(self.empty_slots[env_idx] == 0):
            if not np.all(self.state[env_idx, i] == self.state[env_idx, i, 0]):
                return False
        return True

    def get_valid_actions(self, env_idx: int):
        actions = []
        for i in range(self.num_tubes):
            for j in range(self.num_tubes):
                if (i != j) and self.can_move(env_idx, i, j):
                    actions.append((i, j))
        return actions

    def get_all_valid_actions(self):
        return [self.get_valid_actions(i) for i in range(self.n_envs)]
