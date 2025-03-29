from typing import List, Tuple
from .ball_sort_game import BallSortGame


class BallSortEnv:
    def __init__(self, num_colors: int, tube_capacity: int = 4):
        self.num_colors = num_colors
        self.tube_capacity = tube_capacity
        self.num_tubes = num_colors + 2
        self.max_steps = num_colors * 50
        self.step_count = 0
        self.game = BallSortGame(self.num_colors, self.tube_capacity)
        self.reset()

    def reset(self):
        self.game.reset()
        self.step_count = 0
        return self.get_state()

    def set_game(self, all_balls: List[int]):
        self.game.set_game(all_balls)

    def get_state(self) -> List[List[int]]:
        return [tube.to_list() for tube in self.game.tubes]

    def is_valid_move(self, src: int, dst: int) -> bool:
        if src == dst:
            return False
        if self.game.tubes[src].is_completed_tube():
            return False
        if self.game.tubes[src].is_empty():
            return False
        ball = self.game.tubes[src].top()
        return self.game.tubes[dst].can_receive(ball)

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        actions = []
        for i in range(self.num_tubes):
            if self.game.tubes[i].is_empty() or self.game.tubes[i].is_completed_tube():
                continue
            for j in range(self.num_tubes):
                if i != j and self.is_valid_move(i, j):
                    actions.append((i, j))
        return actions

    def is_solved(self) -> bool:
        empty_tube_count = 0
        completed_tube_count = 0
        for tube in self.game.tubes:
            if tube.is_empty():
                empty_tube_count += 1
            if tube.is_completed_tube():
                completed_tube_count += 1
        return (empty_tube_count == 2) and (completed_tube_count == self.num_colors)

    def step(self, action: Tuple[int, int]) -> Tuple[List[List[int]], float, bool]:
        src, dst = action
        reward = -0.1  # small step penalty
        done = False

        if not self.is_valid_move(src, dst):
            reward = -5.0
            self.step_count += 1
            return self.get_state(), reward, done

        before_streak = self.game.tubes[dst].top_running_length()

        ball = self.game.tubes[src].pop()
        self.game.tubes[dst].push(ball)

        after_streak = self.game.tubes[dst].top_running_length()
        if after_streak > before_streak:
            reward += 0.5  # partial reward for extending color streak

        # Bonus for completing a tube
        if self.game.tubes[dst].is_completed_tube():
            reward += 2.0

        if self.is_solved():
            reward = 10.0
            done = True

        self.step_count += 1
        if self.step_count >= self.max_steps and not done:
            reward = -5.0  # penalty for not solving in time
            done = True

        return self.get_state(), reward, done
