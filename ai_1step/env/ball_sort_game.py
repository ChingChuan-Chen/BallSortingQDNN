from typing import List
from collections import deque
import itertools
import random


class Tube:
    def __init__(self, capacity: int = 4, balls=None):
        if balls is None:
          balls = []
        self.capacity = capacity
        if balls is None:
            self.balls = deque()
        else:
            self.balls = deque(balls)

    def is_full(self) -> bool:
        return len(self.balls) >= self.capacity

    def is_empty(self) -> bool:
        return len(self.balls) == 0

    def is_completed_tube(self) -> bool:
        return (
            len(self.balls) == self.capacity and
            all(ball == self.balls[0] for ball in self.balls)
        )

    def top_running_length(self):
        if self.is_empty():
            return 0
        top_color = self.top()
        count = 0
        for ball in reversed(self.balls):  # top is at the right (end of deque)
            if ball == top_color:
                count += 1
            else:
                break
        return count

    def top(self):
        return self.balls[-1] if not self.is_empty() else None

    def can_receive(self, ball) -> bool:
        if self.is_full():
            return False
        if self.is_empty():
            return True
        return self.top() == ball

    def push(self, ball) -> bool:
        if self.can_receive(ball):
            self.balls.append(ball)
            return True
        return False

    def pop(self):
        if not self.is_empty():
            return self.balls.pop()
        return None

    def to_list(self) -> list:
        return list(self.balls) + [0] * (self.capacity - len(self.balls))

    def __repr__(self):
        return str(self.to_list())


class BallSortGame:
    def __init__(self, num_colors: int, tube_capacity: int = 4):
        self.num_colors = num_colors
        self.tube_capacity = tube_capacity
        self.num_tubes = num_colors + 2
        self.tubes = self._generate_random_game()

    def _get_all_balls(self) -> List[int]:
        return list(itertools.chain(*[list(range(1, self.num_colors + 1)) for _ in range(self.tube_capacity)]))

    def _generate_random_game(self) -> List[Tube]:
        all_balls = self._get_all_balls()
        random.shuffle(all_balls)
        return self.set_game(all_balls)

    def set_game(self, all_balls: List[int]):
        tubes = []
        for i in range(self.num_tubes):
            if i >= self.num_colors:
                tubes.append(Tube(self.tube_capacity))
            else:
                tubes.append(Tube(self.tube_capacity, all_balls[(i*self.tube_capacity):((i+1)*self.tube_capacity)]))
        return tubes

    def reset(self):
        self.tubes = self._generate_random_game()

    def get_state(self):
        return [tube.to_list() for tube in self.tubes]

    def __repr__(self):
        return '\n'.join(f"Tube {i+1}: {tube}" for i, tube in enumerate(self.tubes))