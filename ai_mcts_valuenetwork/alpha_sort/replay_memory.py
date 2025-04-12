import random
from collections import deque
import numpy as np


class ReplayMemory:
    # Initialize a random number generator
    rng = np.random.default_rng(random.randint(0, 2**31 - 1))

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        """Return the current size of the replay memory."""
        return len(self.buffer)
