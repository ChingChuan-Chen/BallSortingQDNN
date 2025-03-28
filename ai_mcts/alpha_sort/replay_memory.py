import random
from collections import deque


class ReplayMemory:
    """Replay memory for storing transitions."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of the replay memory."""
        return len(self.buffer)
