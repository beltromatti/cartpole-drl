from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity):
        """Initialize the replay memory.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.memory = deque(maxlen=capacity)  # Deque with fixed maximum length

    def push(self, transition):
        """Add a transition to the memory.

        Args:
            transition (tuple): Tuple of (state, action, reward, next_state, done).
        """
        self.memory.append(transition)  # Store the transition

    def sample(self, batch_size):
        """Sample a random batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            list: List of randomly sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current number of transitions in memory.

        Returns:
            int: Number of stored transitions.
        """
        return len(self.memory)