from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        
    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = Transition(*args)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)