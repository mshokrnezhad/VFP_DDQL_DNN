import numpy as np


class Memory(object):
    def __init__(self, MAX_SIZE, INPUT_SHAPE):
         # Defining base variables
        self.MEMORY_SIZE = MAX_SIZE
        # Defining complementary variables
        self.counter = 0
        self.state_memory = np.zeros((self.MEMORY_SIZE, INPUT_SHAPE), dtype=np.float32)
        self.resulted_state_memory = np.zeros((self.MEMORY_SIZE, INPUT_SHAPE), dtype=np.float32)
        self.action_memory = np.zeros(self.MEMORY_SIZE, dtype=np.int64)
        self.reward_memory = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self.terminal_memory = np.zeros(self.MEMORY_SIZE, dtype=np.bool)

    def store_transition(self, state, action, reward, resulted_state, done):
        index = self.counter % self.MEMORY_SIZE

        self.state_memory[index] = state
        self.resulted_state_memory[index] = resulted_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.counter += 1

    def sample_buffer(self, batch_size):
        MAX_MEMORY_SIZE = min(self.counter, self.MEMORY_SIZE)
        batch = np.random.choice(MAX_MEMORY_SIZE, batch_size, replace=False)
        # print(batch)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        resulted_states = self.resulted_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, resulted_states, terminals
