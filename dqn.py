import random
import gym
import abc
import sys
sys.path.append('../snake_gym')  # x D
import gym_snake
import numpy as np
from collections import deque


class DQNAgent:
    __metaclass__ = abc.ABCMeta

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self._build_model()

    @abc.abstractmethod
    def _build_model(self):
        return

    def remember(self, state, action, reward, next_state, done):
        state = self.reshape(state)
        next_state = self.reshape(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = self.reshape(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    @abc.abstractmethod
    def replay(self, batch_size):
        return

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    @abc.abstractmethod
    def reshape(self, state):
        return
