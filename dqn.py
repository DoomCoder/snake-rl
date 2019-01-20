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

    def __init__(self, state_shape, action_size, num_last_observations):
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_last_observations = num_last_observations
        self.observations = None
        self.epsilon_decay = None

        self.memory = deque(maxlen=10**4)
        self.gamma = 0.6  # discount rate
        self.epsilon_max = 1.0  # epsilon == exploration rate
        self.epsilon_min = 0.05
        self.epsilon = self.epsilon_max
        self.q_learning_rate = 0.1
        self.model = self._build_model()


    @abc.abstractmethod
    def _build_model(self):
        return

    def get_last_observations(self, observation):
        if self.observations is None:
            self.observations = deque([observation] * self.num_last_observations)
        else:
            self.observations.append(observation)
            self.observations.popleft()

        return np.array(self.observations)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(np.expand_dims(state, 0))
        return np.argmax(act_values[0])  # returns action

    def remember(self, state, action, reward, next_state, done):
        state = self.reshape(state)
        next_state = self.reshape(next_state)
        self.memory.append((state, action, reward, next_state, done))

    @abc.abstractmethod
    def replay(self, batch_size):
        return

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    @abc.abstractmethod
    def reshape(self, state):
        return state
