import random

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dqn import DQNAgent


class SimpleDQNAgent(DQNAgent):
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            target = (reward + self.gamma *
                      np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)  # todo tmp 1 to state
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reshape(self, state):
        return np.reshape(state, [1, self.state_size])
