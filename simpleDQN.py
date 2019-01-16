import random
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dqn import DQNAgent


class SimpleDQNAgent(DQNAgent):
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        state_size = np.prod(self.state_shape)

        model = Sequential()
        model.add(Dense(24, input_dim=state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        xs_batch = []
        ys_batch = []
        for state, action, reward, next_state, done in minibatch:
            exp_state = np.expand_dims(state, axis=0)
            exp_next_state = np.expand_dims(state, axis=0)
            target = (reward + self.gamma *
                      np.amax(self.model.predict(exp_next_state)))
            target_f = self.model.predict(exp_state)
            target_f[0][action] = target
            xs_batch.append(state)
            ys_batch.append(target_f[0])

        self.model.fit(np.array(xs_batch), np.array(ys_batch), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reshape(self, state):
        return state.flatten()
