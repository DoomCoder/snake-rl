import random
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from dqn import DQNAgent

NUM_LAST_FRAMES = 5


class ConvDQNAgent(DQNAgent):
    def _build_model(self):
        model = Sequential()

        # Convolutions.
        model.add(Conv2D(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first',
            input_shape=(NUM_LAST_FRAMES, ) + self.state_shape[1:]
        ))
        model.add(Activation('relu'))
        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first'
        ))
        model.add(Activation('relu'))

        # Dense layers.
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))

        # model.summary() # print model summary
        model.compile(RMSprop(), 'MSE')
        return model

    def replay(self, batch_size):
        memory = np.array(self.memory)
        minibatch = random.sample(self.memory, batch_size)
        minibatch_indices = np.random.randint(high=memory.shape[0], low=NUM_LAST_FRAMES, size=batch_size)
        input_batch = np.empty((0,) + (NUM_LAST_FRAMES, ) + self.state_shape[1:])
        target_batch = np.empty((0, self.action_size))
        # todo could be vectorized
        for state_idx in minibatch_indices:
            state, action, reward, next_state, done = memory[state_idx]
            last_observations = memory[state_idx-NUM_LAST_FRAMES:state_idx-1]
            last_states = np.array([obs[0] for obs in last_observations])
            last_states = last_states[:, 0, :, :]
            # exp_state = np.expand_dims(state, axis=0)
            states = np.append(last_states, state, axis=0)
            exp_next_state = np.expand_dims(next_state, axis=0)
            next_states = np.append(states[1:], state, axis=0)
            exp_next_states = np.expand_dims(next_states, axis=0)
            exp_states = np.expand_dims(states, axis=0)
            target = (reward + self.gamma *
                      np.amax(self.model.predict(exp_next_states)))
            target_f = self.model.predict(exp_states)
            target_f[0][action] = target
            input_batch = np.append(input_batch, exp_states, axis=0)
            target_batch = np.append(target_batch, target_f, axis=0)

        self.model.fit(input_batch, target_batch, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        state = self.reshape(state)
        if np.random.rand() <= self.epsilon or len(self.memory) < NUM_LAST_FRAMES:
            return random.randrange(self.action_size)

        memory = np.array(self.memory)
        last_observations = memory[-NUM_LAST_FRAMES:-1]
        last_states = np.array([obs[0] for obs in last_observations])
        last_states = last_states[:, 0, :, :]

        states = np.append(last_states, state, axis=0)

        exp_states = np.expand_dims(states, axis=0)
        act_values = self.model.predict(exp_states)
        return np.argmax(act_values[0])  # returns action
