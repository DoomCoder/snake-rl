import random
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import RMSprop
from dqn import DQNAgent

NUM_LAST_FRAMES = 1


class ConvDQNAgent(DQNAgent):
    def _build_model(self):
        model = Sequential()

        # Convolutions.
        model.add(Conv2D(
            16,
            kernel_size=(2, 2),
            strides=(1, 1),
            # data_format='channels_first',
            input_shape=self.state_size  # (NUM_LAST_FRAMES, ) +
        ))
        model.add(Activation('relu'))
        model.add(Conv2D(
            32,
            kernel_size=(2, 2),
            strides=(1, 1),
            # data_format='channels_first'
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
        minibatch = random.sample(self.memory, batch_size)
        xs_batch = []
        ys_batch = []

        for state, action, reward, next_state, done in minibatch:
            exp_state = np.expand_dims(state, axis=0)
            # print(exp_state)
            exp_next_state = np.expand_dims(next_state, axis=0)
            target = reward
            target = (reward + self.gamma *
                      np.amax(self.model.predict(exp_next_state)))
            target_f = self.model.predict(exp_state)
            target_f[0][action] = target

            xs_batch.append(state)
            ys_batch.append(target_f[0])

        xs_batch =  np.array(xs_batch)
        ys_batch = np.array(ys_batch)
        self.model.fit(xs_batch, ys_batch, verbose=2)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reshape(self, state):
        state = np.array(state)
        # state = np.expand_dims(state, axis=0)
        state = np.swapaxes(state, 0, 2)  # todo change
        return state
