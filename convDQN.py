import random
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from dqn import DQNAgent

NUM_LAST_FRAMES = 1

class ConvDQNAgent(DQNAgent):
    def _build_model(self):
        model = Sequential()

        # Convolutions.
        model.add(Conv2D(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            # data_format='channels_first',
            input_shape=self.state_size  # (NUM_LAST_FRAMES, ) +
        ))
        model.add(Activation('relu'))
        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            # data_format='channels_first'
        ))
        model.add(Activation('relu'))

        # Dense layers.
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        print(self.action_size)
        model.add(Dense(self.action_size))

        # model.summary() # print model summary
        model.compile(RMSprop(), 'MSE')
        return model

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        input_batch = np.empty((0, 16, 16, 1))  # 8,8,2 observation space
        target_batch = np.empty((0, 4))  # 1,4 returned by model.predict
        for state, action, reward, next_state, done in minibatch:

            exp_state = np.expand_dims(state, axis=0)
            exp_next_state = np.expand_dims(next_state, axis=0)
            target = reward
            target = (reward + self.gamma *
                      np.amax(self.model.predict(exp_next_state)))
            target_f = self.model.predict(exp_state)
            target_f[0][action] = target

            input_batch = np.append(input_batch, np.expand_dims(state, axis=0), axis=0)
            target_batch = np.append(target_batch, target_f, axis=0)

        self.model.fit(input_batch, target_batch, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reshape(self, state):
        state = np.array(state)
        # state = np.expand_dims(state, axis=0)
        state = np.swapaxes(state, 0, 2)  # todo change
        return state
