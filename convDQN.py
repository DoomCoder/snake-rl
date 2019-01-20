import random
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from dqn import DQNAgent

NUM_LAST_FRAMES = 3


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

    def soft_update(self, tau=0.001):
        model_weights = np.array(self.model.get_weights())
        target_weights = np.array(self.target_model.get_weights())
        weights = [model_weights, target_weights]

        new_weights = list()

        for weights_list_tuple in zip(*weights):
            new_weights.append(
                [np.average(np.array(weights_),axis=0, weights=[tau, 1-tau]) \
                 for weights_ in zip(*weights_list_tuple)])

        # new_weights = list()
        #
        # new_weights.append([tau*model_weights+(1-tau)*target_weights])

        self.target_model.set_weights(new_weights)

    def replay(self, batch_size):
        # sorted_memory = sorted(self.memory, key=lambda x: abs(x[2]), reverse=True)
        # p = np.array([0.2 ** i for i in range(len(sorted_memory))])
        # # @todo make this parameter a field in the class
        # p = p / sum(p)
        # sample_idxs = np.random.choice(np.arange(len(sorted_memory)), size=batch_size, p=p)
        # minibatch = [sorted_memory[idx] for idx in sample_idxs]

        minibatch = random.sample(self.memory, batch_size)

        input_batch = np.empty((0,) + (NUM_LAST_FRAMES, ) + self.state_shape[1:])
        target_batch = np.empty((0, self.action_size))
        # todo could be vectorized
        for states, action, reward, next_states, done in minibatch:
            exp_next_states = np.expand_dims(next_states, axis=0)
            exp_states = np.expand_dims(states, axis=0)

            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(exp_next_states)))
            target_f = self.model.predict(exp_states)
            target_f[0][action] = target

            input_batch = np.append(input_batch, exp_states, axis=0)
            target_batch = np.append(target_batch, target_f, axis=0)

        self.model.fit(input_batch, target_batch, verbose=0)
        # self.soft_update()

    def act(self, states):
        # print(states[-1])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        exp_states = np.expand_dims(states, axis=0)
        act_values = self.model.predict(exp_states)
        # print(act_values[0])
        # print(np.argmax(act_values[0]))
        return np.argmax(act_values[0])  # returns action
