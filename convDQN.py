import random
from collections import deque

import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from dqn import DQNAgent
from reporter import Reporter


class ConvDQNAgent(DQNAgent):
    def _build_model(self):
        model = Sequential()

        # Convolutions.
        model.add(Conv2D(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first',
            input_shape=(self.num_last_frames, ) + self.state_shape[1:]  # (NUM_LAST_FRAMES, H, W)
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
        minibatch = random.sample(self.memory, batch_size)

        input_batch = np.empty((0,) + (self.num_last_frames, ) + self.state_shape[1:])
        target_batch = np.empty((0, self.action_size))
        # todo could be vectorized
        for states, action, reward, next_states, done in minibatch:
            exp_next_states = np.expand_dims(next_states, axis=0)
            exp_states = np.expand_dims(states, axis=0)

            target = (reward + self.gamma *
                      np.amax(self.model.predict(exp_next_states)))
            target_f = self.model.predict(exp_states)
            target_f[0][action] = target

            input_batch = np.append(input_batch, exp_states, axis=0)
            target_batch = np.append(target_batch, target_f, axis=0)

        self.model.fit(input_batch, target_batch, verbose=0)

    def act(self, states):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        exp_states = np.expand_dims(states, axis=0)
        act_values = self.model.predict(exp_states)
        return np.argmax(act_values[0])  # returns action

    def get_last_observations(self, observation):
        if self.observations is None:
            self.observations = deque([observation] * self.num_last_frames)
        else:
            self.observations.append(observation)
            self.observations.popleft()

        return np.array(self.observations)
        # return np.expand_dims(self.frames, axis=0)

    def train(self, env, batch_size, n_episodes, exploration_phase_size):
        reporter = Reporter(batch_size, n_episodes)
        # calc constant epsilon decay based on exploration_phase_size
        # (the percentage of the training process at which the exploration rate should reach its minimum)
        self.epsilon_decay = ((self.epsilon - self.epsilon_min) / (n_episodes * exploration_phase_size))
        for e in range(n_episodes):
            observation = env.reset()
            observation = observation[0]

            done = False
            steps = 0
            reward_sum = 0
            self.observations = None
            while not done:
                state = self.get_last_observations(observation)
                action = self.act(state)
                next_observation, has_eaten, done, _ = env.step(action)
                next_observation = next_observation[0]

                # rewards can be changed here
                if done:
                    reward = -1
                elif has_eaten:
                    reward = 1
                else:
                    reward = 0

                reward_sum += reward
                next_state = self.get_last_observations(next_observation)
                self.remember(state, action, reward, next_state, done)
                observation = next_observation
                steps += 1
                if done:
                    reporter.remember(steps, len(env.game.snake.body), reward_sum, self.epsilon)
                    if reporter.wants_to_report():
                        print(reporter.get_report_str())

                    break

                if len(self.memory) > batch_size and (e % batch_size == 0):
                    self.replay(batch_size)

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

            if e % 1000 == 0:
                self.save("./SNEK-dqn600k.h5")
