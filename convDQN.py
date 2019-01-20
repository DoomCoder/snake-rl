import random
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
            # shape: (NUM_LAST_FRAMES, H, W)
            input_shape=(self.num_last_observations,) + self.state_shape
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

        model.summary() # print model summary
        model.compile(RMSprop(), 'MSE')
        return model

    def replay(self, batch_size):
        memory_sample = random.sample(self.memory, batch_size)
        np_memory_sample = np.array(memory_sample)

        states_arr = np.stack(np_memory_sample[:,0])
        actions_arr = np_memory_sample[:,1].astype(np.int)
        rewards_arr = np_memory_sample[:,2]
        next_states_arr = np.stack(np_memory_sample[:, 3])
        done_arr = np_memory_sample[:,4].astype(int)

        updated_targets = rewards_arr + self.gamma * np.amax(self.model.predict(next_states_arr)) * (1-done_arr)
        actions_targets = self.model.predict(states_arr)
        actions_targets[np.arange(actions_arr.shape[0]), actions_arr] = updated_targets
        self.model.train_on_batch(states_arr, actions_targets)

        # NON VECTORIZED & more readable version
        # input_batch = []
        # target_batch = []
        # for state, action, reward, next_state, done in memory_sample:
        #     target = (reward + done * self.gamma *
        #               np.amax(self.model.predict(np.expand_dims(next_state, axis=0))))
        #     target_f = self.model.predict(np.expand_dims(state, 0))
        #     target_f[0][action] = target
        #
        #     input_batch.append(state)
        #     target_batch.append(target_f[0])
        # self.model.train_on_batch(np.array(input_batch), np.array(target_batch))

        # TEST indicating that vectorized version == non vectorized
        # assert (np.array_equal(states_arr, np.array(input_batch)))
        # assert (np.array_equal(target_batch, np.array(target_batch)))

    def train(self, env, batch_size, n_episodes, exploration_phase_size, report_freq, save_freq=1000):
        reporter = Reporter(report_freq, n_episodes)
        # calc constant epsilon decay based on exploration_phase_size
        # (the percentage of the training process at which the exploration rate should reach its minimum)
        self.epsilon_decay = ((self.epsilon - self.epsilon_min) / (n_episodes * exploration_phase_size))
        n_observations = 0
        for e in range(n_episodes):
            self.observations = None
            observation = env.reset()
            done = False
            steps = 0
            reward_sum = 0
            state = self.get_last_observations(observation)
            while not done:
                action = self.act(state)
                new_observation, has_eaten, done, _ = env.step(action)
                # rewards can be changed here
                if done:
                    reward = -1
                elif has_eaten:
                    reward = len(env.game.snake.body)
                else:
                    reward = 0

                reward_sum += reward
                next_state = self.get_last_observations(new_observation)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                steps += 1
                n_observations += 1
                if done:
                    reporter.remember(steps, len(env.game.snake.body), reward_sum, self.epsilon)
                    if reporter.wants_to_report():
                        print(reporter.get_report_str())

                    break

                if len(self.memory) > batch_size:
                    self.replay(batch_size)

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

            if e % save_freq == 0:
                self.save("./SNEK-dqn.h5")
