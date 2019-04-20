import os
import random
import numpy as np
from dqn import DQNAgent
from reporter import Reporter
import snake_logger
import tensorflow as tf
tfe = tf.contrib.eager
from collections import deque


qLogger=snake_logger.QLogger()


class ConvDDPGAgent(DQNAgent):
    def __init__(self, state_shape, action_size, num_last_observations, loss_logging=True, rho=1e-4):
        tf.enable_eager_execution()
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_last_observations = num_last_observations
        self.observations = None
        self.epsilon_decay = None
        self.rho = rho

        self.memory = deque(maxlen=10**4)
        self.gamma = 0.9  # discount rate
        self.epsilon_max = 1.0  # epsilon == exploration rate
        self.epsilon_min = 0.05
        self.epsilon = self.epsilon_max
        self.q_learning_rate = 0.1
        self.q_model = self._build_q_model()
        self.target_q_model = self._build_q_model()
        self.policy_model = self._build_policy_model()
        self.target_policy_model = self._build_policy_model()
        self.q_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.policy_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.target_policy_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        self.target_q_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        # if loss_logging:
        #     self.model.train_on_batch = snake_logger.loss_logger_decorator(self.model.train_on_batch)

    def _build_policy_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, [3, 3], data_format='channels_first',
                                   strides=[1, 1], activation='relu',
                                   input_shape=(self.num_last_observations,) + self.state_shape),
            tf.keras.layers.Conv2D(16, [3, 3], data_format='channels_first',
                                   strides=[1, 1], activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax'),
        ])

        return model

    def _build_q_model(self):
        q_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, [3, 3], data_format='channels_first',
                                   strides=[1, 1], activation='relu',
                                   input_shape=(self.num_last_observations,) + self.state_shape),
            tf.keras.layers.Conv2D(16, [3, 3], data_format='channels_first',
                                   strides=[1, 1], activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='relu'),
        ])

        return q_model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state=tf.cast(state, tf.float32)
        act_values = self.policy_model(np.expand_dims(state, 0))
        return np.argmax(act_values[0])  # returns action

    def average_weights(self, weights):
        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(
                [np.array(weights_).mean(axis=0) \
                 for weights_ in zip(*weights_list_tuple)])

        return new_weights

    def update_targets(self):
        self.target_policy_model.set_weights(list([w1*(1-self.rho)+self.rho*w2 for w1, w2 in zip(self.target_policy_model.get_weights(), self.policy_model.get_weights())]))
        self.target_q_model.set_weights(list([w1*(1-self.rho)+self.rho*w2 for w1, w2 in zip(self.target_q_model.get_weights(), self.q_model.get_weights())]))


    def grad_q(self, states_arr, ys):
        with tf.GradientTape() as q_tape:
            q_values = self.q_model(states_arr, training=True)
            q_loss = tf.reduce_mean(tf.square(q_values - ys))

        q_grads = q_tape.gradient(q_loss, self.q_model.trainable_weights)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_model.trainable_weights),
                                  global_step=tf.train.get_or_create_global_step())

    def replay(self, batch_size):
        memory_sample = random.sample(self.memory, batch_size)
        np_memory_sample = np.array(memory_sample)

        states_arr = tf.cast(np.stack(np_memory_sample[:, 0]), tf.float32)
        actions_arr = tf.cast(np_memory_sample[:, 1].astype(np.int), tf.int64)
        rewards_arr = tf.cast(np_memory_sample[:, 2].astype(np.float), tf.float32)
        next_states_arr = tf.cast(np.stack(np_memory_sample[:, 3]).astype(np.float64), tf.float32)
        done_arr = np_memory_sample[:, 4].astype(np.int)


        tp_preds = self.target_policy_model(next_states_arr)
        tp_actions = np.argmax(tp_preds, axis=1)
        qs = self.target_q_model(next_states_arr).numpy()
        q_corrections = rewards_arr + self.gamma * (1-done_arr) * qs[np.arange(64), tp_actions]
        ys = qs
        for i, (ac, corr) in enumerate(zip(tp_actions, q_corrections)):
            ys[i][ac] += corr

        with tf.GradientTape() as q_tape:
            q_values = self.q_model(states_arr, training=True)
            q_loss = tf.reduce_mean(tf.square(q_values - ys))

        q_grads = q_tape.gradient(q_loss, self.q_model.trainable_weights)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_model.trainable_weights),
                                  global_step=tf.train.get_or_create_global_step())

        with tf.GradientTape() as p_tape:
            policy_actions = np.argmax(self.policy_model(states_arr, training=True), axis=1)
            actions_policy = self.policy_model(states_arr, training=True)
            y = tf.sign(tf.reduce_max(actions_policy, axis=-1, keepdims=True) - actions_policy)
            y = (y - 1) * (-1)
            a_values = self.q_model(states_arr, training=True).numpy()[np.arange(64), policy_actions]
            a_values = tf.reshape(a_values, [-1, 1])
            a_loss = -1*tf.reduce_mean(y*a_values)

        p_grads = p_tape.gradient(a_loss, self.policy_model.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(p_grads, self.policy_model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())


        self.target_policy_model.set_weights(list([w1*(1-self.rho)+self.rho*w2 for w1, w2 in zip(self.target_policy_model.get_weights(), self.policy_model.get_weights())]))
        self.target_q_model.set_weights(list([w1*(1-self.rho)+self.rho*w2 for w1, w2 in zip(self.target_q_model.get_weights(), self.q_model.get_weights())]))

    def train(self, env, batch_size, n_episodes, exploration_phase_size, report_freq, save_freq, models_dir):
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
                model_path = os.path.join(models_dir, f'SNEK-ddpg-{e}-episodes.h5')
                self.save(model_path)

    def load(self, name):
        pass

    def save(self, name):
        pass
