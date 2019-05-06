import os
import random
import numpy as np
from dqn import DQNAgent
from reporter import Reporter
import snake_logger
from collections import deque
from tfNet import *
import tensorflow as tf
qLogger=snake_logger.QLogger()
mse = tf.keras.losses.MeanSquaredError()

class LoopKiller():
    def __init__(self):
        self.memory = set()

    def reset(self):
        self.memory = set()

    def observe(self, state):
        hashable_repr = str(state)
        self.memory.add(hashable_repr)

    def seen(self, state):
        hashable_repr = str(state)
        return hashable_repr in self.memory

class ConvACAgent(DQNAgent):
    def __init__(self, state_shape, action_size, num_last_observations, loss_logging=True, rho=0.9):
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_last_observations = num_last_observations
        self.observations = None
        self.epsilon_decay = None
        self.rho = rho

        self.memory = deque(maxlen=10**4)
        self.temp_memory = deque(maxlen=10**4)
        self.gamma = 0.9  # discount rate
        self.epsilon_max = 1  # epsilon == exploration rate
        self.epsilon_min = 0
        self.epsilon = self.epsilon_max
        self.q_learning_rate = 1
        self.q_model = QNet(num_last_observations, state_shape, self.action_size)
        # self.target_q_model = QNet(num_last_observations, state_shape, self.action_size)
        self.q_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
        self.policy_model = PolicyNet(num_last_observations, state_shape, self.action_size)
        # self.target_policy_model = PolicyNet(num_last_observations, state_shape, self.action_size)
        self.policy_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

    def remember(self, state, action, reward, next_state, done):
        state = self.reshape(state)
        next_state = self.reshape(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            a = random.randrange(self.action_size)
            return a

        act_values = self.policy_model(np.expand_dims(state, 0))
        act_values = act_values[0]
        a = np.random.choice(self.action_size, p=act_values)

        return a  # returns action

    def average_models(self, model1, model2):
        params1 = model1.named_parameters()
        params2 = model2.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(self.rho * param1.data + (1 - self.rho) * dict_params2[name1].data)

        model1.load_state_dict(dict_params2)

    def replay(self, batch_size):
        memory_sample = random.sample(self.memory, batch_size)
        np_memory_sample = np.array(memory_sample)

        states_arr = np.stack(np_memory_sample[:, 0]).astype(np.float64)
        actions_arr_np = np.stack(np_memory_sample[:, 1]).astype(np.int)
        rewards_arr = np_memory_sample[:, 2].astype(np.float64)
        next_states_arr = np.stack(np_memory_sample[:, 3]).astype(np.float64)
        done_arr = np_memory_sample[:, 4].astype(np.int)
        actions_next_prob = self.policy_model(next_states_arr)
        actions_next = [np.random.choice(self.action_size, p=act_values) for act_values in actions_next_prob]

        q_table = self.q_model(states_arr).numpy()
        q_table_next = self.q_model(next_states_arr).numpy()

        # Q-learning:
        # q_corrections = rewards_arr + self.gamma * np.amax(q_table_next, axis=1) * (1-done_arr)

        # Policy:
        q_corrections = rewards_arr + self.gamma * q_table_next[np.arange(actions_arr_np.shape[0]), actions_next] * (1 - done_arr)

        updated_qs_for_taken_actions = (
                                                   (1 - self.q_learning_rate) * q_table[np.arange(actions_arr_np.shape[0]), actions_arr_np] +
                                                   self.q_learning_rate * q_corrections)
        updated_qs_for_taken_actions[np.where(done_arr == 1)[0]] = -1  # when done just set -1

        q_table[np.arange(actions_arr_np.shape[0]), actions_arr_np] = updated_qs_for_taken_actions
        with tf.GradientTape() as tape:
            tape.watch(self.q_model.trainable_variables)
            qs = self.q_model(states_arr)
            q_loss = mse(y_pred=qs, y_true=q_table)

        q_grads = tape.gradient(q_loss, self.q_model.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_model.trainable_variables))

        policy_grads = policy_gradient(self.q_model, self.policy_model, states_arr)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables))

    def train(self, env, batch_size, n_episodes, exploration_phase_size, report_freq, save_freq, models_dir):
        reporter = Reporter(report_freq, n_episodes)
        # calc constant epsilon decay based on exploration_phase_size
        # (the percentage of the training process at which the exploration rate should reach its minimum)
        self.epsilon_decay = ((self.epsilon - self.epsilon_min) / (n_episodes * exploration_phase_size))
        n_observations = 0
        loopKiller = LoopKiller()
        for e in range(n_episodes):
            loopKiller.reset()
            self.observations = None
            observation = env.reset()
            loopKiller.observe(observation)
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
                    reward = 1
                    # reward = len(env.game.snake.body)
                elif loopKiller.seen(new_observation):
                    reward = -1
                    done = True
                else:
                    reward = 0

                loopKiller.observe(new_observation)
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
                for _ in range(1):
                    self.replay(batch_size)

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

            if e % save_freq == 0:
                model_path = os.path.join(models_dir, f'SNEK-AC-{e}-episodes.h5')
                self.save(model_path)

        model_path = os.path.join(models_dir, f'SNEK-AC-{n_episodes}-episodes.h5')
        self.save(model_path)

    def get_last_observations(self, observation):
        obs = super().get_last_observations(observation)
        return tf.transpose(obs, [1, 2, 0])

    def load(self, dir):
        pass
    #     nameQ = dir[:-3]+"-Q"+dir[-3:]
    #     nameP = dir[:-3]+"-P"+dir[-3:]
    #     if torch.cuda.is_available():
    #         self.q_model.load_state_dict(torch.load(nameQ))
    #         # self.q_model.eval()
    #         self.policy_model.load_state_dict(torch.load(nameP))
    #         # self.policy_model.eval()
    #     else:
    #         self.q_model.load_state_dict(torch.load(nameQ, map_location='cpu'))
    #         # self.q_model.eval()
    #         self.policy_model.load_state_dict(torch.load(nameP, map_location='cpu'))
    #         # self.policy_model.eval()
    #
    #
    def save(self, dir):
        pass
    #     nameQ = dir[:-3]+"-Q"+dir[-3:]
    #     nameP = dir[:-3]+"-P"+dir[-3:]
    #     torch.save(self.q_model.state_dict(), nameQ)
    #     torch.save(self.policy_model.state_dict(), nameP)
