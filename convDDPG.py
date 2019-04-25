import os
import random
import numpy as np
from dqn import DQNAgent
from reporter import Reporter
import snake_logger
from collections import deque
from torchNet import *


qLogger=snake_logger.QLogger()


class ConvDDPGAgent(DQNAgent):
    def __init__(self, state_shape, action_size, num_last_observations, loss_logging=True, rho=1-1e-4):
        # tf.enable_eager_execution()
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_last_observations = num_last_observations
        self.observations = None
        self.epsilon_decay = None
        self.rho = rho

        print(state_shape[1])
        self.memory = deque(maxlen=10**4)
        self.gamma = 0.9  # discount rate
        self.epsilon_max = 1.0  # epsilon == exploration rate
        self.epsilon_min = 0.05
        self.epsilon = self.epsilon_max
        self.q_learning_rate = 0.1
        self.q_model = QNet(state_shape[1], self.action_size)
        self.target_q_model = QNet(state_shape[1], self.action_size)
        self.q_optimizer = torch.optim.SGD(self.q_model.parameters(), lr=1e-4)
        self.policy_model = PolicyNet(state_shape[1], self.action_size)
        self.target_policy_model = PolicyNet(state_shape[1], self.action_size)
        self.policy_optimizer = torch.optim.SGD(self.policy_model.parameters(), lr=1e-4)
        # if loss_logging:
        #     self.model.train_on_batch = snake_logger.loss_logger_decorator(self.model.train_on_batch)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # state=tf.cast(state, tf.float32)
        act_values = self.policy_model(torch.from_numpy(np.expand_dims(state, 0)).float())
        print(act_values)
        return np.argmax(act_values[0].detach().numpy())  # returns action

    def average_weights(self, weights):
        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(
                [np.array(weights_).mean(axis=0) \
                 for weights_ in zip(*weights_list_tuple)])

        return new_weights

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

        # s = torch.from_numpy(s).float()
        # a = torch.from_numpy(a).float()

        states_arr = np.stack(np_memory_sample[:, 0]).astype(np.float64)
        actions_arr = np.stack(np_memory_sample[:, 1]).astype(np.int)
        rewards_arr = np_memory_sample[:, 2].astype(np.float64)
        next_states_arr = np.stack(np_memory_sample[:, 3]).astype(np.float64)
        done_arr = np_memory_sample[:, 4].astype(np.int)
        next_states_arr = torch.from_numpy(next_states_arr).float()
        states_arr = torch.from_numpy(states_arr).float()
        actions_arr = torch.from_numpy(actions_arr).float()


        ys = rewards_arr + self.gamma * self.target_q_model((
            next_states_arr.float(),
            self.target_policy_model(next_states_arr).detach()))\
            .detach().numpy() * (1-done_arr)

        qs = self.q_model((states_arr, actions_arr
        ))

        q_loss = q_loss_fn(qs, torch.from_numpy(ys).float())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        policy_actions = self.policy_model(states_arr)
        policy_loss = pi_loss_fn(self.q_model, states_arr, policy_actions)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.average_models(self.q_model, self.target_q_model)
        self.average_models(self.policy_model, self.target_policy_model)

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
                onehot_action = np.zeros(self.action_size)
                onehot_action[action] = 1
                self.remember(state, onehot_action, reward, next_state, done)
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
