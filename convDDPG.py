import os
import random
import numpy as np
from dqn import DQNAgent
from reporter import Reporter
import snake_logger
from collections import deque
from torchNet import *
from torch.distributions import Categorical, OneHotCategorical
qLogger=snake_logger.QLogger()


class ConvDDPGAgent(DQNAgent):
    def __init__(self, state_shape, action_size, num_last_observations, loss_logging=True, rho=1-1e-2):
        # tf.enable_eager_execution()
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_last_observations = num_last_observations
        self.observations = None
        self.epsilon_decay = None
        self.rho = rho

        print(state_shape[1])
        self.memory = deque(maxlen=10**4)
        self.gamma = 0.7  # discount rate
        self.epsilon_max = 0.99  # epsilon == exploration rate
        self.epsilon_min = 0.05
        self.epsilon = self.epsilon_max
        self.q_learning_rate = 0.1
        self.q_model = QNet(state_shape[1], self.action_size)
        self.target_q_model = QNet(state_shape[1], self.action_size)
        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=1e-4)
        self.policy_model = PolicyNet(state_shape[1], self.action_size)
        self.target_policy_model = PolicyNet(state_shape[1], self.action_size)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-4)

        self.q_model.cuda()
        self.target_q_model.cuda()
        self.policy_model.cuda()
        self.target_policy_model.cuda()
        self.cuda = torch.device('cuda')
        # if loss_logging:
        #     self.model.train_on_batch = snake_logger.loss_logger_decorator(self.model.train_on_batch)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a = random.randrange(self.action_size)
            onehot_action = np.zeros(self.action_size)
            onehot_action[a] = 1
            return a, onehot_action

        # state=tf.cast(state, tf.float32)
        # act_values = self.policy_model(torch.from_numpy(np.expand_dims(state, 0)).float().to(self.cuda)).cpu()
        # act_values = F.softmax(act_values, dim=1).cpu()
        act_values = self.q_model((torch.from_numpy(np.expand_dims(state, 0)).float().to(self.cuda),None)).cpu()
        if np.random.rand() <= 1e-2:
            print(act_values)

        act_values = act_values[0].detach().numpy()
        # a = np.random.choice(self.action_size, p=act_values)
        a = np.argmax(act_values)
        return a, act_values  # returns action

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
        actions_arr_np = np.stack(np_memory_sample[:, 1]).astype(np.int)
        rewards_arr = np_memory_sample[:, 2].astype(np.float64)
        next_states_arr = np.stack(np_memory_sample[:, 3]).astype(np.float64)
        done_arr = np_memory_sample[:, 4].astype(np.int)
        next_states_arr = torch.from_numpy(next_states_arr).float().to(self.cuda)
        states_arr = torch.from_numpy(states_arr).float().to(self.cuda)
        actions_arr = torch.from_numpy(actions_arr_np).float().to(self.cuda)
        actions_amax = torch.from_numpy(np.argmax(actions_arr_np,axis=1)).to(self.cuda)

        pred_actions = self.target_policy_model(next_states_arr).detach()
        # pred_actions = (pred_actions==torch.max(pred_actions)).float()

        q_table = self.target_q_model((
            next_states_arr,
            pred_actions))\
            .detach().cpu().numpy()
        done_exp = np.repeat((1-done_arr)[:, np.newaxis], self.action_size, axis=1)
        rewards_exp = np.repeat((1-rewards_arr)[:, np.newaxis], self.action_size, axis=1)
        ys = rewards_exp + self.gamma * q_table * done_exp  # * actions_arr_np
        ys = (1-self.q_learning_rate)*q_table[np.arange(actions_arr_np.shape[0])[:,np.newaxis], actions_arr_np] + self.q_learning_rate * ys # - (1-done_exp)

        q_table[np.arange(actions_arr_np.shape[0])[:,np.newaxis], actions_arr_np] = ys
        qs = self.q_model((states_arr, actions_arr))
        # if np.random.rand() <= 1e-2:
        #     print(qs)

        print("qs:")
        print(qs)

        print("qtab:")
        print(q_table)
        q_loss = q_loss_fn(qs, torch.from_numpy(q_table).float().to(self.cuda))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # policy_actions = self.policy_model(states_arr)
        # policy_loss = pi_loss_fn(self.q_model, states_arr, policy_actions, actions_amax, self.cuda)

        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()

        self.average_models(self.target_q_model, self.q_model)
        # self.average_models(self.target_policy_model, self.policy_model)

    def train(self, env, batch_size, n_episodes, exploration_phase_size, report_freq, save_freq, models_dir):
        reporter = Reporter(report_freq, n_episodes)
        # calc constant epsilon decay based on exploration_phase_size
        # (the percentage of the training process at which the exploration rate should reach its minimum)
        self.epsilon_decay = ((self.epsilon - self.epsilon_min) / (n_episodes * exploration_phase_size))
        n_observations = 0
        for e in range(n_episodes):
            self.observations = None
            observation = env.reset()
            observation = observation/4
            done = False
            steps = 0
            reward_sum = 0
            state = self.get_last_observations(observation)
            while not done:
                action, action_distribution = self.act(state)
                new_observation, has_eaten, done, _ = env.step(action)
                new_observation= new_observation/4
                # rewards can be changed here
                if done:
                    reward = 0
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
