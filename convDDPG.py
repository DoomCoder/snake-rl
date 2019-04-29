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

class ConvDDPGAgent(DQNAgent):
    def __init__(self, state_shape, action_size, num_last_observations, loss_logging=True, rho=0.9):
        # tf.enable_eager_execution()
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_last_observations = num_last_observations
        self.observations = None
        self.epsilon_decay = None
        self.rho = rho

        self.memory = deque(maxlen=10**4)
        self.temp_memory = deque(maxlen=10**4)
        self.gamma = 0.9  # discount rate
        self.epsilon_max = 0.01  # epsilon == exploration rate
        self.epsilon_min = 0.0
        self.epsilon = self.epsilon_max
        self.q_learning_rate = 1
        self.q_model = QNet(state_shape[1], self.action_size)
        self.target_q_model = QNet(state_shape[1], self.action_size)
        self.q_optimizer = torch.optim.RMSprop(self.q_model.parameters(), lr=1e-4)
        self.policy_model = PolicyNet(state_shape[1], self.action_size)
        self.target_policy_model = PolicyNet(state_shape[1], self.action_size)
        self.policy_optimizer = torch.optim.RMSprop(self.policy_model.parameters(), lr=1e-4)

        self.q_model.cuda()
        self.target_q_model.cuda()
        self.policy_model.cuda()
        self.target_policy_model.cuda()
        self.cuda = torch.device('cuda')
        # if loss_logging:
        #     self.model.train_on_batch = snake_logger.loss_logger_decorator(self.model.train_on_batch)

    def remember(self, state, action, reward, next_state, done):
        state = self.reshape(state)
        next_state = self.reshape(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            a = random.randrange(self.action_size)
            onehot_action = np.zeros(self.action_size)
            onehot_action[a] = 1
            return a, onehot_action

        act_values = self.policy_model(torch.from_numpy(np.expand_dims(state, 0)).float().to(self.cuda)).cpu()
        # act_values = F.softmax(act_values, dim=1).cpu()
        # act_values = self.q_model(torch.from_numpy(np.expand_dims(state, 0)).float().to(self.cuda)).cpu()
        pr = np.random.rand()
        # if pr <= 1e-4:
        #     print(act_values)

        act_values = act_values[0].detach().numpy()
        a = np.random.choice(self.action_size, p=act_values)
        # a = np.argmax(act_values)
        # if pr <= 1e-4:
        #     print(a)

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

        states_arr = np.stack(np_memory_sample[:, 0]).astype(np.float64)
        actions_arr_np = np.stack(np_memory_sample[:, 1]).astype(np.int)
        rewards_arr = np_memory_sample[:, 2].astype(np.float64)
        next_states_arr = np.stack(np_memory_sample[:, 3]).astype(np.float64)
        done_arr = np_memory_sample[:, 4].astype(np.int)
        next_states_arr = torch.from_numpy(next_states_arr).float().to(self.cuda)
        states_arr = torch.from_numpy(states_arr).float().to(self.cuda)

        actions = self.policy_model(states_arr)
        actions_next_prob = self.policy_model(next_states_arr).detach().cpu().numpy()
        actions_next = [np.random.choice(self.action_size, p=act_values) for act_values in actions_next_prob]

        q_table = self.q_model(states_arr).detach().cpu().numpy()
        q_table_next = self.q_model(next_states_arr).detach().cpu().numpy()

        # Q-learning:
        # q_corrections = rewards_arr + self.gamma * np.amax(q_table_next, axis=1) * (1-done_arr)

        # Policy:
        q_corrections = rewards_arr + self.gamma * q_table_next[np.arange(actions_arr_np.shape[0]),actions_next] * (1 - done_arr)

        updated_qs_for_taken_actions = (
                                                   (1 - self.q_learning_rate) * q_table[np.arange(actions_arr_np.shape[0]), actions_arr_np] +
                                                   self.q_learning_rate * q_corrections)
        updated_qs_for_taken_actions[np.where(done_arr == 1)[0]] = -1  # when done just set -1

        q_table[np.arange(actions_arr_np.shape[0]), actions_arr_np] = updated_qs_for_taken_actions
        qs = self.q_model(states_arr)
        pr = np.random.rand()
        # if pr <= 1e-3:
        #     print(np.linalg.norm(qs.detach().cpu().numpy()-q_table, ord=2))

        q_loss = 0.5*q_loss_fn(qs, torch.from_numpy(q_table).float().to(self.cuda))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        policy_loss = pi_loss_fn(self.q_model, states_arr, actions)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

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
                action, action_distribution = self.act(state)
                new_observation, has_eaten, done, _ = env.step(action)
                # rewards can be changed here
                if done:
                    reward = -1
                    # reward = -1 * len(env.game.snake.body)
                elif has_eaten:
                    reward = 1
                    # reward = len(env.game.snake.body)
                elif loopKiller.seen(new_observation):
                    reward = -1
                    # print("I helped!")
                    done = True
                else:
                    reward = 0

                loopKiller.observe(new_observation)
                reward_sum += reward
                next_state = self.get_last_observations(new_observation)
                onehot_action = np.zeros(self.action_size)
                onehot_action[action] = 1
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
                # model_path = os.path.join(models_dir, f'SNEK-ddpg-{e}-episodes.h5')
                self.save(models_dir, e)

        self.save(models_dir, n_episodes)

    def load(self, dir, e, nameQ=None, nameP=None):
        if nameQ is None:
            nameQ = f'SNEK-pg-2-Q-{e}-episodes.h5'
        if nameP is None:
            nameP = f'SNEK-pg-2-P-{e}-episodes.h5'

        self.q_model.load_state_dict(torch.load(os.path.join(dir, nameQ)))
        # self.q_model.eval()
        # model.eval()?
        self.policy_model.load_state_dict(torch.load(os.path.join(dir, nameP)))
        # self.policy_model.eval()

    def save(self, dir, e):
        torch.save(self.q_model.state_dict(), os.path.join(dir, f'SNEK-pg-2-Q-{e}-episodes.h5'))
        torch.save(self.policy_model.state_dict(), os.path.join(dir, f'SNEK-pg-2-P-{e}-episodes.h5'))
