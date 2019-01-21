from convDQN import ConvDQNAgent
import os
import random
import numpy as np
import snake_logger
from reporter import Reporter

qLogger=snake_logger.QLogger()

class ConvDQNTAgent(ConvDQNAgent):
    def __init__(self, *args, target_update_freq=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_model = self._build_model()
        self.target_update_freq = target_update_freq

    def save(self, name):
        self.model.save_weights(name)
        self.target_model.save_weights(name[:-3]+'_target'+name[-3:])

    def replay(self, batch_size):
        memory_sample = random.sample(self.memory, batch_size)
        np_memory_sample = np.array(memory_sample)

        states_arr = np.stack(np_memory_sample[:, 0])
        actions_arr = np_memory_sample[:, 1].astype(np.int)
        rewards_arr = np_memory_sample[:, 2].astype(np.float)
        next_states_arr = np.stack(np_memory_sample[:, 3])
        done_arr = np_memory_sample[:, 4].astype(int)

        q_corrections = rewards_arr + self.gamma * np.amax(self.target_model.predict(next_states_arr), axis=1) * (1-done_arr)
        q_table = self.model.predict(states_arr)
        qLogger.log(q_table)

        updated_qs_for_taken_actions = (
                (1 - self.q_learning_rate) * q_table[np.arange(actions_arr.shape[0]), actions_arr] +
                self.q_learning_rate * q_corrections
        )

        q_table[np.arange(actions_arr.shape[0]), actions_arr] = updated_qs_for_taken_actions
        self.model.train_on_batch(states_arr, q_table)


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
                model_path = os.path.join(models_dir, f'SNEK-dqnt-{e}-episodes.h5')
                self.save(model_path)

            if e % self.target_update_freq == 0:
                model_path = os.path.join(models_dir, f'SNEK-dqnt-update.h5')
                super().save(model_path)
                self.target_model.load_weights(model_path)